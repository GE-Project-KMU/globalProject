import { test, expect, Page } from "@playwright/test";
import { readFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { createHash, randomUUID } from "node:crypto";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../../", import.meta.url));
const settings: Record<string, string> = {};
for (const line of readFileSync(root + ".env", "utf8").split(/\r?\n/)) {
  const m = line.match(/^([A-Z_]+)=(.*)$/);
  if (m) settings[m[1]] = m[2].trim().replace(/^['"]|['"]$/g, "");
}
const packPath = root + "data/demo-reference/road-reference-pack.json";
const python = fileURLToPath(
  new URL(
    process.platform === "win32"
      ? "../../.venv/Scripts/python.exe"
      : "../../.venv/bin/python",
    import.meta.url,
  ),
);
function isolatedFixture() {
  return JSON.parse(
    execFileSync(python, ["scripts/bootstrap_browser_fixture.py"], {
      cwd: root,
      encoding: "utf8",
      env: { ...process.env, PYTHONPYCACHEPREFIX: root + ".cache/pycache" },
    }),
  );
}
async function login(
  page: Page,
  identity?: { org_code: string; login: string; password: string },
) {
  await page.goto("/");
  if (identity) {
    await page.getByLabel("기관 코드").fill(identity.org_code);
    await page.getByLabel("이메일 또는 로그인 ID").fill(identity.login);
  }
  await page
    .getByLabel("비밀번호", { exact: true })
    .fill(identity?.password || settings.ROAD_BOOTSTRAP_PASSWORD);
  await page.getByRole("button", { name: "로그인", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "점검 개요", level: 1, exact: true }),
  ).toBeVisible();
}
function diagnostics(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (e) => {
    if (e.type() === "error") errors.push(e.text());
  });
  page.on("requestfailed", (r) => {
    if (!r.failure()?.errorText.includes("ERR_ABORTED"))
      errors.push(r.failure()?.errorText || "failed");
  });
  page.on("response", (r) => {
    if (r.url().includes("/api/") && r.status() >= 400)
      errors.push(String(r.status()));
  });
  return errors;
}

test("isolated HTTP photos are excluded by default and keep truthful badges on read-only viewing", async ({
  page,
  request,
}, info) => {
  const errors = diagnostics(page);
  await login(page, isolatedFixture());
  await expect(page.getByLabel("자료 출처 범위")).toHaveValue("presentation");
  await expect(
    page.locator(".row-target em").filter({ hasText: "합성 HTTP 테스트" }),
  ).toHaveCount(0);
  await page.getByLabel("자료 출처 범위").selectOption("test");
  const first = page.locator(".row-target").first();
  await expect(first).toContainText("합성 HTTP 테스트");
  await first.click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toContainText("합성 HTTP 테스트입니다");
  const image = dialog.locator(".evidence-image img");
  await expect(image).toBeVisible();
  expect(
    await image.evaluate(
      (e: HTMLImageElement) => e.complete && e.naturalWidth > 0,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: info.outputPath("isolated-http-synthetic-badge.png"),
    fullPage: true,
  });
  await dialog.getByRole("button", { name: "상세 닫기" }).click();
  await page.getByLabel("자료 출처 범위").selectOption("presentation");
  await expect(
    page.locator(".row-target em").filter({ hasText: "합성 HTTP 테스트" }),
  ).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("real road references stay local, require user acknowledgement, and show no invented inference", async ({
  page,
}, info) => {
  test.skip(
    !existsSync(packPath),
    "Actual RTK source pack is local-only; CI must not download or redistribute research photographs.",
  );
  const evidenceRelative =
    "data/demo-reference/browser-evidence/" + randomUUID();
  const evidence = root + evidenceRelative;
  mkdirSync(evidence, { recursive: true });
  await page.setViewportSize({ width: 1280, height: 720 });
  const errors = diagnostics(page);
  const writes: string[] = [];
  page.on("request", (r) => {
    if (
      r.url().includes("/api/") &&
      !["GET", "HEAD"].includes(r.method()) &&
      !r.url().endsWith("/auth/login")
    )
      writes.push(r.method() + " " + r.url());
  });
  await login(page);
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "로컬 참고사진", exact: true })
    .click();
  await page
    .getByLabel("로컬 참고사진 묶음", { exact: true })
    .setInputFiles(packPath);
  await expect(page.locator(".reference-grid img")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "참고사진 로컬에서 열기" }),
  ).toBeDisabled();
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "참고사진 로컬에서 열기" }).click();
  await expect(page.locator(".reference-grid img")).toHaveCount(3);
  for (const img of await page.locator(".reference-grid img").all())
    expect(
      await img.evaluate(
        (e: HTMLImageElement) =>
          e.complete && e.naturalWidth === 160 && e.naturalHeight === 64,
      ),
    ).toBeTruthy();
  await expect(page.locator(".reference-grid")).toContainText(
    "검수 미완료 · AI 검출 아님",
  );
  await expect(page.locator("footer")).toContainText(
    "공유 승인 없음 · AI 분석 미실행",
  );
  async function stableTop() {
    await page.evaluate(() =>
      window.scrollTo({ top: 0, left: 0, behavior: "instant" }),
    );
    await expect
      .poll(() =>
        page.evaluate(() => ({ x: window.scrollX, y: window.scrollY })),
      )
      .toEqual({ x: 0, y: 0 });
    await page.evaluate(
      () =>
        new Promise<void>((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
        ),
    );
  }
  async function sidebarGeometry() {
    return page.locator(".sidebar").evaluate((el) => {
      const r = el.getBoundingClientRect();
      return {
        x: r.x,
        y: r.y,
        right: r.right,
        width: r.width,
        height: r.height,
        transform: getComputedStyle(el).transform,
        runningAnimations: el
          .getAnimations()
          .filter((a) => a.playState === "running").length,
      };
    });
  }
  async function mainHitTest() {
    const heading = page.getByRole("heading", {
      name: "실노면 로컬 참고사진",
      exact: true,
    });
    const hit = await heading.evaluate((el) => {
      const r = el.getBoundingClientRect();
      const top = document.elementFromPoint(
        r.left + r.width / 2,
        r.top + r.height / 2,
      );
      return {
        reachable: !!top && (top === el || el.contains(top)),
        targetTag: top?.tagName,
      };
    });
    expect(hit.reachable).toBeTruthy();
    await heading.evaluate((el) =>
      el.addEventListener(
        "click",
        () => el.setAttribute("data-qa-main-click", "received"),
        { once: true },
      ),
    );
    await heading.click();
    await expect(heading).toHaveAttribute("data-qa-main-click", "received");
    return hit;
  }
  const priorDesktopCaptureState = await page.evaluate(() => ({
    scrollY: window.scrollY,
    sidebarY: document.querySelector(".sidebar")!.getBoundingClientRect().y,
    viewportHeight: innerHeight,
  }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await stableTop();
  const desktopSidebar = await sidebarGeometry();
  expect(desktopSidebar.x).toBe(0);
  expect(desktopSidebar.y).toBe(0);
  const desktopHit = await mainHitTest();
  await stableTop();
  await page.screenshot({
    path: evidence + "/PRIVATE-local-reference-desktop.png",
    fullPage: false,
    animations: "disabled",
  });
  await page.screenshot({
    path: evidence + "/PRIVATE-local-reference-desktop-fullpage.png",
    fullPage: true,
    animations: "disabled",
  });
  await page.setViewportSize({ width: 390, height: 844 });
  const resizeTransitionState = await sidebarGeometry();
  const menu = page.getByRole("button", { name: "메뉴 열기", exact: true });
  await expect(menu).toHaveAttribute("aria-expanded", "false");
  await expect
    .poll(async () => (await sidebarGeometry()).right)
    .toBeLessThanOrEqual(0.5);
  await menu.click();
  await expect(menu).toHaveAttribute("aria-expanded", "true");
  await expect
    .poll(async () => Math.abs((await sidebarGeometry()).x))
    .toBeLessThan(0.5);
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "로컬 참고사진", exact: true })
    .click();
  await expect(menu).toHaveAttribute("aria-expanded", "false");
  await expect(page.locator(".sidebar")).toHaveAttribute("aria-hidden", "true");
  await expect(page.locator(".sidebar")).toHaveAttribute("inert", "");
  await expect(page.locator(".mobile-shade")).toHaveCount(0);
  await expect
    .poll(async () => (await sidebarGeometry()).right)
    .toBeLessThanOrEqual(0.5);
  await expect
    .poll(async () => (await sidebarGeometry()).runningAnimations)
    .toBe(0);
  await stableTop();
  const mobileSidebar = await sidebarGeometry();
  const mobileHit = await mainHitTest();
  await stableTop();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: evidence + "/PRIVATE-local-reference-mobile.png",
    fullPage: false,
    animations: "disabled",
  });
  await page.screenshot({
    path: evidence + "/PRIVATE-local-reference-mobile-fullpage.png",
    fullPage: true,
    animations: "disabled",
  });
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
  writeFileSync(
    evidence + "/private-evidence.json",
    JSON.stringify(
      {
        expires_at: JSON.parse(readFileSync(packPath, "utf8")).expires_at,
        source_pack_sha256: createHash("sha256")
          .update(readFileSync(packPath))
          .digest("hex"),
        local_only: true,
        human_privacy_reviewed: false,
        inference_performed: false,
        screenshot_qa: {
          priorDesktopCaptureState,
          resizeTransitionState,
          desktop: {
            sidebar: desktopSidebar,
            mainHit: desktopHit,
            scrollTop: 0,
          },
          mobile: {
            sidebar: mobileSidebar,
            mainHit: mobileHit,
            menuClosed: true,
            shadeCount: 0,
            scrollTop: 0,
          },
          animations_disabled: true,
        },
        images: [
          { path: evidenceRelative + "/PRIVATE-local-reference-desktop.png" },
          {
            path:
              evidenceRelative +
              "/PRIVATE-local-reference-desktop-fullpage.png",
          },
          { path: evidenceRelative + "/PRIVATE-local-reference-mobile.png" },
          {
            path:
              evidenceRelative + "/PRIVATE-local-reference-mobile-fullpage.png",
          },
        ],
      },
      null,
      2,
    ),
  );
});

test("malicious or expired reference packs cannot fetch remote photos or bypass local-only review", async ({
  page,
}) => {
  const errors = diagnostics(page);
  const external: string[] = [];
  page.on("request", (r) => {
    if (/^https?:/.test(r.url()) && !r.url().startsWith("http://127.0.0.1:"))
      external.push(r.url());
  });
  await login(page);
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "로컬 참고사진", exact: true })
    .click();
  const original = JSON.parse(
    execFileSync(python, ["scripts/reference_preview_fixture.py"], {
      cwd: root,
      encoding: "utf8",
      env: { ...process.env, PYTHONPYCACHEPREFIX: root + ".cache/pycache" },
    }),
  );
  const cases = [
    (p: any) => {
      p.images[0].data_url = "https://example.invalid/private.jpg";
    },
    (p: any) => {
      p.images[0].data_url = "C:/outside/private.jpg";
    },
    (p: any) => {
      p.images = Array(7).fill(p.images[0]);
    },
    (p: any) => {
      p.extra_path = "data/private.jpg";
    },
    (p: any) => {
      p.expires_at = "2000-01-01T00:00:00Z";
    },
    (p: any) => {
      p.expires_at = p.source_expires_at = "2099-01-01T00:00:00Z";
    },
    (p: any) => {
      p.human_privacy_reviewed = true;
    },
    (p: any) => {
      p.source_expires_at = "2027-02-01";
    },
    (p: any) => {
      p.images[0].width = 4096;
      p.images[0].height = 4096;
    },
  ];
  for (const change of cases) {
    const pack = structuredClone(original);
    change(pack);
    await page.getByLabel("로컬 참고사진 묶음", { exact: true }).setInputFiles({
      name: "invalid.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(pack)),
    });
    await expect(page.getByRole("alert")).toBeVisible();
    await expect(page.locator(".reference-grid img")).toHaveCount(0);
  }
  await page.getByLabel("로컬 참고사진 묶음", { exact: true }).setInputFiles({
    name: "oversize.json",
    mimeType: "application/json",
    buffer: Buffer.alloc(5_000_001, 32),
  });
  await expect(page.getByRole("alert")).toContainText("5MB");
  const short = structuredClone(original);
  short.expires_at = new Date(Date.now() + 3000).toISOString();
  await page.getByLabel("로컬 참고사진 묶음", { exact: true }).setInputFiles({
    name: "short.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(short)),
  });
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "참고사진 로컬에서 열기" }).click();
  await expect(page.locator(".reference-grid img")).toHaveCount(3);
  await expect(page.getByRole("alert")).toContainText("보관 기한이 지나", {
    timeout: 7000,
  });
  await expect(page.locator(".reference-grid img")).toHaveCount(0);
  expect(external).toEqual([]);
  expect(errors).toEqual([]);
});
