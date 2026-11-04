import { test, expect } from "@playwright/test";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const envFile = fileURLToPath(new URL("../../.env", import.meta.url));
const settings: Record<string, string> = {};
if (existsSync(envFile))
  for (const line of readFileSync(envFile, "utf8").split(/\r?\n/)) {
    const m = line.match(/^([A-Z0-9_]+)=(.*)$/);
    if (m) settings[m[1]] = m[2].trim().replace(/^['"]|['"]$/g, "");
  }
const password =
  process.env.ROAD_BOOTSTRAP_PASSWORD || settings.ROAD_BOOTSTRAP_PASSWORD;

test("live gateway: login, inspection review, reports, service health and responsive layout", async ({
  page,
}, info) => {
  test.skip(
    !password,
    "Live integration requires the project bootstrap password in .env.",
  );
  const consoleErrors: string[] = [],
    pageErrors: string[] = [],
    failedRequests: string[] = [],
    httpErrors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => pageErrors.push(e.message));
  page.on("requestfailed", (r) => {
    if (!r.failure()?.errorText.includes("ERR_ABORTED"))
      failedRequests.push(
        `${r.method()} ${r.url()}: ${r.failure()?.errorText}`,
      );
  });
  page.on("response", (r) => {
    if (r.url().includes("/api/") && r.status() >= 400)
      httpErrors.push(`${r.status()} ${new URL(r.url()).pathname}`);
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/");
  await page.getByLabel("비밀번호", { exact: true }).fill(password!);
  await page.getByRole("button", { name: "로그인", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "점검 개요", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".loading")).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await page.screenshot({
    path: info.outputPath("live-desktop-dashboard.png"),
    fullPage: true,
    animations: "disabled",
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  const rows = page.locator(".row-target");
  if (await rows.count()) {
    await rows.first().click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    await expect(dialog.locator(".evidence-image img")).toBeVisible();
    await expect(dialog.getByLabel("변경 사유")).toBeVisible();
    await dialog.getByRole("button", { name: "상세 닫기" }).click();
    await expect(dialog).toHaveCount(0);
  }
  await page.getByRole("button", { name: "서비스 상태", exact: true }).click();
  await expect(page.locator(".service-card")).not.toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await page.screenshot({
    path: info.outputPath("live-service-status.png"),
    fullPage: true,
    animations: "disabled",
  });
  await page.getByRole("button", { name: "보고서", exact: true }).click();
  await page.getByRole("button", { name: "보고서 생성", exact: true }).click();
  await expect(page.locator(".job-card")).toBeVisible();
  for (
    let i = 0;
    i < 10 &&
    !(await page
      .getByRole("button", { name: "다운로드", exact: true })
      .count());
    i++
  ) {
    await page.getByRole("button", { name: "보고서 상태 새로고침" }).click();
    await page.waitForTimeout(250);
  }
  await expect(
    page.getByRole("button", { name: "다운로드", exact: true }),
  ).toBeVisible();
  const pendingDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "다운로드", exact: true }).click();
  expect((await pendingDownload).suggestedFilename()).toContain(".csv");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "메뉴 열기" }).click();
  await page.getByRole("button", { name: "점검 목록", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "점검 목록", exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: info.outputPath("live-mobile-inspections.png"),
    fullPage: true,
    animations: "disabled",
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await info.attach("live-browser-diagnostics", {
    body: JSON.stringify(
      { consoleErrors, pageErrors, failedRequests, httpErrors },
      null,
      2,
    ),
    contentType: "application/json",
  });
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(failedRequests).toEqual([]);
  expect(httpErrors).toEqual([]);
});

test("live management operations and privacy access use real services", async ({
  page,
  request,
}, info) => {
  test.setTimeout(60000);
  test.skip(!password, "Project bootstrap password required");
  const root = fileURLToPath(new URL("../../", import.meta.url));
  const isolated = JSON.parse(
    execFileSync(
      fileURLToPath(
        new URL(
          process.platform === "win32"
            ? "../../.venv/Scripts/python.exe"
            : "../../.venv/bin/python",
          import.meta.url,
        ),
      ),
      ["scripts/bootstrap_browser_fixture.py"],
      {
        cwd: root,
        encoding: "utf8",
        env: {
          ...process.env,
          PYTHONPYCACHEPREFIX: fileURLToPath(
            new URL("../../.cache/pycache", import.meta.url),
          ),
        },
      },
    ),
  );
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("requestfailed", (r) => {
    if (!r.failure()?.errorText.includes("ERR_ABORTED"))
      errors.push(r.failure()?.errorText || "requestfailed");
  });
  page.on("response", (r) => {
    if (r.url().includes("/api/") && r.status() >= 400)
      errors.push(`${r.status()} ${new URL(r.url()).pathname}`);
  });
  const auth = await request.post("http://127.0.0.1:8000/api/v1/auth/login", {
    data: {
      ...isolated,
    },
  });
  expect(auth.ok()).toBeTruthy();
  const token = (await auth.json()).data.access_token;
  const headers = { Authorization: `Bearer ${token}` };
  const observations = (
    await (
      await request.get("http://127.0.0.1:8000/api/v1/observations?limit=200", {
        headers,
      })
    ).json()
  ).data.items;
  const demo = observations.find((x: any) => x.demo);
  expect(demo).toBeTruthy();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/");
  await page.getByLabel("기관 코드").fill(isolated.org_code);
  await page.getByLabel("로그인 ID").fill(isolated.login);
  await page.getByLabel("비밀번호", { exact: true }).fill(isolated.password);
  await page.getByRole("button", { name: "로그인", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "점검 개요", level: 1 }),
  ).toBeVisible();
  const nav = async (name: string) => {
    await page
      .getByRole("navigation")
      .getByRole("button", { name, exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name, level: 1, exact: true }),
    ).toBeVisible();
  };
  const unique = Date.now();
  await nav("계정 관리");
  await page
    .getByLabel("로그인 식별자")
    .fill(`browser-${unique}@synthetic.test`);
  await page.getByLabel("초기 비밀번호").fill(`synthetic-browser-${unique}`);
  await page.getByRole("button", { name: "계정 저장" }).click();
  const user = page.locator("article").filter({
    has: page.getByRole("heading", {
      name: `browser-${unique}@synthetic.test`,
      exact: true,
    }),
  });
  await expect(user).toBeVisible();
  await user.getByRole("button", { name: "계정 편집" }).click();
  await page.getByLabel("계정 활성", { exact: true }).uncheck();
  await page.getByRole("button", { name: "계정 저장" }).click();
  await expect(user).toContainText("비활성");
  await nav("수집 기기");
  await page.getByLabel("기기 이름").fill(`가상 브라우저 시험 ${unique}`);
  await page.getByRole("button", { name: "기기 저장" }).click();
  await expect(page.getByText("기기 토큰 · 이번에만 표시됩니다")).toBeVisible();
  await page.getByRole("button", { name: "토큰 화면에서 지우기" }).click();
  const device = page.locator("article").filter({
    has: page.getByRole("heading", {
      name: `가상 브라우저 시험 ${unique}`,
      exact: true,
    }),
  });
  await device.getByRole("button", { name: "기기 편집" }).click();
  await page.getByLabel("교체 사유").fill("가상 브라우저 통합 시험");
  await page.getByRole("button", { name: "새 토큰 발급" }).click();
  await expect(page.getByText("기기 토큰 · 이번에만 표시됩니다")).toBeVisible();
  await page.getByRole("button", { name: "토큰 화면에서 지우기" }).click();
  await device.getByRole("button", { name: "기기 편집" }).click();
  await page.getByLabel("기기 활성").uncheck();
  await page.getByRole("button", { name: "기기 저장" }).click();
  await expect(device).toContainText("비활성");
  for (const name of [
    "장착 프로파일",
    "모델 버전",
    "분석 운영",
    "감사기록",
    "주행 기록",
  ]) {
    await nav(name);
    await expect(page.getByRole("alert")).toHaveCount(0);
  }
  const runButton = page
    .getByRole("button", { name: "주행 상세", exact: true })
    .first();
  if (await runButton.count()) {
    await runButton.click();
    await expect(
      page.getByRole("heading", { name: "주행 상세", exact: true }),
    ).toBeVisible();
  }
  await nav("개인정보 요청");
  await page.getByLabel("대상 UUID").fill(demo.observation_id);
  await page.getByLabel("요청 사유").fill(`가상 자료 열람 시험 ${unique}`);
  await page.getByRole("checkbox").nth(0).check();
  await page.getByRole("checkbox").nth(1).check();
  await page.getByRole("button", { name: "요청 접수", exact: true }).click();
  await expect(
    page.getByText("요청을 접수했습니다.", { exact: false }),
  ).toBeVisible();
  const rows = (
    await (
      await request.get("http://127.0.0.1:8000/api/v1/privacy-requests", {
        headers,
      })
    ).json()
  ).data.items;
  const own = rows.find(
    (x: any) => x.reason === `가상 자료 열람 시험 ${unique}`,
  );
  expect(own).toBeTruthy();
  const row = page
    .locator("article")
    .filter({ has: page.getByText(own.request_id, { exact: true }) });
  await expect
    .poll(
      async () =>
        (
          await (
            await request.get(
              "http://127.0.0.1:8000/api/v1/privacy-requests/" + own.request_id,
              { headers },
            )
          ).json()
        ).data.status,
    )
    .toBe("in_progress");
  await row.getByRole("button", { name: "처리 내역" }).click();
  await page.getByLabel("처리 근거").fill("가상 자료 대상 및 권한 확인");
  await page.getByRole("button", { name: "처리 기록 저장" }).click();
  await expect(
    page.getByRole("button", { name: "승인된 열람 자료 다운로드" }),
  ).toBeVisible();
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: "승인된 열람 자료 다운로드" }).click();
  await downloadEvent;
  await page.screenshot({
    path: info.outputPath("live-management-desktop.png"),
    fullPage: true,
    animations: "disabled",
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: info.outputPath("live-management-mobile.png"),
    fullPage: true,
    animations: "disabled",
  });
  expect(errors).toEqual([]);
  await request.post("http://127.0.0.1:8000/api/v1/auth/logout", { headers });
});
