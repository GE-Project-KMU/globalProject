import { test, expect } from "@playwright/test";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

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
    await dialog
      .getByLabel("변경 사유")
      .fill("브라우저 통합 시험: 현재 상태 유지 확인");
    await dialog.getByRole("button", { name: "검토 결과 저장" }).click();
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
