import { test, expect, Page } from "@playwright/test";
const oid = (n: number) => `10000000-0000-4000-8000-00000000000${n}`;
const cid = (n: number) => `60000000-0000-4000-8000-00000000000${n}`;
const base = {
  captured_at: "2026-11-03T01:15:30Z",
  profile_id: "profile-test",
  model_version: "fixture-only",
  expires_at: "2027-02-01T01:15:30Z",
  demo: true,
  image_content_path: "/api/v1/images/image-test/content",
};
function fixtures() {
  return [
    {
      ...base,
      observation_id: oid(1),
      event_id: "event-1",
      case_id: cid(1),
      case_version: 1,
      type: "pothole",
      location: { lat: 37.55, lon: 126.98, quality: "valid" },
      review: { status: "unreviewed", priority: "normal" },
      pothole: {
        confidence: 0.92,
        track_observations: 4,
        bbox_xywh_px: [10, 10, 40, 30],
      },
      lane: null,
    },
    {
      ...base,
      observation_id: oid(2),
      event_id: "event-2",
      case_id: cid(2),
      case_version: 3,
      type: "lane_segment",
      location: { lat: 37.552, lon: 126.984, quality: "valid" },
      review: { status: "confirmed", priority: "normal" },
      pothole: null,
      lane: {
        retention_pct: 55,
        grade: "worn_candidate",
        valid_area_ratio: 0.9,
        segment_support_ratio: 0.85,
        color: "white",
        line_type: "dashed",
        paint_mask_rle: { size: [2, 2], counts: [1, 2, 1] },
        outline_polygon_px: [
          [0, 0],
          [50, 0],
          [50, 50],
          [0, 50],
        ],
      },
    },
    {
      ...base,
      observation_id: oid(3),
      event_id: "event-3",
      case_id: cid(3),
      case_version: 1,
      type: "lane_segment",
      location: { lat: null, lon: null, quality: "unavailable" },
      review: { status: "unreviewed", priority: "normal" },
      pothole: null,
      lane: {
        retention_pct: null,
        grade: "unassessable",
        valid_area_ratio: null,
        segment_support_ratio: null,
        unassessable_reason: "missing_outline",
      },
    },
  ];
}
async function mock(page: Page, role = "admin") {
  const items = fixtures();
  const calls: { method: string; path: string; body: any }[] = [];
  let jobPolls = 0;
  await page.route("**/api/v1/**", async (route) => {
    const req = route.request(),
      u = new URL(req.url()),
      p = u.pathname.replace("/api/v1", ""),
      method = req.method();
    let body: any = {};
    try {
      body = req.postDataJSON();
    } catch {}
    calls.push({ method, path: p, body });
    const ok = (data: any, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify({ data, meta: { request_id: "ui-fixture" } }),
      });
    if (p === "/auth/login") {
      if (body.password === "wrong")
        return route.fulfill({
          status: 401,
          contentType: "application/json",
          body: JSON.stringify({
            error: {
              code: "UNAUTHENTICATED",
              message: "로그인 정보를 확인해 주세요.",
            },
          }),
        });
      return ok({
        access_token: "ui-fixture-token",
        expires_in: 1800,
        user: {
          id: "user-fixture",
          login: "admin@road.local",
          role,
          org_id: "demo",
        },
      });
    }
    if (p === "/auth/logout") return route.fulfill({ status: 204, body: "" });
    if (p === "/summary")
      return ok({
        total: 3,
        pending: 2,
        potholes: 1,
        worn_lanes: 1,
        completed: 0,
        unassessable: 1,
      });
    if (p === "/observations" && method === "GET") {
      let result = items;
      if (u.searchParams.get("type"))
        result = result.filter((x) => x.type === u.searchParams.get("type"));
      if (u.searchParams.get("review_status"))
        result = result.filter(
          (x) => x.review.status === u.searchParams.get("review_status"),
        );
      if (u.searchParams.get("quality"))
        result = result.filter(
          (x) => x.location.quality === u.searchParams.get("quality"),
        );
      return ok({
        items: result,
        next_cursor: null,
        counts_by_type: { pothole: 1, lane_segment: 2 },
      });
    }
    if (/^\/observations\/[^/]+\/review$/.test(p)) {
      const item = items.find((x) => p.includes(x.observation_id))!;
      item.review = { status: body.status, priority: body.priority };
      item.case_version++;
      return ok({
        case_id: item.case_id,
        version: item.case_version,
        review: item.review,
      });
    }
    if (/^\/observations\/[^/]+$/.test(p))
      return ok(items.find((x) => p.includes(x.observation_id)));
    if (p.endsWith("/history")) return ok({ entries: [] });
    if (p.endsWith("/content"))
      return route.fulfill({
        status: 200,
        contentType: "image/png",
        body: Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aFN0AAAAASUVORK5CYII=",
          "base64",
        ),
      });
    if (p === "/duplicate-links") return ok({ items: [] });
    if (p === "/reports")
      return ok({ job_id: "job-fixture", status: "pending" }, 202);
    if (p === "/jobs/job-fixture") {
      jobPolls++;
      return ok({
        job_id: "job-fixture",
        status: "succeeded",
        progress: 100,
        download_path: "/api/v1/reports/job-fixture/download",
      });
    }
    if (p.endsWith("/download"))
      return route.fulfill({
        status: 200,
        contentType: "text/csv",
        body: "type,status\npothole,confirmed\n",
      });
    if (p === "/system/status")
      return ok({
        services: [
          { name: "identity", status: "healthy" },
          { name: "inspection", status: "healthy" },
          { name: "media", status: "healthy" },
        ],
      });
    if (p === "/image-approvals")
      return ok({ approval_id: "approval-fixture" }, 201);
    if (p === "/images")
      return ok({ image_id: "uploaded-fixture", state: "ready" }, 201);
    return route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({
        error: { message: "Unknown UI test fixture route: " + p },
      }),
    });
  });
  return { calls, items };
}
async function login(page: Page) {
  await page.goto("/");
  await page.getByLabel("비밀번호", { exact: true }).fill("ui-test-password");
  await page.getByRole("button", { name: "로그인", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "점검 개요", exact: true }),
  ).toBeVisible();
}
test("late response from an older provenance filter cannot overwrite the selected scope", async ({
  page,
}) => {
  const state = await mock(page);
  await page.route("**/api/v1/observations?**", async (route) => {
    const scope = new URL(route.request().url()).searchParams.get("data_scope");
    if (scope === "presentation")
      await new Promise((resolve) => setTimeout(resolve, 600));
    const items =
      scope === "test"
        ? [{ ...state.items[0], demo: false, data_origin: "http_test" }]
        : state.items;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ data: { items, next_cursor: null }, meta: {} }),
    });
  });
  await login(page);
  await page.getByLabel("자료 출처 범위").selectOption("test");
  await expect(page.locator(".row-target")).toHaveCount(1);
  await expect(page.locator(".row-target")).toContainText("합성 HTTP 테스트");
  await page.waitForTimeout(750);
  await expect(page.getByLabel("자료 출처 범위")).toHaveValue("test");
  await expect(page.locator(".row-target")).toHaveCount(1);
  await expect(page.locator(".row-target")).toContainText("합성 HTTP 테스트");
});
test("desktop dashboard, filters, detail and real request shape", async ({
  page,
}, info) => {
  const state = await mock(page);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  // React StrictMode/unmount intentionally aborts obsolete image requests.
  // Every other failed request remains a test failure.
  page.on("requestfailed", (r) => {
    if (!r.failure()?.errorText.includes("ERR_ABORTED"))
      errors.push(`${r.url()}: ${r.failure()?.errorText}`);
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await login(page);
  await expect(page.getByText("현재 목록에 가상 시연 자료")).toBeVisible();
  await expect(page.locator(".observation-table tbody tr")).toHaveCount(3);
  await page.screenshot({
    path: info.outputPath("desktop-dashboard.png"),
    fullPage: true,
    animations: "disabled",
  });
  await page.getByRole("button", { name: "점검 목록", exact: true }).click();
  await page.getByLabel("관측 종류", { exact: true }).selectOption("pothole");
  await expect(page.locator(".observation-table tbody tr")).toHaveCount(1);
  await page.getByLabel("관측 검색").fill("does-not-exist");
  await expect(page.getByText("조건에 맞는 관측이 없습니다")).toBeVisible();
  await page
    .getByRole("button", { name: "필터 초기화", exact: true })
    .first()
    .click();
  await expect(page.locator(".observation-table tbody tr")).toHaveCount(3);
  await page.locator(".row-target").nth(1).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await dialog
    .getByLabel("검토 상태", { exact: true })
    .selectOption("action_planned");
  await dialog.getByLabel("변경 사유").fill("UI 계약 시험용 조치 계획");
  await dialog.getByRole("button", { name: "검토 결과 저장" }).click();
  await expect(dialog).toHaveCount(0);
  expect(
    state.calls.find((x) => x.method === "PATCH" && x.path.endsWith("/review"))
      ?.body,
  ).toMatchObject({
    expected_case_id: cid(2),
    expected_version: 3,
    status: "action_planned",
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});
test("mobile navigation and dialog keyboard focus remain usable", async ({
  page,
}, info) => {
  await mock(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await login(page);
  await expect(page.getByRole("button", { name: "메뉴 열기" })).toBeVisible();
  await page.getByRole("button", { name: "메뉴 열기" }).click();
  await page.getByRole("button", { name: "점검 목록", exact: true }).click();
  await page.locator(".row-target").first().click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "상세 닫기" }).focus();
  await page.keyboard.press("Shift+Tab");
  expect(
    await page.evaluate(
      () => !!document.activeElement?.closest('[role="dialog"]'),
    ),
  ).toBeTruthy();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await page.screenshot({
    path: info.outputPath("mobile-inspections.png"),
    fullPage: true,
    animations: "disabled",
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
});
test("reports poll and download without exposing a public asset URL", async ({
  page,
}) => {
  await mock(page);
  await login(page);
  await page.getByRole("button", { name: "보고서", exact: true }).click();
  await page.getByRole("button", { name: "보고서 생성", exact: true }).click();
  await expect(page.getByText("생성 대기", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "보고서 상태 새로고침" }).click();
  await expect(
    page.getByText("다운로드 준비 완료", { exact: true }),
  ).toBeVisible();
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "다운로드", exact: true }).click();
  expect((await pending).suggestedFilename()).toBe("road-insight.csv");
});
test("viewer cannot mutate an inspection", async ({ page }) => {
  await mock(page, "viewer");
  await login(page);
  await page.locator(".row-target").first().click();
  await expect(
    page.getByRole("button", { name: "검토 결과 저장" }),
  ).toBeDisabled();
  await expect(page.getByLabel("검토 상태", { exact: true })).toBeDisabled();
});
test("local media rejects corrupt data and supports keyboard redaction", async ({
  page,
}) => {
  const state = await mock(page);
  await login(page);
  await page
    .getByRole("button", { name: "사진 공유 검수", exact: true })
    .click();
  const file = page.locator("input[type=file]");
  await file.setInputFiles({
    name: "corrupt.png",
    mimeType: "image/png",
    buffer: Buffer.from("not-a-png"),
  });
  await expect(
    page.getByText("이미지를 읽지 못했습니다.", { exact: false }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "승인 및 전송" }),
  ).toBeDisabled();
  const png = await page.evaluate(() => {
    const c = document.createElement("canvas");
    c.width = 256;
    c.height = 128;
    c.getContext("2d")!.fillRect(0, 0, 256, 128);
    return c.toDataURL("image/png").split(",")[1];
  });
  await file.setInputFiles({
    name: "test.png",
    mimeType: "image/png",
    buffer: Buffer.from(png, "base64"),
  });
  await expect(
    page.getByRole("group", { name: "좌표로 가림 영역 추가", exact: false }),
  ).toBeVisible();
  await page.getByLabel("왼쪽 X", { exact: true }).fill("10");
  await page.getByLabel("위쪽 Y", { exact: true }).fill("10");
  await page.getByLabel("너비", { exact: true }).fill("20");
  await page.getByLabel("높이", { exact: true }).fill("20");
  await page
    .getByRole("button", { name: "가림 영역 추가", exact: true })
    .click();
  await expect(page.getByText("가림 영역 1개", { exact: false })).toBeVisible();
  await page.getByLabel("실제 촬영 시각").fill("2026-11-03T10:00");
  await page
    .getByLabel("연결할 기기 ID")
    .fill("40000000-0000-4000-8000-000000000001");
  await page.getByLabel("기기 전송 토큰").fill("fixture-device-token");
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "승인 및 전송" }).click();
  await expect(page.getByRole("status")).toContainText("완료");
  expect(
    state.calls.find((x) => x.path === "/image-approvals")?.body,
  ).toMatchObject({
    mime: "image/png",
    redaction_model_version: "manual-opaque-v1",
  });
});
test("API failure is visible and can be retried", async ({ page }) => {
  await mock(page);
  await page.route("**/api/v1/observations?**", (r) =>
    r.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ error: { message: "점검 서비스 일시 장애" } }),
    }),
  );
  await login(page);
  await expect(page.getByRole("alert")).toContainText("점검 서비스 일시 장애");
  await expect(page.getByRole("button", { name: "다시 시도" })).toBeVisible();
});

test("duplicate decisions require viewing both evidence images first", async ({
  page,
}) => {
  const state = await mock(page);
  let decided = false;
  const candidate = {
    id: "link-fixture",
    source_case_id: cid(1),
    target_case_id: cid(2),
    source_observation_id: oid(1),
    target_observation_id: oid(2),
    source_version: 1,
    target_version: 3,
    status: "suggested",
    distance_m: 4.2,
  };
  await page.route("**/api/v1/duplicate-links**", (route) => {
    if (route.request().method() === "POST") {
      decided = true;
      state.calls.push({
        method: "POST",
        path: "/duplicate-links",
        body: route.request().postDataJSON(),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ data: { items: decided ? [] : [candidate] } }),
    });
  });
  await page.route(`**/api/v1/observations/${oid(2)}`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        data: {
          ...state.items[0],
          observation_id: oid(2),
          case_id: cid(2),
          case_version: 3,
        },
      }),
    }),
  );
  await login(page);
  await page.getByRole("button", { name: "중복 후보", exact: true }).click();
  await page.getByLabel("검토 사유").fill("근접 관측");
  await expect(
    page.getByRole("button", { name: "동일 대상 연결" }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "사진·시각 비교" }).click();
  await expect(page.locator(".comparison .evidence-image img")).toHaveCount(2);
  await page
    .getByLabel("검토 사유")
    .fill("두 사진의 같은 결손과 촬영 방향을 확인");
  await page.getByRole("button", { name: "동일 대상 연결" }).click();
  await expect(page.getByText("검토할 중복 후보가 없습니다")).toBeVisible();
  expect(
    state.calls.find(
      (x) => x.path === "/duplicate-links" && x.method === "POST",
    )?.body,
  ).toMatchObject({
    decision: "approved",
    source_version: 1,
    target_version: 3,
  });
});

test("evidence overlays render actual boxes and optional lane RLE without inventing absent masks", async ({
  page,
}) => {
  await mock(page);
  await login(page);
  await page.getByRole("button", { name: "점검 목록", exact: true }).click();
  await page.locator(".row-target").first().click();
  await expect(page.locator(".evidence-overlay rect")).toHaveAttribute(
    "width",
    "40",
  );
  await page.getByLabel("검출 박스·도막·외곽 표시").uncheck();
  await expect(page.locator(".evidence-overlay")).toHaveCount(0);
  await page.getByRole("button", { name: "상세 닫기" }).click();
  await page.locator(".row-target").nth(1).click();
  await expect(page.locator(".evidence-overlay image")).toHaveCount(1);
  await expect(page.locator(".evidence-overlay polygon")).toHaveCount(1);
  await page.getByRole("button", { name: "상세 닫기" }).click();
  await page.locator(".row-target").nth(2).click();
  await expect(
    page.getByText("도막 픽셀 마스크 미제공", { exact: false }),
  ).toBeVisible();
});
