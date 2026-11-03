import { test, expect, Page } from "@playwright/test";
const target = "10000000-0000-4000-8000-000000000001";
async function setup(page: Page, role = "admin") {
  const calls: any[] = [];
  let users = [
    { id: "u1", login: "manager@test.local", role: "reviewer", active: true },
  ];
  let devices = [{ id: "d1", name: "수집 차량 A", active: true }];
  let privacy: any[] = [];
  let profiles: any[] = [];
  let bundles: any[] = [];
  await page.route("**/api/v1/**", async (route) => {
    const req = route.request(),
      u = new URL(req.url()),
      p = u.pathname.replace("/api/v1", ""),
      method = req.method();
    let body: any = {};
    try {
      body = req.postDataJSON();
    } catch {}
    calls.push({ p, method, body, query: u.search });
    const ok = (data: any, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify({ data }),
      });
    if (p === "/auth/login")
      return ok({
        access_token: "fixture",
        user: { id: "self", role, login: "admin@road.local" },
      });
    if (p === "/summary")
      return ok({
        total: 0,
        pending: 0,
        potholes: 0,
        worn_lanes: 0,
        completed: 0,
        unassessable: 0,
        trend_by_type: {
          pothole: [
            { date: "2026-11-03", candidates: 3, confirmed: 2, completed: 1 },
          ],
          lane_segment: [
            { date: "2026-11-03", candidates: 7, confirmed: 4, completed: 2 },
          ],
        },
      });
    if (p === "/observations") return ok({ items: [], next_cursor: null });
    if (p === "/users") {
      if (method === "POST") users.push({ id: "new", ...body, active: true });
      return ok(method === "GET" ? { items: users } : users.at(-1));
    }
    if (p === "/users/u1") {
      users[0] = { ...users[0], ...body };
      return ok(users[0]);
    }
    if (p === "/devices") {
      if (method === "POST") {
        devices.push({ id: "new-device", ...body, active: true });
        return ok({
          device_id: "new-device",
          device_token: "fixture-token-one-time",
        });
      }
      return ok({ items: devices });
    }
    if (p === "/devices/d1/credentials")
      return ok({ device_token: "fixture-rotated-token" });
    if (p === "/devices/d1") {
      devices[0] = { ...devices[0], ...body };
      return ok(devices[0]);
    }
    if (p === "/profiles") {
      if (method === "POST") profiles.push({ id: "p1", ...body });
      return ok(method === "GET" ? { items: profiles } : profiles.at(-1));
    }
    if (p === "/model-bundles") {
      if (method === "POST") bundles.push({ id: "b1", ...body });
      return ok(method === "GET" ? { items: bundles } : bundles.at(-1));
    }
    if (p === "/privacy-requests") {
      if (method === "POST") {
        privacy.push({
          request_id: "r" + privacy.length,
          version: 1,
          status: "in_progress",
          result_code: "MANUAL_ADMIN_REVIEW_REQUIRED",
          ...body,
        });
        return ok(privacy.at(-1), 202);
      }
      return ok({ items: privacy });
    }
    if (p.startsWith("/privacy-requests/")) {
      const x = privacy.find((x) => x.request_id === p.split("/")[2]);
      if (p.endsWith("/download"))
        return route.fulfill({
          contentType: "application/json",
          body: "{}",
          headers: {
            "Content-Disposition": 'attachment; filename="privacy.json"',
          },
        });
      if (method === "PATCH")
        Object.assign(x, {
          version: 2,
          status: "completed",
          result_code:
            body.action === "approve_access"
              ? "ACCESS_APPROVED"
              : "CORRECTION_APPLIED",
          decision_reason: body.reason,
          ...(body.action === "approve_access"
            ? {
                download_path: `/api/v1/privacy-requests/${x.request_id}/download`,
              }
            : {}),
        });
      return ok(x);
    }
    if (p === "/runs")
      return ok({
        items: [
          {
            id: "run1",
            status: "completed",
            started_at: "2026-11-03T01:00:00Z",
          },
        ],
      });
    if (p === "/runs/run1")
      return ok({
        id: "run1",
        status: "completed",
        device_id: "d1",
        profile_id: "p1",
        segments: [
          {
            coordinates: [
              [127, 37],
              [127.01, 37.01],
            ],
            quality: "valid",
            visible_scope: { assessment_state: "assessed" },
          },
          {
            coordinates: [
              [127.1, 37.1],
              [127.11, 37.11],
            ],
            quality: "degraded",
          },
        ],
      });
    if (p === "/audit-logs")
      return ok({
        items: [
          {
            id: "a1",
            action: "device.rotate",
            actor_hash: "actor-hash",
            target_hash: "target-hash",
            at: "2026-11-03T01:00:00Z",
            service: "identity",
          },
        ],
      });
    if (p === "/analysis/capabilities")
      return ok({
        raw_processing: "local_agent",
        server_raw_upload: false,
        manifest_count: 2,
        model_status: "target_weights_required",
      });
    if (p === "/analysis/jobs")
      return method === "POST"
        ? ok({ id: "job-fixture", ...body, status: "queued" }, 202)
        : ok({ items: [] });
    if (p === "/analysis/jobs/job-fixture")
      return ok({
        id: "job-fixture",
        status: "blocked",
        reason: "MODEL_NOT_READY",
      });
    return ok({});
  });
  await page.goto("/");
  await page
    .getByLabel("비밀번호", { exact: true })
    .fill("fixture-password-2026");
  await page.getByRole("button", { name: "로그인", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "점검 개요", exact: true }),
  ).toBeVisible();
  return calls;
}
async function nav(page: Page, name: string) {
  await page
    .getByRole("navigation")
    .getByRole("button", { name, exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name, exact: true, level: 1 }),
  ).toBeVisible();
}
test("management accounts, devices and immutable registries use actual API bodies", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const calls = await setup(page);
  await nav(page, "계정 관리");
  await page.getByLabel("로그인 식별자").fill("new@test.local");
  await page.getByLabel("초기 비밀번호").fill("valid-password-2026");
  await page.getByRole("button", { name: "계정 저장" }).click();
  await expect(
    page.getByRole("heading", { name: "new@test.local" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "계정 편집" }).first().click();
  await page
    .getByRole("combobox", { name: "권한", exact: true })
    .selectOption("viewer");
  await page.getByLabel("계정 활성", { exact: true }).uncheck();
  await page.getByRole("button", { name: "계정 저장" }).click();
  await expect
    .poll(() => calls.find((x) => x.p === "/users/u1")?.body)
    .toEqual({ role: "viewer", active: false });
  await nav(page, "수집 기기");
  await page.getByLabel("기기 이름").fill("ZED 2i B");
  await page.getByRole("button", { name: "기기 저장" }).click();
  await expect(
    page.getByText("fixture-token-one-time", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "토큰 화면에서 지우기" }).click();
  await expect(
    page.getByText("fixture-token-one-time", { exact: true }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "기기 편집" }).first().click();
  await page.getByLabel("교체 사유").fill("키 정기 교체");
  await page.getByRole("button", { name: "새 토큰 발급" }).click();
  await expect(
    page.getByText("fixture-rotated-token", { exact: true }),
  ).toBeVisible();
  await nav(page, "장착 프로파일");
  await page.getByLabel("프로파일 JSON").fill("{");
  await page.getByRole("button", { name: "새 버전 등록" }).click();
  await expect(page.getByRole("alert")).toBeVisible();
  await page.getByLabel("프로파일 JSON").fill(
    JSON.stringify({
      name: "실측 파일 fixture",
      version: 1,
      camera_serial: "fixture",
      lens_label: "fixture",
      calibration: {},
      config_hash: "a".repeat(64),
    }),
  );
  await page.getByRole("button", { name: "새 버전 등록" }).click();
  await expect(
    page.getByRole("heading", { name: "실측 파일 fixture · v1" }),
  ).toBeVisible();
  await nav(page, "모델 버전");
  await page.getByLabel("모델 manifest JSON").fill(
    JSON.stringify({
      version: "fixture-v1",
      code_commit: "fixture-code-commit",
      environment_hash: "c".repeat(64),
      weights_hashes: {},
      preprocess: {},
      thresholds: {},
      config_hash: "b".repeat(64),
    }),
  );
  await page.getByRole("button", { name: "새 버전 등록" }).click();
  await expect(page.getByRole("heading", { name: "fixture-v1" })).toBeVisible();
  expect(errors).toEqual([]);
});
test("privacy requires verified receipt, versioned approval and authenticated export", async ({
  page,
}) => {
  const calls = await setup(page);
  await nav(page, "개인정보 요청");
  await page.getByLabel("대상 UUID").fill(target);
  await page.getByLabel("요청 사유").fill("모의 열람 요청");
  await expect(
    page.getByRole("button", { name: "요청 접수", exact: true }),
  ).toBeDisabled();
  await page.getByRole("checkbox").nth(0).check();
  await page.getByRole("checkbox").nth(1).check();
  await page.getByRole("button", { name: "요청 접수", exact: true }).click();
  await page.getByRole("button", { name: "처리 내역" }).click();
  await expect(
    page.getByText("관리자 검토 대기", { exact: false }).first(),
  ).toBeVisible();
  await expect(
    page.getByText("MANUAL_ADMIN_REVIEW_REQUIRED", { exact: false }),
  ).toHaveCount(0);
  await page.getByLabel("처리 근거").fill("권한 및 대상 검증 완료");
  await page.getByRole("button", { name: "처리 기록 저장" }).click();
  await expect
    .poll(
      () =>
        calls.find(
          (x) => x.method === "PATCH" && x.p === "/privacy-requests/r0",
        )?.body,
    )
    .toEqual({
      expected_version: 1,
      action: "approve_access",
      reason: "권한 및 대상 검증 완료",
    });
  const event = page.waitForEvent("download");
  await expect(
    page.getByText("열람 승인 완료", { exact: false }).first(),
  ).toBeVisible();
  await page.getByRole("button", { name: "승인된 열람 자료 다운로드" }).click();
  expect((await event).suggestedFilename()).toBe("privacy-access.json");
});
test("run paths keep gaps separate, audit uses hashes, analysis reports unprepared weights", async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(e.message));
  await setup(page);
  await nav(page, "주행 기록");
  await page.getByRole("button", { name: "주행 상세", exact: true }).click();
  await expect(
    page.getByRole("img", { name: "주행 경로, 독립 구간 2개" }),
  ).toBeVisible();
  await expect(page.locator(".leaflet-overlay-pane path")).toHaveCount(2);
  await nav(page, "감사기록");
  await expect(page.getByText("행위자 참조: actor-hash")).toBeVisible();
  await nav(page, "분석 운영");
  await expect(page.getByText("실제 학습 가중치 필요")).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "메뉴 열기" }).click();
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "개인정보 요청", exact: true })
    .click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: info.outputPath("mobile-privacy.png"),
    fullPage: true,
    animations: "disabled",
  });
  expect(errors).toEqual([]);
});
test("period and geographic filters validate before sending and preserve query parameters", async ({
  page,
}) => {
  const calls = await setup(page);
  await nav(page, "점검 목록");
  await page.getByText("기간·구역 필터", { exact: true }).click();
  await page.getByLabel("촬영 시작").fill("2026-11-03T12:00");
  await page.getByLabel("촬영 종료").fill("2026-11-02T12:00");
  await page
    .getByRole("button", { name: "기간·구역 적용", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText("종료 시각");
  await page.getByLabel("촬영 종료").fill("2026-11-04T12:00");
  await page.getByLabel("구역 WGS84").fill("126.9,37.5,127.1,37.7");
  await page
    .getByRole("button", { name: "기간·구역 적용", exact: true })
    .click();
  await expect
    .poll(() => calls.filter((x) => x.p === "/observations").at(-1)?.query)
    .toContain("bbox=126.9%2C37.5%2C127.1%2C37.7");
});
test("viewer does not see administrative operations", async ({ page }) => {
  await setup(page, "viewer");
  for (const name of [
    "계정 관리",
    "수집 기기",
    "개인정보 요청",
    "감사기록",
    "주행 기록",
  ])
    await expect(
      page.getByRole("navigation").getByRole("button", { name, exact: true }),
    ).toHaveCount(0);
  await nav(page, "장착 프로파일");
  await expect(page.getByRole("button", { name: "새 버전 등록" })).toHaveCount(
    0,
  );
});

test("trend separates potholes and lane units and report preserves current filters", async ({
  page,
}) => {
  const calls = await setup(page);
  await expect(
    page.getByRole("img", {
      name: "포트홀 점검 건 날짜별 후보·확인·완료 현황",
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("meter", { name: "2026-11-03 후보 3건" }),
  ).toHaveAttribute("value", "3");
  await page
    .getByRole("combobox", { name: "현황 집계 종류" })
    .selectOption("lane_segment");
  await expect(
    page.getByRole("meter", { name: "2026-11-03 후보 7건" }),
  ).toBeVisible();
  await nav(page, "점검 목록");
  await page
    .getByLabel("관측 종류", { exact: true })
    .selectOption("lane_segment");
  await nav(page, "보고서");
  await expect(
    page.getByRole("combobox", { name: "보고서 관측 종류" }),
  ).toHaveValue("lane_segment");
  await page.getByRole("button", { name: "보고서 생성", exact: true }).click();
  await expect
    .poll(() => calls.find((x) => x.p === "/reports")?.query)
    .toContain("type=lane_segment");
});
test("privacy correction sends immutable addendum with expected version", async ({
  page,
}) => {
  const calls = await setup(page);
  await nav(page, "개인정보 요청");
  await page
    .getByRole("combobox", { name: "요청 종류" })
    .selectOption("correction");
  await page.getByLabel("대상 UUID").fill(target);
  await page.getByLabel("요청 사유").fill("모의 정정 요청");
  await page.getByRole("checkbox").nth(0).check();
  await page.getByRole("checkbox").nth(1).check();
  await page.getByRole("button", { name: "요청 접수", exact: true }).click();
  await page.getByRole("button", { name: "처리 내역" }).click();
  const addenda = [
    {
      observation_id: target,
      field: "annotation",
      value: "가상 자료 정정 검토",
    },
  ];
  await page.getByLabel("정정 내용 JSON").fill(JSON.stringify(addenda));
  await page.getByLabel("처리 근거").fill("근거 확인 완료");
  await page.getByRole("button", { name: "처리 기록 저장" }).click();
  await expect
    .poll(
      () =>
        calls.find(
          (x) => x.p === "/privacy-requests/r0" && x.method === "PATCH",
        )?.body,
    )
    .toEqual({
      expected_version: 1,
      action: "apply_correction",
      reason: "근거 확인 완료",
      corrections: addenda,
    });
});

test("analysis jobs send prepared IDs and expose actual blocked result", async ({
  page,
}) => {
  const calls = await setup(page);
  await nav(page, "분석 운영");
  await page.getByLabel("분석할 주행 UUID").fill(target);
  await page.getByLabel("등록 모델 번들 UUID").fill(target);
  await page.getByLabel("준비 설정 UUID").fill(target);
  await page
    .getByRole("button", { name: "배치 분석 요청", exact: true })
    .click();
  await expect
    .poll(
      () =>
        calls.find((x) => x.p === "/analysis/jobs" && x.method === "POST")
          ?.body,
    )
    .toEqual({
      run_id: target,
      model_bundle_id: target,
      prepared_config_id: target,
    });
  await page.getByRole("button", { name: "선택 작업 새로고침" }).click();
  await expect(
    page.getByRole("heading", { name: "분석 차단 · 준비 필요", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("분석에 필요한 학습 가중치가 준비되지 않았습니다.", {
      exact: true,
    }),
  ).toBeVisible();
});

test("public demo report sends explicit true only when selected", async ({
  page,
}) => {
  const calls = await setup(page);
  await nav(page, "보고서");
  await expect(
    page.getByRole("checkbox", { name: "공개 시연용으로 위치·시각 축약" }),
  ).not.toBeChecked();
  await page.getByRole("button", { name: "보고서 생성", exact: true }).click();
  await expect
    .poll(() => calls.filter((x) => x.p === "/reports").length)
    .toBe(1);
  expect(calls.find((x) => x.p === "/reports").query).not.toContain(
    "public_demo",
  );
  await page
    .getByRole("checkbox", { name: "공개 시연용으로 위치·시각 축약" })
    .check();
  await expect(
    page.getByText(
      "익명성을 보장하지 않으므로 공개 전 기관 승인이 필요합니다.",
      { exact: false },
    ),
  ).toBeVisible();
  await page.getByRole("button", { name: "보고서 생성", exact: true }).click();
  await expect
    .poll(() => calls.filter((x) => x.p === "/reports").at(-1)?.query)
    .toContain("public_demo=true");
});

test("registry shows quarantine count without exposing invalid legacy settings and requires model lineage", async ({
  page,
}) => {
  const calls = await setup(page);
  let quarantined = 2;
  await page.route("**/api/v1/profiles", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        data: {
          items: [
            {
              id: "valid-profile",
              name: "검증된 이전 프로파일",
              version: 1,
              camera_serial: "verified-camera",
              lens_label: "verified-lens",
            },
          ],
          quarantined_count: quarantined,
        },
      }),
    });
  });
  await nav(page, "장착 프로파일");
  await expect(
    page.getByText(
      "검증 기준을 충족하지 않는 이전 설정 2건을 격리했습니다. 새 버전으로 다시 등록하세요.",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: "검증된 이전 프로파일 · v1",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.locator(".management-list article")).toHaveCount(1);
  await expect(page.getByText("legacy-invalid-sensitive-value")).toHaveCount(0);
  quarantined = 0;
  await page.getByRole("button", { name: "새로고침", exact: true }).click();
  await expect(
    page.getByText("검증 기준을 충족하지 않는 이전 설정", { exact: false }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("heading", {
      name: "검증된 이전 프로파일 · v1",
      exact: true,
    }),
  ).toBeVisible();
  await nav(page, "모델 버전");
  await page
    .getByLabel("모델 manifest JSON")
    .fill(
      JSON.stringify({
        version: "missing-lineage",
        weights_hashes: {},
        preprocess: {},
        thresholds: {},
        config_hash: "a".repeat(64),
      }),
    );
  await page.getByRole("button", { name: "새 버전 등록" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "code_commit, environment_hash",
  );
  expect(
    calls.some((x) => x.p === "/model-bundles" && x.method === "POST"),
  ).toBeFalsy();
});
