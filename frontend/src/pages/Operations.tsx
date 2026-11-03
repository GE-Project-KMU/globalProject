import { useCallback, useEffect, useRef, useState } from "react";
import L from "leaflet";
import { request } from "../api";
import type { User } from "../types";
import { date, analysisReasonLabel } from "../labels";
import {
  Collection,
  JsonDetails,
  useCollection,
} from "../components/Management";
import { ErrorBox } from "../components/UI";

function RunMap({ segments }: { segments: any[] }) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!host.current) return;
    const m = L.map(host.current, {
        scrollWheelZoom: false,
        attributionControl: false,
      }).setView([37.55, 126.98], 12),
      bounds: L.LatLngTuple[] = [];
    for (const s of segments) {
      const points: L.LatLngTuple[] = (s.coordinates || [])
        .filter(
          (p: any) =>
            Array.isArray(p) && p.length === 2 && p.every(Number.isFinite),
        )
        .map(([lon, lat]: number[]) => [lat, lon]);
      if (points.length < 2) continue;
      const assessed = s.visible_scope?.assessment_state === "assessed";
      L.polyline(points, {
        color: assessed ? "#257e70" : "#9c916a",
        dashArray: assessed ? undefined : "5 7",
        weight: 4,
      })
        .addTo(m)
        .bindTooltip(
          `${assessed ? "판정 범위 있음" : "범위·판정 확인 필요"} · ${s.quality || "품질 미상"}`,
        );
      bounds.push(...points);
    }
    if (bounds.length)
      m.fitBounds(L.latLngBounds(bounds), {
        padding: [24, 24],
        maxZoom: 17,
        animate: false,
      });
    const obs = new ResizeObserver(() =>
      m.invalidateSize({ animate: false, pan: false }),
    );
    obs.observe(host.current);
    return () => {
      obs.disconnect();
      m.stop();
      m.remove();
    };
  }, [segments]);
  return (
    <div className="map-wrap">
      <div
        ref={host}
        className="map-canvas"
        role="img"
        aria-label={`주행 경로, 독립 구간 ${segments.length}개`}
      />
      <div className="map-caption">
        배경지도 미연결 · 구간 사이 공백은 연결하지 않습니다
      </div>
    </div>
  );
}
export function Runs({ token }: { token: string }) {
  const state = useCollection(token, "/runs"),
    [selected, setSelected] = useState<any>(null),
    [error, setError] = useState("");
  async function open(x: any) {
    try {
      setError("");
      setSelected(await request(token, `/runs/${x.run_id || x.id}`));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <>
      <Collection
        title="주행 기록"
        description="수집·종료·GNSS 품질·실제 가시 범위를 확인합니다. 운행 시작과 종료는 정차 후 로컬 수집 에이전트에서 실행합니다."
        state={state}
      >
        <div className="management-list">
          {state.items.map((x) => (
            <article key={x.run_id || x.id}>
              <div>
                <h3>
                  {(
                    {
                      recording: "수집 중",
                      completed: "종료",
                      interrupted: "중단",
                    } as any
                  )[x.status] || x.status}
                </h3>
                <p>
                  {date(x.started_at)} → {date(x.ended_at)}
                </p>
                <code>{x.run_id || x.id}</code>
              </div>
              <button className="button secondary" onClick={() => void open(x)}>
                주행 상세
              </button>
            </article>
          ))}
        </div>
      </Collection>
      {error && <ErrorBox message={error} />}{" "}
      {selected && (
        <section className="panel management-panel">
          <div className="panel-heading">
            <h2>주행 상세</h2>
            <button
              className="button secondary"
              onClick={() => setSelected(null)}
            >
              주행 상세 닫기
            </button>
          </div>
          <p>
            관측 경로는 차량의 이동 기록입니다. 도로 전체나 모든 차로의 검사
            완료를 뜻하지 않습니다.
          </p>
          {selected.segments?.length ? (
            <RunMap segments={selected.segments} />
          ) : (
            <p className="notice">
              유효 경로 구간이 없습니다. 위치 공백은 정상 점검으로 간주하지
              않습니다.
            </p>
          )}
          <dl className="metadata-grid">
            <div>
              <dt>기기</dt>
              <dd>{selected.device_id}</dd>
            </div>
            <div>
              <dt>장착 프로파일</dt>
              <dd>{selected.profile_id}</dd>
            </div>
            <div>
              <dt>기한</dt>
              <dd>
                {selected.expires_at
                  ? date(selected.expires_at)
                  : selected.expires
                    ? date(new Date(selected.expires * 1000).toISOString())
                    : "기록 참조"}
              </dd>
            </div>
          </dl>
          <JsonDetails
            label="사전 점검·품질·구간 근거"
            value={{
              precheck: selected.precheck,
              summary: selected.summary,
              segments: selected.segments,
              manifest_hash: selected.manifest_hash,
            }}
          />
        </section>
      )}
    </>
  );
}
export function Audit({ token, user }: { token: string; user: User }) {
  const state = useCollection(token, "/audit-logs", user.role === "admin");
  if (user.role !== "admin")
    return <p className="panel">감사기록은 관리자만 조회할 수 있습니다.</p>;
  return (
    <Collection
      title="최소 감사기록"
      description="접근·설정·검토·삭제 이력을 확인합니다. 영상·상세 좌표·민감한 자유 서술은 기록하지 않습니다. 기본 보관 180일."
      state={state}
    >
      <div className="management-list">
        {state.items.map((x, i) => (
          <article key={x.id || i}>
            <div>
              <h3>
                {x.action} · {x.service}
              </h3>
              <p>
                {date(
                  x.at ||
                    x.created_at ||
                    x.timestamp ||
                    (x.created
                      ? new Date(x.created * 1000).toISOString()
                      : undefined),
                )}
              </p>
              <p>행위자 참조: {x.actor_hash || "시스템"}</p>
              <code>대상 참조: {x.target_hash || "—"}</code>
            </div>
          </article>
        ))}
      </div>
    </Collection>
  );
}
export function AnalysisStatus({ token, user }: { token: string; user: User }) {
  const [data, setData] = useState<any>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const load = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      setData(await request(token, "/analysis/capabilities"));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [token]);
  useEffect(() => {
    void load();
  }, [load]);
  return (
    <>
      <section className="panel management-panel">
        <div className="panel-heading">
          <div>
            <h2>분석·수집 준비 상태</h2>
            <p>
              실제 장비와 가중치 준비 상태를 확인합니다. 등록 건수와 모델 성능을
              구분합니다.
            </p>
          </div>
          <button
            className="button secondary"
            onClick={() => void load()}
            disabled={busy}
          >
            새로고침
          </button>
        </div>
        {error && <ErrorBox message={error} onRetry={load} />}{" "}
        {data && (
          <>
            <div className="service-grid">
              <div className="service-card">
                <h3>원본 처리 위치</h3>
                <p>
                  {data.raw_processing === "local_agent"
                    ? "암호화 로컬 에이전트"
                    : data.raw_processing}
                </p>
                <small>
                  서버 원본 업로드 {data.server_raw_upload ? "허용" : "차단"}
                </small>
              </div>
              <div className="service-card">
                <h3>모델 준비</h3>
                <p>
                  {data.model_status === "target_weights_required"
                    ? "실제 학습 가중치 필요"
                    : data.model_status}
                </p>
                <small>
                  등록 manifest {data.manifest_count ?? 0}개 · 정확도 검증은
                  별도
                </small>
              </div>
            </div>
            <p>
              장착·렌즈·GNSS·저장 공간·촬영 안내를 확인하고 정차 후 로컬 도구를
              사용하세요.
            </p>
            <code>{data.local_agent_command || "python -m agent --help"}</code>
            <JsonDetails label="전체 분석 기능 상태" value={data} />
          </>
        )}
      </section>
      <AnalysisJobs token={token} user={user} />
    </>
  );
}

function AnalysisJobs({ token, user }: { token: string; user: User }) {
  const state = useCollection(token, "/analysis/jobs", user.role !== "viewer");
  const [run, setRun] = useState(""),
    [bundle, setBundle] = useState(""),
    [prepared, setPrepared] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [selected, setSelected] = useState<any>(null);
  const names: Record<string, string> = {
    queued: "분석 대기",
    running: "분석 중",
    completed: "로컬 검수 패키지 생성 완료",
    blocked: "분석 차단 · 준비 필요",
    failed: "분석 실패",
  };
  useEffect(() => {
    if (
      user.role === "viewer" ||
      !state.items.some((x) => ["queued", "running"].includes(x.status))
    )
      return;
    const timer = window.setInterval(() => void state.load(), 3000);
    return () => clearInterval(timer);
  }, [state.items, state.load, user.role]);
  if (user.role === "viewer") return null;
  async function create(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const row = await request<any>(token, "/analysis/jobs", {
        method: "POST",
        body: JSON.stringify({
          run_id: run.trim(),
          model_bundle_id: bundle.trim(),
          prepared_config_id: prepared.trim(),
        }),
      });
      setSelected(row);
      await state.load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function refresh(id: string) {
    setError("");
    try {
      setSelected(await request(token, `/analysis/jobs/${id}`));
      await state.load();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <>
      <Collection
        title="배치 분석 작업"
        description="실제 로컬 에이전트를 실행합니다. 완료된 결과도 사람이 사진 공유를 검수하기 전까지 공개되지 않습니다."
        state={state}
      >
        <div className="management-list">
          {state.items.map((x) => (
            <article key={x.id}>
              <div>
                <h3>{names[x.status] || x.status}</h3>
                <p>
                  {date(x.created_at)}
                  {x.reason ? ` · ${analysisReasonLabel(x.reason)}` : ""}
                </p>
                <code>{x.id}</code>
                {x.status === "completed" && (
                  <p>
                    출력 관측 {x.observation_count ?? 0}건 · 로컬 근거·마스킹
                    검수 필요
                  </p>
                )}
              </div>
              <button
                className="button secondary"
                onClick={() => void refresh(x.id)}
              >
                작업 상태
              </button>
            </article>
          ))}
        </div>
      </Collection>
      <section className="panel management-panel">
        <h2>준비된 주행 분석 실행</h2>
        <p>
          운영자가 먼저 data/agent-jobs/prepared/&lt;설정 UUID&gt;.json에
          장비·프로파일·모델·입출력 설정을 준비해야 합니다. 서버에는 UUID만
          전달하며 원본 영상이나 임의 파일 경로를 업로드하지 않습니다.
        </p>
        {error && <ErrorBox message={error} />}
        <form className="management-form" onSubmit={create}>
          {[
            ["분석할 주행 UUID", run, setRun],
            ["등록 모델 번들 UUID", bundle, setBundle],
            ["준비 설정 UUID", prepared, setPrepared],
          ].map(([label, value, setter]) => (
            <label key={label as string}>
              {label as string}
              <input
                required
                value={value as string}
                pattern="[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
                onChange={(e) =>
                  (setter as (s: string) => void)(e.target.value)
                }
              />
            </label>
          ))}
          <button className="button primary" disabled={busy}>
            {busy ? "실행 요청 중…" : "배치 분석 요청"}
          </button>
        </form>
        {selected && (
          <div className="notice">
            <div>
              <h3>{names[selected.status] || selected.status}</h3>
              <p>
                {analysisReasonLabel(selected.reason) ||
                  "실행 상태와 로컬 검수 패키지를 확인하세요."}
              </p>
              <button
                className="button secondary"
                onClick={() => void refresh(selected.id)}
              >
                선택 작업 새로고침
              </button>
              <JsonDetails label="실제 분석 작업 정보" value={selected} />
            </div>
          </div>
        )}
      </section>
    </>
  );
}
