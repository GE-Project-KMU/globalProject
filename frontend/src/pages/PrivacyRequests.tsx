import { useState, type FormEvent } from "react";
import { request, download, ApiError } from "../api";
import type { User } from "../types";
import {
  Collection,
  Feedback,
  JsonDetails,
  useCollection,
} from "../components/Management";
import { date } from "../labels";
const names: Record<string, string> = {
  access: "열람",
  correction: "정정",
  deletion: "삭제",
  received: "접수됨",
  in_progress: "처리 중",
  completed: "완료",
  failed: "실패",
  rejected: "반려",
};
export default function PrivacyRequests({
  token,
  user,
}: {
  token: string;
  user: User;
}) {
  const state = useCollection(
    token,
    "/privacy-requests",
    user.role === "admin",
  );
  const [type, setType] = useState("access"),
    [kind, setKind] = useState("observation"),
    [targets, setTargets] = useState(""),
    [reason, setReason] = useState(""),
    [verified, setVerified] = useState(false),
    [selected, setSelected] = useState<any>(null),
    [action, setAction] = useState("approve_access"),
    [decision, setDecision] = useState(""),
    [corrections, setCorrections] = useState(""),
    [confirmed, setConfirmed] = useState(false),
    [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false);
  if (user.role !== "admin")
    return (
      <p className="panel">
        개인정보 요청의 접수·처리는 관리자 권한이 필요합니다.
      </p>
    );
  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!verified || !confirmed) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const refs = [...new Set(targets.trim().split(/[\s,]+/))].map((id) => ({
        kind,
        id,
      }));
      if (refs.length > 100)
        throw new Error("요청당 최대 100개 대상을 지정하세요.");
      await request(token, "/privacy-requests", {
        method: "POST",
        body: JSON.stringify({
          request_type: type,
          target_refs: refs,
          reason: reason.trim(),
        }),
      });
      setTargets("");
      setReason("");
      setVerified(false);
      setConfirmed(false);
      setMessage(
        "요청을 접수했습니다. 처리 결과와 파생자료 정리 결과를 확인하세요.",
      );
      await state.load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function process(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const body: any = {
        expected_version: selected.version,
        action,
        reason: decision.trim(),
      };
      if (action === "apply_correction")
        body.corrections = JSON.parse(corrections);
      const d = await request<any>(
        token,
        `/privacy-requests/${selected.request_id}`,
        { method: "PATCH", body: JSON.stringify(body) },
      );
      setSelected(d);
      setDecision("");
      setMessage("처리 기록을 저장했습니다.");
      await state.load();
    } catch (e) {
      if (
        e instanceof ApiError &&
        e.status === 409 &&
        e.code === "VERSION_CONFLICT"
      ) {
        setError(
          "처리 상태가 갱신되었습니다. 최신 내역을 확인한 뒤 처리 기록을 다시 저장하세요.",
        );
        try {
          setSelected(
            await request(token, `/privacy-requests/${selected.request_id}`),
          );
        } catch {}
      } else setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function open(x: any) {
    setError("");
    try {
      const d = await request<any>(token, `/privacy-requests/${x.request_id}`);
      setSelected(d);
      setAction(
        d.request_type === "access"
          ? "approve_access"
          : d.request_type === "correction"
            ? "apply_correction"
            : d.status === "failed"
              ? "retry"
              : "reject",
      );
      setDecision("");
      setCorrections("");
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <>
      <Collection
        title="개인정보 처리 요청"
        description="최소 대상 참조와 처리 이력만 기록합니다. 실제 신원 자료·연락처·차량 번호는 입력하지 마세요."
        state={state}
      >
        <div className="management-list">
          {state.items.map((x) => (
            <article key={x.request_id}>
              <div>
                <h3>
                  {names[x.request_type]} · {names[x.status] || x.status}
                </h3>
                <p>
                  {date(x.created_at)} · {x.result_code || "결과 대기"}
                </p>
                <code>{x.request_id}</code>
              </div>
              <button className="button secondary" onClick={() => void open(x)}>
                처리 내역
              </button>
            </article>
          ))}
        </div>
      </Collection>
      <Feedback error={error} message={message} />
      {selected && (
        <section className="panel management-panel">
          <div className="panel-heading">
            <h2>요청 처리 · {names[selected.request_type]}</h2>
            <button
              className="button secondary"
              onClick={() => void open(selected)}
            >
              처리 상태 갱신
            </button>
          </div>
          <p>
            {names[selected.status] || selected.status} · 버전{" "}
            {selected.version} · {selected.result_code || "대기"}
          </p>
          <p>접수 사유: {selected.reason}</p>
          <JsonDetails
            label="대상·처리 결과 보기"
            value={{
              target_refs: selected.target_refs,
              actions: selected.actions,
              decision_reason: selected.decision_reason,
              resolved_at: selected.resolved_at,
            }}
          />
          {selected.download_path && (
            <button
              className="button secondary"
              onClick={async () => {
                try {
                  await download(
                    token,
                    selected.download_path,
                    "privacy-access.json",
                  );
                } catch (e) {
                  setError((e as Error).message);
                }
              }}
            >
              승인된 열람 자료 다운로드
            </button>
          )}
          {!["completed", "rejected"].includes(selected.status) &&
            (selected.request_type === "deletion"
              ? selected.status === "failed"
              : selected.status === "in_progress" &&
                selected.result_code === "MANUAL_ADMIN_REVIEW_REQUIRED") && (
              <form className="management-form" onSubmit={process}>
                <label>
                  처리 방법
                  <select
                    value={action}
                    onChange={(e) => setAction(e.target.value)}
                  >
                    {selected.request_type === "access" && (
                      <option value="approve_access">열람 승인</option>
                    )}
                    {selected.request_type === "correction" && (
                      <option value="apply_correction">
                        정정 부가 기록 적용
                      </option>
                    )}
                    {selected.request_type === "deletion" &&
                      selected.status === "failed" && (
                        <option value="retry">삭제 재시도</option>
                      )}
                    {selected.request_type !== "deletion" && (
                      <option value="reject">사유를 남기고 반려</option>
                    )}
                  </select>
                </label>
                {action === "apply_correction" && (
                  <>
                    <p>
                      원본 추론은 보존하고 정정 기록을 연결합니다. JSON 배열:
                      observation_id, field (location 또는 annotation), value.
                      location은 lat/lon/quality, annotation은 정정 설명
                      문자열입니다.
                    </p>
                    <label>
                      정정 내용 JSON
                      <textarea
                        required
                        rows={7}
                        value={corrections}
                        onChange={(e) => setCorrections(e.target.value)}
                      />
                    </label>
                  </>
                )}
                <label>
                  처리 근거
                  <textarea
                    required
                    maxLength={500}
                    value={decision}
                    onChange={(e) => setDecision(e.target.value)}
                  />
                </label>
                <button
                  className="button primary"
                  disabled={busy || !decision.trim()}
                >
                  처리 기록 저장
                </button>
              </form>
            )}
          <button
            className="button secondary"
            onClick={() => setSelected(null)}
          >
            내역 닫기
          </button>
        </section>
      )}
      <section className="panel management-panel">
        <h2>요청 접수</h2>
        <form className="management-form" onSubmit={submit}>
          <div className="form-grid">
            <label>
              요청 종류
              <select
                value={type}
                onChange={(e) => {
                  setType(e.target.value);
                  setConfirmed(false);
                }}
              >
                {["access", "correction", "deletion"].map((k) => (
                  <option key={k} value={k}>
                    {names[k]}
                  </option>
                ))}
              </select>
            </label>
            <label>
              대상 종류
              <select value={kind} onChange={(e) => setKind(e.target.value)}>
                {[
                  ["observation", "관측"],
                  ["image", "사진"],
                  ["run", "주행"],
                  ["case", "점검 건"],
                ].map(([k, n]) => (
                  <option key={k} value={k}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label>
            대상 UUID (줄바꿈 또는 쉼표로 구분)
            <textarea
              required
              rows={3}
              value={targets}
              onChange={(e) => setTargets(e.target.value)}
              spellCheck={false}
            />
          </label>
          <label>
            요청 사유 (민감정보 제외)
            <textarea
              required
              maxLength={500}
              value={reason}
              onChange={(e) => setReason(e.target.value)}
            />
          </label>
          <label className="check-row">
            <input
              type="checkbox"
              checked={verified}
              onChange={(e) => setVerified(e.target.checked)}
            />
            기관 절차에 따라 요청자의 권한과 대상 연결을 확인했습니다.
          </label>
          <label className="check-row">
            <input
              type="checkbox"
              checked={confirmed}
              onChange={(e) => setConfirmed(e.target.checked)}
            />
            {type === "deletion"
              ? "삭제는 복구할 수 없으며 관련 사진·보고서·상세 경로도 처리됩니다. 대상과 요청을 확인했습니다."
              : "최소 식별정보만 입력했으며 접수 후 승인·처리 결과를 확인하겠습니다."}
          </label>
          <button
            className="button primary"
            disabled={busy || !verified || !confirmed || !reason.trim()}
          >
            {busy ? "접수 중…" : "요청 접수"}
          </button>
        </form>
      </section>
    </>
  );
}
