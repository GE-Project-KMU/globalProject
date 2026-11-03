import { useEffect, useState } from "react";
import { Check, LoaderCircle, X } from "lucide-react";
import { ApiError, request } from "../api";
import type { Observation, User } from "../types";
import { statusLabels, gradeLabels, date, id, typeName } from "../labels";
import { Badge, ErrorBox } from "./UI";
import ProtectedImage from "./ProtectedImage";

export default function Detail({
  observation,
  token,
  user,
  onClose,
  onUpdated,
}: {
  observation: Observation;
  token: string;
  user: User;
  onClose: () => void;
  onUpdated: () => void;
}) {
  const [item, setItem] = useState(observation),
    [status, setStatus] = useState(observation.review.status),
    [priority, setPriority] = useState(observation.review.priority || "normal"),
    [reason, setReason] = useState(""),
    [override, setOverride] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [history, setHistory] = useState<any[]>([]);
  const editable = ["admin", "reviewer"].includes(user.role);
  useEffect(() => {
    let alive = true;
    request<Observation>(token, `/observations/${id(observation)}`)
      .then((x) => {
        if (alive) {
          setItem(x);
          setStatus(x.review.status);
          setPriority(x.review.priority || "normal");
        }
      })
      .catch(() => {});
    request<any>(token, `/cases/${observation.case_id}/history`)
      .then((x) => {
        if (alive) setHistory(x.entries || x.items || []);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [token, observation]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", key);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", key);
      document.body.style.overflow = previous;
    };
  }, [onClose]);
  async function save(e: React.FormEvent) {
    e.preventDefault();
    if (!reason.trim()) return;
    setBusy(true);
    setError("");
    try {
      const result = await request<any>(
        token,
        `/observations/${id(item)}/review`,
        {
          method: "PATCH",
          body: JSON.stringify({
            expected_case_id: item.case_id,
            expected_version: item.case_version,
            status,
            priority,
            reason: reason.trim(),
            override_transition: override,
          }),
        },
      );
      setItem({
        ...item,
        case_version: result.version || result.case_version,
        review: result.review || { status, priority },
      });
      setReason("");
      onUpdated();
      onClose();
    } catch (e) {
      setError(
        e instanceof ApiError && e.status === 409
          ? "다른 담당자가 변경했거나 허용되지 않는 상태 전이입니다. 상세 정보를 다시 열어 최신 상태를 확인해 주세요."
          : (e as Error).message,
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <section
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label="관측 상세"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="drawer-heading">
          <div>
            <span className="eyebrow">INSPECTION DETAIL</span>
            <h2>{typeName(item)} 관측 상세</h2>
          </div>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="상세 닫기"
            autoFocus
          >
            <X />
          </button>
        </header>
        <div className="drawer-body">
          {(item.demo || item.synthetic) && (
            <div className="notice small">
              가상 시연 자료입니다. 실제 탐지 성능을 나타내지 않습니다.
            </div>
          )}
          <ProtectedImage
            token={token}
            path={
              item.image_content_path ||
              (item.image?.file_id
                ? `/api/v1/images/${item.image.file_id}/content`
                : undefined)
            }
          />
          <div className="detail-meta">
            <Badge status={item.review.status} />
            <span>{date(item.captured_at)} KST</span>
          </div>
          <dl className="detail-grid">
            <div>
              <dt>관측 종류</dt>
              <dd>{typeName(item)}</dd>
            </div>
            <div>
              <dt>
                {item.type === "pothole" ? "모델 confidence" : "시각적 잔존율"}
              </dt>
              <dd>
                {item.type === "pothole"
                  ? `${((item.pothole?.confidence || 0) * 100).toFixed(1)}%`
                  : item.lane?.retention_pct == null
                    ? "판정 불가"
                    : `${item.lane.retention_pct.toFixed(1)}%`}
              </dd>
            </div>
            <div>
              <dt>차량 관측 위치</dt>
              <dd>
                {item.location?.lat == null
                  ? "위치 없음"
                  : `${item.location.lat.toFixed(5)}, ${item.location.lon!.toFixed(5)}`}
              </dd>
            </div>
            <div>
              <dt>위치 품질</dt>
              <dd>
                {(
                  {
                    valid: "유효",
                    degraded: "품질 저하",
                    unavailable: "없음",
                  } as any
                )[item.location?.quality || ""] || "미기록"}
              </dd>
            </div>
            {item.lane && (
              <>
                <div>
                  <dt>차선 판정</dt>
                  <dd>{gradeLabels[item.lane.grade] || item.lane.grade}</dd>
                </div>
                <div>
                  <dt>유효 영역</dt>
                  <dd>
                    {item.lane.valid_area_ratio == null
                      ? "근거 없음"
                      : `${(item.lane.valid_area_ratio * 100).toFixed(0)}%`}
                  </dd>
                </div>
              </>
            )}
          </dl>
          {item.lane?.unassessable_reason && (
            <div className="notice">
              판정 불가 사유: {item.lane.unassessable_reason}
            </div>
          )}
          <p className="help-text">
            confidence는 사고 위험 확률이 아니며, 시각적 잔존율은 법정
            재귀반사성능 검사 결과가 아닙니다.
          </p>
          <h3 className="section-label">검토 및 조치</h3>
          {error && <ErrorBox message={error} />}
          <form onSubmit={save}>
            <div className="form-row">
              <label>
                검토 상태
                <select
                  value={status}
                  onChange={(e) => setStatus(e.target.value)}
                  disabled={!editable}
                >
                  {Object.entries(statusLabels).map(([v, l]) => (
                    <option key={v} value={v}>
                      {l}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                검토 우선순위
                <select
                  value={priority}
                  onChange={(e) => setPriority(e.target.value)}
                  disabled={!editable}
                >
                  <option value="normal">일반</option>
                  <option value="urgent">긴급</option>
                  <option value="low">낮음</option>
                </select>
              </label>
            </div>
            <label>
              변경 사유
              <textarea
                maxLength={500}
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="현장 확인 결과나 조치 사유를 입력해 주세요. 개인정보는 입력하지 마세요."
                required
                disabled={!editable}
              />
            </label>
            <label className="check-label">
              <input
                type="checkbox"
                checked={override}
                onChange={(e) => setOverride(e.target.checked)}
                disabled={!editable}
              />
              기각·완료 취소 또는 예외 상태 전이
            </label>
            <button
              className="button primary full"
              disabled={!editable || busy || !reason.trim()}
            >
              {busy ? (
                <LoaderCircle className="spin" size={16} />
              ) : (
                <Check size={16} />
              )}
              검토 결과 저장
            </button>
            {!editable && (
              <p className="help-text">
                조회 권한입니다. 검토자 또는 관리자가 상태를 변경할 수 있습니다.
              </p>
            )}
          </form>
          <h3 className="section-label">변경 이력</h3>
          {history.length ? (
            history.map((h, i) => (
              <div className="history-row" key={i}>
                <span className="history-dot" />
                <div>
                  <b>{statusLabels[h.to_status] || h.action || "변경 기록"}</b>
                  <p>{h.reason}</p>
                  <small>{date(h.at || h.created_at)}</small>
                </div>
              </div>
            ))
          ) : (
            <p className="muted">등록된 변경 이력이 없습니다.</p>
          )}
          <details className="technical">
            <summary>관측 식별·버전 정보</summary>
            <p>관측 ID: {id(item)}</p>
            <p>
              점검 건: {item.case_id} · v{item.case_version}
            </p>
            <p>모델: {item.model_version || "미기록"}</p>
            <p>보관기한: {date(item.expires_at)}</p>
          </details>
        </div>
      </section>
    </div>
  );
}
