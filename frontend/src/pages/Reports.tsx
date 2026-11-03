import { useState } from "react";
import {
  FileText,
  ArrowDownToLine,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { download, request } from "../api";
import { ErrorBox } from "../components/UI";
import { statusLabels } from "../labels";

export default function Reports({
  token,
  initialFilters = {},
}: {
  token: string;
  initialFilters?: Record<string, string>;
}) {
  const [format, setFormat] = useState("csv"),
    [busy, setBusy] = useState(false),
    [job, setJob] = useState<any>(null),
    [error, setError] = useState("");
  const [filters, setFilters] =
      useState<Record<string, string>>(initialFilters),
    [jobFormat, setJobFormat] = useState("csv");
  async function create() {
    setBusy(true);
    setError("");
    try {
      if (
        filters.from &&
        filters.to &&
        new Date(filters.from) >= new Date(filters.to)
      )
        throw new Error("종료 시각은 시작 시각보다 뒤여야 합니다.");
      const query = new URLSearchParams({ format });
      for (const [key, value] of Object.entries(filters)) {
        if (value)
          query.set(
            key,
            ["from", "to"].includes(key)
              ? new Date(value).toISOString()
              : value,
          );
      }
      const d = await request<any>(token, `/reports?${query}`);
      setJobFormat(format);
      setJob(d);
      if (d.status === "succeeded" || d.download_path) {
        await download(
          token,
          d.download_path || `/reports/${d.job_id || d.id}/download`,
          `road-insight.${format}`,
        );
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function refresh() {
    if (!job) return;
    setBusy(true);
    setError("");
    try {
      setJob(await request(token, `/jobs/${job.job_id || job.id}`));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="narrow-section">
      <div className="panel report-panel">
        <span className="report-icon">
          <FileText size={30} />
        </span>
        <h2>검토 결과를 보고서로</h2>
        <p className="muted">
          승인된 비식별 자료와 검토·조치 이력을 내보냅니다.
          <br />
          포트홀은 점검 건, 차선은 관측 구간으로 구분해 집계합니다.
        </p>
        {error && <ErrorBox message={error} />}
        <details className="report-filters" open>
          <summary>출력 조회 조건 · 현재 점검 필터에서 가져옴</summary>
          <div className="management-form">
            <div className="form-grid">
              {[
                ["from", "촬영 시작"],
                ["to", "촬영 종료 (미포함)"],
              ].map(([key, label]) => (
                <label key={key}>
                  {label}
                  <input
                    type="datetime-local"
                    value={filters[key] || ""}
                    onChange={(e) =>
                      setFilters({ ...filters, [key]: e.target.value })
                    }
                  />
                </label>
              ))}
            </div>
            <label>
              보고서 관측 종류
              <select
                value={filters.type || ""}
                onChange={(e) =>
                  setFilters({ ...filters, type: e.target.value })
                }
              >
                <option value="">모든 종류</option>
                <option value="pothole">포트홀</option>
                <option value="lane_segment">차선 구간</option>
              </select>
            </label>
            <label>
              보고서 검토 상태
              <select
                value={filters.review_status || ""}
                onChange={(e) =>
                  setFilters({ ...filters, review_status: e.target.value })
                }
              >
                <option value="">모든 상태</option>
                {Object.entries(statusLabels).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            <label>
              보고서 위치 품질
              <select
                value={filters.quality || ""}
                onChange={(e) =>
                  setFilters({ ...filters, quality: e.target.value })
                }
              >
                <option value="">모든 품질</option>
                <option value="valid">유효</option>
                <option value="degraded">저하</option>
                <option value="unavailable">없음</option>
              </select>
            </label>
            <label>
              보고서 구역 WGS84 경계
              <input
                value={filters.bbox || ""}
                onChange={(e) =>
                  setFilters({ ...filters, bbox: e.target.value })
                }
                placeholder="서경도,남위도,동경도,북위도"
              />
            </label>
            <p>
              기간 최대 90일, 최대 10,000개 대표 점검 건. 텍스트 검색은 출력
              조건에 포함되지 않습니다.
            </p>
          </div>
        </details>
        <label>
          내보내기 형식
          <select value={format} onChange={(e) => setFormat(e.target.value)}>
            <option value="csv">CSV · 데이터 분석용</option>
            <option value="pdf">PDF · 검토 보고용</option>
          </select>
        </label>
        <button
          className="button primary full"
          onClick={create}
          disabled={busy}
        >
          {busy ? (
            <LoaderCircle size={16} className="spin" />
          ) : (
            <ArrowDownToLine size={16} />
          )}
          보고서 생성
        </button>
        {job && (
          <div className="job-card">
            <div>
              <b>
                {(
                  {
                    pending: "생성 대기",
                    running: "생성 중",
                    succeeded: "다운로드 준비 완료",
                    failed: "생성 실패",
                  } as any
                )[job.status] || "작업 접수됨"}
              </b>
              <small>{job.job_id || job.id}</small>
            </div>
            <button
              className="icon-button"
              onClick={refresh}
              disabled={busy}
              aria-label="보고서 상태 새로고침"
            >
              <RefreshCw size={17} />
            </button>
            {job.status === "succeeded" && (
              <button
                className="button secondary full"
                onClick={() =>
                  download(
                    token,
                    job.download_path ||
                      `/reports/${job.job_id || job.id}/download`,
                    `road-insight.${jobFormat}`,
                  ).catch((e) => setError(e.message))
                }
              >
                다운로드
              </button>
            )}
            {job.error_code && <p className="danger-text">{job.error_code}</p>}
          </div>
        )}
      </div>
      <div className="notice">
        <ShieldCheck size={17} />
        <div>
          보고서는 최대 24시간 보관되며 원자료의 기한을 넘기지 않습니다. 법정
          반사성능 검사와 현장 판단을 대신하지 않습니다.
        </div>
      </div>
    </div>
  );
}
