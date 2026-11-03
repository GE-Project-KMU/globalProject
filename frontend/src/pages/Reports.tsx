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

export default function Reports({ token }: { token: string }) {
  const [format, setFormat] = useState("csv"),
    [busy, setBusy] = useState(false),
    [job, setJob] = useState<any>(null),
    [error, setError] = useState("");
  async function create() {
    setBusy(true);
    setError("");
    try {
      const d = await request<any>(token, `/reports?format=${format}`);
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
                    `road-insight.${format}`,
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
