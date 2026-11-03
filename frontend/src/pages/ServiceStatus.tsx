import { useState, useCallback, useEffect } from "react";
import { RefreshCw, LoaderCircle, Database, ShieldCheck } from "lucide-react";
import { request } from "../api";
import { ErrorBox } from "../components/UI";

export default function ServiceStatus({ token }: { token: string }) {
  const [items, setItems] = useState<any[]>([]),
    [busy, setBusy] = useState(true),
    [error, setError] = useState("");
  const load = useCallback(() => {
    setBusy(true);
    setError("");
    request<any>(token, "/system/status")
      .then((d) => setItems(Array.isArray(d) ? d : d.services || []))
      .catch((e) => setError(e.message))
      .finally(() => setBusy(false));
  }, [token]);
  useEffect(load, [load]);
  return (
    <>
      <div className="panel">
        <div className="panel-heading">
          <div>
            <h2>서비스 연결 상태</h2>
            <p>독립 서비스의 상태를 확인합니다.</p>
          </div>
          <button className="button secondary" onClick={load} disabled={busy}>
            <RefreshCw size={16} />
            새로고침
          </button>
        </div>
        {error && <ErrorBox message={error} />}
        <div className="service-grid">
          {busy ? (
            <div className="loading">
              <LoaderCircle className="spin" />
              확인 중
            </div>
          ) : (
            items.map((s, i) => (
              <div className="service-card" key={s.name || i}>
                <div>
                  <Database size={20} />
                  <span
                    className={`service-dot ${["ok", "healthy", "up", "online"].includes(s.status) ? "healthy" : "unhealthy"}`}
                  />
                </div>
                <h3>{s.name}</h3>
                <p>
                  {["ok", "healthy", "up", "online"].includes(s.status)
                    ? "정상 연결"
                    : s.status || "응답 없음"}
                </p>
              </div>
            ))
          )}
        </div>
      </div>
      <div className="panel policy-panel">
        <ShieldCheck />
        <div>
          <h2>개인정보와 보관 원칙</h2>
          <p>
            원본은 암호화된 로컬에 7일 이내, 마스킹된 점검 상세는 90일 이내,
            최소 감사기록은 180일 보관합니다. 사진은 공유 전에 사람의 확인과
            해시 승인이 필요합니다.
          </p>
          <small>
            프로젝트 설계 기준 · 기관 운영 정책과 자료 사용 조건을 우선
            확인합니다
          </small>
        </div>
      </div>
    </>
  );
}
