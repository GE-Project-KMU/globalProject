import { useState, useCallback, useEffect } from "react";
import { Link2, RefreshCw, LoaderCircle, ArrowRight } from "lucide-react";
import { request } from "../api";
import type { User } from "../types";
import { ErrorBox, Empty } from "../components/UI";

export default function Duplicates({
  token,
  user,
}: {
  token: string;
  user: User;
}) {
  const [items, setItems] = useState<any[]>([]),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(true),
    [reason, setReason] = useState(""),
    [working, setWorking] = useState("");
  const load = useCallback(() => {
    setBusy(true);
    setError("");
    request<any>(token, "/duplicate-links")
      .then((d) => setItems(d.items || d.links || []))
      .catch((e) => setError(e.message))
      .finally(() => setBusy(false));
  }, [token]);
  useEffect(load, [load]);
  async function decide(item: any, decision: string) {
    setWorking(item.id);
    setError("");
    try {
      await request(token, "/duplicate-links", {
        method: "POST",
        body: JSON.stringify({
          source_observation_id:
            item.source_observation_id || item.evidence?.source_observation_id,
          target_observation_id:
            item.target_observation_id || item.evidence?.target_observation_id,
          decision,
          source_version: item.source_version || 1,
          target_version: item.target_version || 1,
          reason: reason.trim(),
        }),
      });
      setReason("");
      load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setWorking("");
    }
  }
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>반복 주행의 중복 후보</h2>
          <p>
            근접 위치만으로 자동 합치지 않습니다. 근거를 확인한 후 결정하세요.
          </p>
        </div>
        <button
          className="icon-button"
          onClick={load}
          aria-label="중복 후보 새로고침"
        >
          <RefreshCw size={17} />
        </button>
      </div>
      {error && <ErrorBox message={error} />}
      <div className="panel-padding">
        <label>
          검토 사유
          <input
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="동일 대상 여부를 판단한 근거"
            maxLength={500}
          />
        </label>
      </div>
      {busy ? (
        <div className="loading">
          <LoaderCircle className="spin" />
          후보를 불러오는 중
        </div>
      ) : !items.length ? (
        <Empty
          title="검토할 중복 후보가 없습니다"
          text="반복 주행에서 연결 가능성이 있는 자료가 발견되면 표시됩니다."
        />
      ) : (
        <div className="duplicate-list">
          {items.map((item) => (
            <div className="duplicate-card" key={item.id}>
              <div className="duplicate-icon">
                <Link2 />
              </div>
              <div className="duplicate-description">
                <strong>
                  {item.source_case_id?.slice(0, 8)} <ArrowRight size={14} />{" "}
                  {item.target_case_id?.slice(0, 8)}
                </strong>
                <p>
                  관측점 거리{" "}
                  {item.distance_m == null
                    ? "미확인"
                    : `${item.distance_m.toFixed(1)} m`}{" "}
                  · {item.status}
                </p>
              </div>
              <button
                className="button secondary"
                disabled={!reason.trim() || !!working || user.role === "viewer"}
                onClick={() => decide(item, "rejected")}
              >
                다른 대상
              </button>
              <button
                className="button primary"
                disabled={!reason.trim() || !!working || user.role === "viewer"}
                onClick={() => decide(item, "approved")}
              >
                동일 대상 연결
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
