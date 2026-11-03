import { useCallback, useEffect, useState, type ReactNode } from "react";
import { request } from "../api";
import { Empty, ErrorBox } from "./UI";

export function useCollection(token: string, path: string, enabled = true) {
  const [items, setItems] = useState<any[]>([]),
    [busy, setBusy] = useState(false),
    [quarantinedCount, setQuarantinedCount] = useState(0),
    [error, setError] = useState("");
  const load = useCallback(async () => {
    if (!enabled) return;
    setBusy(true);
    setError("");
    try {
      const data = await request<any>(token, path);
      setItems(data.items || []);
      setQuarantinedCount(
        Number.isSafeInteger(data.quarantined_count) &&
          data.quarantined_count > 0
          ? data.quarantined_count
          : 0,
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [token, path, enabled]);
  useEffect(() => {
    void load();
  }, [load]);
  return { items, busy, error, load, setError, quarantinedCount };
}
export function Collection({
  title,
  description,
  state,
  children,
}: {
  title: string;
  description: string;
  state: ReturnType<typeof useCollection>;
  children: ReactNode;
}) {
  return (
    <section className="panel management-panel">
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
        <button
          className="button secondary"
          onClick={() => void state.load()}
          disabled={state.busy}
        >
          새로고침
        </button>
      </div>
      {state.error && <ErrorBox message={state.error} onRetry={state.load} />}
      {state.busy ? (
        <p role="status">불러오는 중…</p>
      ) : state.items.length ? (
        children
      ) : (
        <Empty
          title="등록된 자료가 없습니다"
          text="새 자료를 등록하거나 조회 조건을 확인하세요."
        />
      )}
    </section>
  );
}
export function JsonDetails({
  value,
  label = "등록 내용 보기",
}: {
  value: unknown;
  label?: string;
}) {
  return (
    <details className="json-details">
      <summary>{label}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}
export function Feedback({
  error,
  message,
}: {
  error: string;
  message: string;
}) {
  return (
    <>
      {error && <ErrorBox message={error} />}
      {message && (
        <p className="success-message" role="status">
          {message}
        </p>
      )}
    </>
  );
}
