import { useState, type FormEvent } from "react";
import { request } from "../api";
import type { User } from "../types";
import { Collection, Feedback, useCollection } from "../components/Management";
export default function Devices({
  token,
  user,
}: {
  token: string;
  user: User;
}) {
  const state = useCollection(token, "/devices", user.role !== "viewer");
  const [name, setName] = useState(""),
    [editing, setEditing] = useState<any>(null),
    [secret, setSecret] = useState(""),
    [reason, setReason] = useState(""),
    [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false);
  if (user.role === "viewer")
    return (
      <p className="panel">
        기기 조회는 관리자 또는 검토 담당자 권한이 필요합니다.
      </p>
    );
  async function mutate(path: string, method: string, body: any) {
    setBusy(true);
    setError("");
    setMessage("");
    setSecret("");
    try {
      const d = await request<any>(token, path, {
        method,
        body: JSON.stringify(body),
      });
      setSecret(d.device_token || "");
      setMessage("기기 설정을 저장했습니다.");
      setEditing(null);
      setName("");
      setReason("");
      await state.load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function save(e: FormEvent) {
    e.preventDefault();
    void mutate(
      editing ? `/devices/${editing.id}` : "/devices",
      editing ? "PATCH" : "POST",
      editing
        ? { name: name.trim(), active: editing.active }
        : { name: name.trim() },
    );
  }
  return (
    <>
      <Collection
        title="수집 기기"
        description="ZED 2i 수집 에이전트의 기관별 자격증명을 관리합니다."
        state={state}
      >
        <div className="management-list">
          {state.items.map((x) => (
            <article key={x.id}>
              <div>
                <h3>{x.name}</h3>
                <p>{x.active ? "활성" : "비활성"}</p>
                <code>{x.id}</code>
              </div>
              {user.role === "admin" && (
                <button
                  className="button secondary"
                  onClick={() => {
                    setEditing({ ...x });
                    setName(x.name);
                    setSecret("");
                  }}
                >
                  기기 편집
                </button>
              )}
            </article>
          ))}
        </div>
      </Collection>
      {user.role === "admin" && (
        <section className="panel management-panel">
          <h2>{editing ? "기기 변경" : "기기 등록"}</h2>
          <Feedback error={error} message={message} />
          {secret && (
            <div className="secret-box">
              <strong>기기 토큰 · 이번에만 표시됩니다</strong>
              <p>
                신뢰하는 로컬 에이전트 설정에 보관하세요. 화면을 떠나면 다시
                조회할 수 없습니다.
              </p>
              <code>{secret}</code>
              <button
                className="button secondary"
                onClick={() => setSecret("")}
              >
                토큰 화면에서 지우기
              </button>
            </div>
          )}
          <form className="management-form" onSubmit={save}>
            <label>
              기기 이름
              <input
                required
                maxLength={100}
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </label>
            {editing && (
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={editing.active}
                  onChange={(e) =>
                    setEditing({ ...editing, active: e.target.checked })
                  }
                />
                기기 활성
              </label>
            )}
            <div className="form-actions">
              <button className="button primary" disabled={busy}>
                기기 저장
              </button>
              {editing && (
                <button
                  type="button"
                  className="button secondary"
                  onClick={() => {
                    setEditing(null);
                    setName("");
                  }}
                >
                  편집 취소
                </button>
              )}
            </div>
          </form>
          {editing && (
            <form
              className="management-form"
              onSubmit={(e) => {
                e.preventDefault();
                void mutate(`/devices/${editing.id}/credentials`, "POST", {
                  reason: reason.trim(),
                });
              }}
            >
              <h3>자격증명 교체</h3>
              <p>
                기존 토큰은 즉시 폐기됩니다. 로컬 에이전트 설정도 교체하세요.
              </p>
              <label>
                교체 사유
                <input
                  required
                  maxLength={500}
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                />
              </label>
              <button
                className="button secondary"
                disabled={busy || !reason.trim()}
              >
                새 토큰 발급
              </button>
            </form>
          )}
        </section>
      )}
    </>
  );
}
