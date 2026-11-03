import { useState, type FormEvent } from "react";
import { request } from "../api";
import type { User } from "../types";
import { Collection, Feedback, useCollection } from "../components/Management";

const roles: Record<string, string> = {
  admin: "관리자",
  reviewer: "검토 담당자",
  viewer: "조회 사용자",
};
export default function Accounts({
  token,
  user,
}: {
  token: string;
  user: User;
}) {
  const state = useCollection(token, "/users", user.role === "admin");
  const [login, setLogin] = useState(""),
    [password, setPassword] = useState(""),
    [role, setRole] = useState("reviewer"),
    [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false),
    [editing, setEditing] = useState<any>(null);
  if (user.role !== "admin")
    return <p className="panel">계정 관리는 관리자 권한이 필요합니다.</p>;
  async function save(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await request(token, editing ? `/users/${editing.id}` : "/users", {
        method: editing ? "PATCH" : "POST",
        body: JSON.stringify(
          editing
            ? {
                role,
                active: editing.active,
                ...(password ? { password } : {}),
              }
            : { login: login.trim(), password, role },
        ),
      });
      setPassword("");
      setLogin("");
      setEditing(null);
      setMessage(
        "계정 설정을 저장했습니다. 변경된 계정의 기존 세션은 종료됩니다.",
      );
      await state.load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Collection
        title="기관 계정"
        description="기관 내 권한과 사용 상태를 관리합니다. 마지막 관리자는 해제할 수 없습니다."
        state={state}
      >
        <div className="management-list">
          {state.items.map((x) => (
            <article key={x.id}>
              <div>
                <h3>{x.login}</h3>
                <p>
                  {roles[x.role]} · {x.active ? "활성" : "비활성"}
                  {x.id === user.id ? " · 현재 계정" : ""}
                </p>
              </div>
              <button
                className="button secondary"
                onClick={() => {
                  setEditing({ ...x });
                  setLogin(x.login);
                  setRole(x.role);
                  setPassword("");
                  setMessage("");
                }}
              >
                계정 편집
              </button>
            </article>
          ))}
        </div>
      </Collection>
      <section className="panel management-panel">
        <h2>{editing ? "계정 변경" : "계정 추가"}</h2>
        <form className="management-form" onSubmit={save}>
          <label>
            로그인 식별자
            <input
              required
              maxLength={254}
              value={login}
              disabled={!!editing}
              onChange={(e) => setLogin(e.target.value)}
              autoComplete="off"
            />
          </label>
          <label>
            {editing ? "새 비밀번호 (변경할 때만)" : "초기 비밀번호"}
            <input
              type="password"
              required={!editing}
              minLength={12}
              maxLength={128}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
            />
          </label>
          <label>
            권한
            <select value={role} onChange={(e) => setRole(e.target.value)}>
              {Object.entries(roles).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
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
              계정 활성
            </label>
          )}
          <p>권한·상태·비밀번호 변경 시 해당 계정의 모든 세션이 만료됩니다.</p>
          <Feedback error={error} message={message} />
          <div className="form-actions">
            <button className="button primary" disabled={busy}>
              {busy ? "저장 중…" : "계정 저장"}
            </button>
            {editing && (
              <button
                type="button"
                className="button secondary"
                onClick={() => {
                  setEditing(null);
                  setLogin("");
                  setPassword("");
                }}
              >
                편집 취소
              </button>
            )}
          </div>
        </form>
      </section>
    </>
  );
}
