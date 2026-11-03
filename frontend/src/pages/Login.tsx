import { useState } from "react";
import {
  Route,
  MapPin,
  ShieldCheck,
  LoaderCircle,
  ArrowRight,
} from "lucide-react";
import { request } from "../api";
import type { User } from "../types";
import { ErrorBox } from "../components/UI";

export default function Login({
  onLogin,
}: {
  onLogin: (token: string, user: User) => void;
}) {
  const [org, setOrg] = useState("demo"),
    [login, setLogin] = useState("admin@road.local"),
    [password, setPassword] = useState(""),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const d = await request<any>("", "/auth/login", {
        method: "POST",
        body: JSON.stringify({ org_code: org, login, password }),
      });
      onLogin(d.access_token, d.user);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="login-page">
      <section className="login-story">
        <div className="brand">
          <span className="brand-symbol">
            <Route size={23} />
          </span>
          <span>
            Road Insight<small>도로점검 워크스페이스</small>
          </span>
        </div>
        <div className="login-message">
          <span className="eyebrow">OBSERVE. REVIEW. IMPROVE.</span>
          <h1>
            도로의 작은 변화도,
            <br />
            놓치지 않도록.
          </h1>
          <p>
            포트홀과 마모차선의 관측부터
            <br />
            현장 검토와 조치 이력까지 한곳에서 관리하세요.
          </p>
          <div className="login-road">
            <div className="road-line" />
            <MapPin size={38} />
            <span className="road-point" />
          </div>
        </div>
        <div className="login-foot">
          <ShieldCheck size={17} /> 승인된 비식별 자료만 공유합니다
        </div>
      </section>
      <section className="login-form-wrap">
        <form className="login-form" onSubmit={submit}>
          <span className="eyebrow">WELCOME BACK</span>
          <h2>워크스페이스 로그인</h2>
          <p>소속 기관 계정으로 안전하게 시작하세요.</p>
          {error && <ErrorBox message={error} />}
          <label>
            기관 코드
            <input
              autoComplete="organization"
              value={org}
              onChange={(e) => setOrg(e.target.value)}
              required
            />
          </label>
          <label>
            이메일 또는 로그인 ID
            <input
              autoComplete="username"
              value={login}
              onChange={(e) => setLogin(e.target.value)}
              required
            />
          </label>
          <label>
            비밀번호
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>
          <button className="button primary full" disabled={busy}>
            {busy ? <LoaderCircle size={17} className="spin" /> : null}로그인
            <ArrowRight size={17} />
          </button>
          <p className="login-hint">
            초기 계정은 프로젝트 실행 시 생성됩니다.
            <br />
            계정 정보는 프로젝트의 부트스트랩 파일을 확인하세요.
          </p>
        </form>
      </section>
    </main>
  );
}
