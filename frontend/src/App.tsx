import { useCallback, useEffect, useRef, useState } from "react";
import {
  LayoutDashboard,
  ClipboardCheck,
  MapPin,
  Link2,
  ImagePlus,
  FileText,
  Settings,
  Route,
  LogOut,
  Menu,
  ChevronRight,
  RefreshCw,
  TriangleAlert,
  ShieldCheck,
} from "lucide-react";
import { ApiError, request } from "./api";
import type { ListResult, Observation, Summary, User } from "./types";
import { statusLabels, id, typeName, synthetic } from "./labels";
import { ErrorBox } from "./components/UI";
import Detail from "./components/Detail";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Inspections from "./pages/Inspections";
import Reports from "./pages/Reports";
import Duplicates from "./pages/Duplicates";
import ServiceStatus from "./pages/ServiceStatus";
import MediaReview from "./pages/MediaReview";
import ReferencePreview from "./pages/ReferencePreview";
import Accounts from "./pages/Accounts";
import Devices from "./pages/Devices";
import Registry from "./pages/Registry";
import PrivacyRequests from "./pages/PrivacyRequests";
import { Runs, Audit, AnalysisStatus } from "./pages/Operations";

const navItems = [
  { id: "overview", label: "점검 개요", icon: LayoutDashboard },
  { id: "inspections", label: "점검 목록", icon: ClipboardCheck },
  { id: "map", label: "관측 지도", icon: MapPin },
  { id: "duplicates", label: "중복 후보", icon: Link2 },
  { id: "media", label: "사진 공유 검수", icon: ImagePlus },
  { id: "reference", label: "로컬 참고사진", icon: ImagePlus },
  { id: "reports", label: "보고서", icon: FileText },
  { id: "settings", label: "서비스 상태", icon: Settings },
  { id: "runs", label: "주행 기록", icon: Route },
  { id: "analysis", label: "분석 운영", icon: Settings },
  { id: "profiles", label: "장착 프로파일", icon: Settings },
  { id: "models", label: "모델 버전", icon: Settings },
  { id: "devices", label: "수집 기기", icon: Settings },
  { id: "accounts", label: "계정 관리", icon: ShieldCheck },
  { id: "privacy", label: "개인정보 요청", icon: ShieldCheck },
  { id: "audit", label: "감사기록", icon: ShieldCheck },
];

export default function App() {
  const loadVersion = useRef(0);
  const [token, setToken] = useState(""),
    [user, setUser] = useState<User | null>(null),
    [page, setPage] = useState("overview"),
    [mobile, setMobile] = useState(false),
    [isCompact, setIsCompact] = useState(
      () => window.matchMedia("(max-width: 700px)").matches,
    ),
    [items, setItems] = useState<Observation[]>([]),
    [summary, setSummary] = useState<Summary | null>(null),
    [loading, setLoading] = useState(false),
    [error, setError] = useState(""),
    [selected, setSelected] = useState<Observation | null>(null),
    [search, setSearch] = useState(""),
    [type, setType] = useState(""),
    [status, setStatus] = useState(""),
    [quality, setQuality] = useState(""),
    [dataScope, setDataScope] = useState("presentation"),
    [period, setPeriod] = useState({ from: "", to: "", bbox: "" }),
    [nextCursor, setNextCursor] = useState<string | null>(null),
    [moreLoading, setMoreLoading] = useState(false);
  const logout = useCallback(() => {
    loadVersion.current++;
    if (token)
      request(token, "/auth/logout", { method: "POST" }).catch(() => {});
    setToken("");
    setUser(null);
    setSelected(null);
    setItems([]);
    setSummary(null);
    setDataScope("presentation");
    setPage("overview");
  }, [token]);
  const load = useCallback(() => {
    if (!token) return;
    const version = ++loadVersion.current;
    setLoading(true);
    setError("");
    const q = new URLSearchParams({ limit: "50", data_scope: dataScope });
    if (type) q.set("type", type);
    if (status) q.set("review_status", status);
    if (quality) q.set("quality", quality);
    if (period.from) q.set("from", new Date(period.from).toISOString());
    if (period.to) q.set("to", new Date(period.to).toISOString());
    if (period.bbox) q.set("bbox", period.bbox);
    Promise.all([
      request<ListResult>(token, `/observations?${q}`),
      request<Summary>(
        token,
        `/summary?${new URLSearchParams(Object.fromEntries([["data_scope", dataScope], ...(period.from ? [["from", new Date(period.from).toISOString()]] : []), ...(period.to ? [["to", new Date(period.to).toISOString()]] : [])]))}`,
      ),
    ])
      .then(([d, s]) => {
        if (version !== loadVersion.current) return;
        setItems(d.items || []);
        setNextCursor(d.next_cursor || null);
        setSummary(s);
      })
      .catch((e) => {
        if (version !== loadVersion.current) return;
        if (e instanceof ApiError && e.status === 401) {
          logout();
          return;
        }
        setError(e.message);
      })
      .finally(() => {
        if (version === loadVersion.current) setLoading(false);
      });
  }, [token, type, status, quality, period, dataScope, logout]);
  useEffect(load, [load]);
  useEffect(() => {
    const media = window.matchMedia("(max-width: 700px)");
    const update = () => setIsCompact(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  const onSelect = useCallback((o: Observation) => setSelected(o), []),
    onClose = useCallback(() => setSelected(null), []);
  async function more() {
    if (!nextCursor) return;
    const version = loadVersion.current;
    setMoreLoading(true);
    try {
      const q = new URLSearchParams({
        limit: "50",
        cursor: nextCursor,
        data_scope: dataScope,
      });
      if (type) q.set("type", type);
      if (status) q.set("review_status", status);
      if (quality) q.set("quality", quality);
      if (period.from) q.set("from", new Date(period.from).toISOString());
      if (period.to) q.set("to", new Date(period.to).toISOString());
      if (period.bbox) q.set("bbox", period.bbox);
      const d = await request<ListResult>(token, `/observations?${q}`);
      if (version !== loadVersion.current) return;
      setItems((x) => [
        ...x,
        ...d.items.filter((o) => !x.some((p) => id(p) === id(o))),
      ]);
      setNextCursor(d.next_cursor || null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setMoreLoading(false);
    }
  }
  if (!token || !user)
    return (
      <Login
        onLogin={(t, u) => {
          setToken(t);
          setUser(u);
        }}
      />
    );
  const filtered = items.filter((o) =>
    `${id(o)} ${o.case_id} ${typeName(o)} ${statusLabels[o.review?.status] || ""}`
      .toLowerCase()
      .includes(search.toLowerCase()),
  );
  const current = navItems.find((n) => n.id === page)!;
  const showFilters = ["overview", "inspections", "map"].includes(page);
  const demo = items.some(synthetic);
  const clear = () => {
    setType("");
    setStatus("");
    setQuality("");
    setDataScope("presentation");
    setSearch("");
    setPeriod({ from: "", to: "", bbox: "" });
  };
  return (
    <div className="workspace">
      <aside
        id="workspace-navigation"
        className={`sidebar ${mobile ? "open" : ""}`}
        aria-hidden={isCompact && !mobile}
        inert={isCompact && !mobile}
      >
        <div className="brand">
          <span className="brand-symbol">
            <Route size={22} />
          </span>
          <span>
            Road Insight<small>도로점검 워크스페이스</small>
          </span>
        </div>
        <div className="workspace-label">
          <span className="org-avatar">R</span>
          <div>
            <strong>
              {user.org_id === "demo"
                ? "시연 워크스페이스"
                : "기관 워크스페이스"}
            </strong>
            <small>ROAD MANAGEMENT</small>
          </div>
        </div>
        <p className="nav-caption">WORKSPACE</p>
        <nav aria-label="주 메뉴">
          {navItems
            .filter(
              (n) =>
                !(
                  ["accounts", "privacy", "audit"].includes(n.id) &&
                  user.role !== "admin"
                ) &&
                !(["devices", "runs"].includes(n.id) && user.role === "viewer"),
            )
            .map((n) => (
              <button
                key={n.id}
                className={page === n.id ? "active" : ""}
                onClick={() => {
                  setPage(n.id);
                  setMobile(false);
                }}
              >
                <n.icon size={18} />
                {n.label}
                {page === n.id && <ChevronRight size={15} />}
              </button>
            ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="privacy-mini">
            <ShieldCheck size={20} />
            <p>
              안전한 자료 공유<small>비식별화 · 검수 · 이력 관리</small>
            </p>
          </div>
          <button
            className="account-button"
            onClick={logout}
            aria-label="로그아웃"
          >
            <span className="user-avatar">
              {(user.login || "A")[0].toUpperCase()}
            </span>
            <span>
              <b>{user.login || "담당자"}</b>
              <small>
                {(
                  {
                    admin: "관리자",
                    reviewer: "검토 담당자",
                    viewer: "조회 사용자",
                  } as any
                )[user.role] || user.role}
              </small>
            </span>
            <LogOut size={16} />
          </button>
        </div>
      </aside>
      {mobile && (
        <button
          className="mobile-shade"
          aria-label="메뉴 닫기"
          onClick={() => setMobile(false)}
        />
      )}
      <div className="main-wrap">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              onClick={() => setMobile(true)}
              aria-label="메뉴 열기"
              aria-controls="workspace-navigation"
              aria-expanded={mobile}
            >
              <Menu size={21} />
            </button>
            <span>워크스페이스</span>
            <ChevronRight size={13} />
            <strong>{current.label}</strong>
          </div>
          <div className="topbar-right">
            <span className="scope">
              <i />
              주간 · 건조 노면
            </span>
            <span className="top-date">
              {new Intl.DateTimeFormat("ko-KR", {
                timeZone: "Asia/Seoul",
                year: "numeric",
                month: "long",
                day: "numeric",
              }).format(new Date())}
            </span>
          </div>
        </header>
        <main className="main-content">
          <div className="page-heading">
            <div>
              <span className="eyebrow">ROAD INSPECTION WORKSPACE</span>
              <h1>{current.label}</h1>
              <p>
                {
                  (
                    {
                      overview:
                        "관측에서 조치까지, 오늘 확인할 도로의 상태를 한눈에 살펴보세요.",
                      inspections:
                        "근거 사진과 위치 품질을 확인하고 검토 결과를 기록하세요.",
                      map: "차량에서 실제 관측한 위치를 지도 위에서 확인하세요.",
                      duplicates:
                        "반복 주행에서 수집한 후보의 동일성을 검토하세요.",
                      media:
                        "개인정보를 확인하고 승인된 사진만 서버에 공유하세요.",
                      reference:
                        "연구 참고사진을 서버에 전송하지 않고 이 브라우저에서 검토하세요.",
                      reports: "점검과 조치 이력을 필요한 형식으로 내보내세요.",
                      settings: "각 서비스의 연결과 운영 상태를 확인하세요.",
                      runs: "주행별 사전 점검과 경로·위치 품질을 확인하세요.",
                      analysis:
                        "로컬 취득과 모델 분석의 실제 준비 상태를 확인하세요.",
                      profiles:
                        "차종별 ZED 2i 장착 정보와 실측 보정값을 관리하세요.",
                      models:
                        "재현 가능한 가중치·전처리·임계값 버전을 등록하세요.",
                      devices:
                        "기관 수집 기기의 활성 상태와 자격증명을 관리하세요.",
                      accounts: "사용자 역할과 계정 접근을 관리하세요.",
                      privacy:
                        "열람·정정·삭제 요청의 접수와 처리 결과를 기록하세요.",
                      audit: "기관 내 중요 변경의 최소 감사기록을 조회하세요.",
                    } as any
                  )[page]
                }
              </p>
            </div>
            {showFilters && (
              <button
                className="button secondary"
                onClick={load}
                disabled={loading}
              >
                <RefreshCw size={16} className={loading ? "spin" : ""} />
                새로고침
              </button>
            )}
          </div>
          {demo && showFilters && (
            <div className="demo-banner">
              <TriangleAlert size={16} />
              현재 목록에 가상 시연 자료가 포함되어 있습니다. 실제 도로 탐지
              성능이나 실증 결과가 아닙니다.
            </div>
          )}
          {error && showFilters && <ErrorBox message={error} onRetry={load} />}
          {page === "overview" && (
            <Dashboard
              summary={summary}
              items={items}
              filtered={filtered}
              onSelect={onSelect}
              setPage={setPage}
            />
          )}
          {showFilters && (
            <Inspections
              page={page}
              items={items}
              filtered={filtered}
              search={search}
              setSearch={setSearch}
              type={type}
              setType={setType}
              status={status}
              setStatus={setStatus}
              quality={quality}
              dataScope={dataScope}
              setDataScope={setDataScope}
              period={period}
              setPeriod={setPeriod}
              setQuality={setQuality}
              clear={clear}
              loading={loading}
              onSelect={onSelect}
              nextCursor={nextCursor}
              moreLoading={moreLoading}
              more={more}
            />
          )}
          {page === "duplicates" && <Duplicates token={token} user={user} />}{" "}
          {page === "reports" && (
            <Reports
              token={token}
              initialFilters={{
                ...period,
                type,
                review_status: status,
                quality,
                data_scope: dataScope,
              }}
            />
          )}{" "}
          {page === "settings" && <ServiceStatus token={token} />}{" "}
          {page === "media" && <MediaReview token={token} user={user} />}
          {page === "reference" && <ReferencePreview />}
          {page === "accounts" && <Accounts token={token} user={user} />}
          {page === "devices" && <Devices token={token} user={user} />}
          {page === "profiles" && (
            <Registry
              key="profiles"
              token={token}
              user={user}
              kind="profiles"
            />
          )}
          {page === "models" && (
            <Registry
              key="models"
              token={token}
              user={user}
              kind="model-bundles"
            />
          )}
          {page === "privacy" && <PrivacyRequests token={token} user={user} />}
          {page === "runs" && <Runs token={token} />}
          {page === "audit" && <Audit token={token} user={user} />}
          {page === "analysis" && <AnalysisStatus token={token} user={user} />}
          <footer className="page-footer">
            <span>
              {page === "reference" ? (
                <TriangleAlert size={14} />
              ) : (
                <ShieldCheck size={14} />
              )}
              {page === "reference"
                ? "로컬 참고사진 · 개인정보 검수 및 공유 승인 없음 · AI 분석 미실행"
                : "승인된 비식별 자료를 기관 권한 안에서 관리합니다."}
            </span>
            <span>Road Insight · MVP</span>
          </footer>
        </main>
      </div>
      {selected && (
        <Detail
          observation={selected}
          token={token}
          user={user}
          onClose={onClose}
          onUpdated={load}
        />
      )}
    </div>
  );
}
