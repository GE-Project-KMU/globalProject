import { useCallback, useEffect, useState } from "react";
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
  CheckCircle2,
  ShieldCheck,
  ArrowRight,
  Search,
  X,
  LoaderCircle,
} from "lucide-react";
import { ApiError, request } from "./api";
import type { ListResult, Observation, Summary, User } from "./types";
import { statusLabels, date, id, typeName } from "./labels";
import { Badge, Empty, ErrorBox } from "./components/UI";
import ObservationMap from "./components/ObservationMap";
import ObservationTable from "./components/ObservationTable";
import Detail from "./components/Detail";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Inspections from "./pages/Inspections";
import Reports from "./pages/Reports";
import Duplicates from "./pages/Duplicates";
import ServiceStatus from "./pages/ServiceStatus";
import MediaReview from "./pages/MediaReview";

const navItems = [
  { id: "overview", label: "점검 개요", icon: LayoutDashboard },
  { id: "inspections", label: "점검 목록", icon: ClipboardCheck },
  { id: "map", label: "관측 지도", icon: MapPin },
  { id: "duplicates", label: "중복 후보", icon: Link2 },
  { id: "media", label: "사진 공유 검수", icon: ImagePlus },
  { id: "reports", label: "보고서", icon: FileText },
  { id: "settings", label: "서비스 상태", icon: Settings },
];

export default function App() {
  const [token, setToken] = useState(""),
    [user, setUser] = useState<User | null>(null),
    [page, setPage] = useState("overview"),
    [mobile, setMobile] = useState(false),
    [items, setItems] = useState<Observation[]>([]),
    [summary, setSummary] = useState<Summary | null>(null),
    [loading, setLoading] = useState(false),
    [error, setError] = useState(""),
    [selected, setSelected] = useState<Observation | null>(null),
    [search, setSearch] = useState(""),
    [type, setType] = useState(""),
    [status, setStatus] = useState(""),
    [quality, setQuality] = useState(""),
    [nextCursor, setNextCursor] = useState<string | null>(null),
    [moreLoading, setMoreLoading] = useState(false);
  const logout = useCallback(() => {
    if (token)
      request(token, "/auth/logout", { method: "POST" }).catch(() => {});
    setToken("");
    setUser(null);
    setSelected(null);
    setItems([]);
    setSummary(null);
    setPage("overview");
  }, [token]);
  const load = useCallback(() => {
    if (!token) return;
    setLoading(true);
    setError("");
    const q = new URLSearchParams({ limit: "50" });
    if (type) q.set("type", type);
    if (status) q.set("review_status", status);
    if (quality) q.set("quality", quality);
    Promise.all([
      request<ListResult>(token, `/observations?${q}`),
      request<Summary>(token, "/summary"),
    ])
      .then(([d, s]) => {
        setItems(d.items || []);
        setNextCursor(d.next_cursor || null);
        setSummary(s);
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) {
          logout();
          return;
        }
        setError(e.message);
      })
      .finally(() => setLoading(false));
  }, [token, type, status, quality, logout]);
  useEffect(load, [load]);
  const onSelect = useCallback((o: Observation) => setSelected(o), []),
    onClose = useCallback(() => setSelected(null), []);
  async function more() {
    if (!nextCursor) return;
    setMoreLoading(true);
    try {
      const q = new URLSearchParams({ limit: "50", cursor: nextCursor });
      if (type) q.set("type", type);
      if (status) q.set("review_status", status);
      if (quality) q.set("quality", quality);
      const d = await request<ListResult>(token, `/observations?${q}`);
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
  const demo = items.some((o) => o.demo || o.synthetic);
  const clear = () => {
    setType("");
    setStatus("");
    setQuality("");
    setSearch("");
  };
  return (
    <div className="workspace">
      <aside className={`sidebar ${mobile ? "open" : ""}`}>
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
          {navItems.map((n) => (
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
                      reports: "점검과 조치 이력을 필요한 형식으로 내보내세요.",
                      settings: "각 서비스의 연결과 운영 상태를 확인하세요.",
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
          {page === "reports" && <Reports token={token} />}{" "}
          {page === "settings" && <ServiceStatus token={token} />}{" "}
          {page === "media" && <MediaReview token={token} user={user} />}
          <footer className="page-footer">
            <span>
              <ShieldCheck size={14} />
              승인된 비식별 자료를 기관 권한 안에서 관리합니다.
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
