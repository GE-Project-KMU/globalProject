import {
  Route,
  ClipboardCheck,
  CheckCircle2,
  TriangleAlert,
  ArrowRight,
  ChevronRight,
  ShieldCheck,
} from "lucide-react";
import type { Observation, Summary } from "../types";
import { date, id, typeName } from "../labels";
import ObservationMap from "../components/ObservationMap";
import TrendChart from "../components/TrendChart";
interface Props {
  summary: Summary | null;
  items: Observation[];
  filtered: Observation[];
  onSelect: (o: Observation) => void;
  setPage: (page: string) => void;
}
export default function Dashboard({
  summary,
  items,
  filtered,
  onSelect,
  setPage,
}: Props) {
  return (
    <>
      <div className="stats-grid">
        {[
          {
            label: "전체 관측",
            value: summary?.total,
            sub: "등록된 관측",
            icon: Route,
            color: "green",
          },
          {
            label: "검토 대기",
            value: summary?.pending,
            sub: "담당자 확인 필요",
            icon: ClipboardCheck,
            color: "orange",
          },
          {
            label: "조치 완료",
            value: summary?.completed,
            sub: "처리 이력 기록",
            icon: CheckCircle2,
            color: "blue",
          },
          {
            label: "판정 불가 차선",
            value: summary?.unassessable,
            sub: "추가 근거 · 재촬영",
            icon: TriangleAlert,
            color: "gray",
          },
        ].map((s) => (
          <section className="stat-card" key={s.label}>
            <div>
              <span>{s.label}</span>
              <span className={`stat-icon ${s.color}`}>
                <s.icon size={19} />
              </span>
            </div>
            <strong>
              {s.value == null ? "—" : s.value.toLocaleString()}
              <small>건</small>
            </strong>
            <p>{s.sub}</p>
          </section>
        ))}
      </div>
      <TrendChart summary={summary} />
      <div className="overview-grid">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>관측 위치</h2>
              <p>현재 목록의 유효 좌표</p>
            </div>
            <button className="text-button" onClick={() => setPage("map")}>
              전체 지도
              <ArrowRight size={15} />
            </button>
          </div>
          <ObservationMap items={filtered} onSelect={onSelect} />
        </section>
        <section className="panel attention-panel">
          <span className="eyebrow">REVIEW QUEUE</span>
          <h2>확인이 필요한 관측</h2>
          <p className="muted">사진과 현장 근거를 바탕으로 판단하세요.</p>
          {items
            .filter((o) => o.review.status === "unreviewed")
            .slice(0, 4)
            .map((o) => (
              <button
                className="attention-item"
                key={id(o)}
                onClick={() => onSelect(o)}
              >
                <span className={`type-icon ${o.type}`}>
                  <Route size={18} />
                </span>
                <span>
                  <b>{typeName(o)}</b>
                  <small>{date(o.captured_at)}</small>
                </span>
                <ChevronRight size={16} />
              </button>
            ))}
          {!items.some((o) => o.review.status === "unreviewed") && (
            <div className="queue-clear">
              <CheckCircle2 size={27} />
              <p>확인 대기 중인 관측이 없습니다.</p>
            </div>
          )}
          <div className="attention-foot">
            <ShieldCheck size={16} />
            자동 판단을 현장 검토로 연결합니다
          </div>
        </section>
      </div>
    </>
  );
}
