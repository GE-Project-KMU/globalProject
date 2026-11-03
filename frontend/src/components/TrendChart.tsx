import { useState } from "react";
import type { Summary } from "../types";

export default function TrendChart({ summary }: { summary: Summary | null }) {
  const [type, setType] = useState<"pothole" | "lane_segment">("pothole");
  if (!summary?.trend_by_type) return null;
  const days = summary.trend_by_type[type] || [];
  const max = Math.max(
    1,
    ...days.flatMap((d) => [d.candidates, d.confirmed, d.completed]),
  );
  const unit = type === "pothole" ? "포트홀 점검 건" : "차선 점검 구간";
  return (
    <section className="panel trend-panel">
      <div className="panel-heading">
        <div>
          <h2>기간별 점검 현황</h2>
          <p>UTC 대표 촬영일별 현재 상태 · {unit}</p>
        </div>
        <label className="trend-type">
          <span className="sr-only">현황 집계 종류</span>
          <select
            aria-label="현황 집계 종류"
            value={type}
            onChange={(e) => setType(e.target.value as typeof type)}
          >
            <option value="pothole">포트홀 점검 건</option>
            <option value="lane_segment">차선 점검 구간</option>
          </select>
        </label>
      </div>
      <div
        className="trend-chart"
        role="img"
        aria-label={`${unit} 날짜별 후보·확인·완료 현황`}
      >
        {days.length ? (
          days.map((day) => (
            <div className="trend-day" key={day.date}>
              <span>{day.date}</span>
              {(
                [
                  ["candidates", "후보"],
                  ["confirmed", "확인"],
                  ["completed", "완료"],
                ] as const
              ).map(([key, label]) => (
                <div className={`trend-bar ${key}`} key={key}>
                  <span>{label}</span>
                  <meter
                    min={0}
                    max={max}
                    value={day[key]}
                    aria-label={`${day.date} ${label} ${day[key]}건`}
                  />
                  <b>{day[key]}</b>
                </div>
              ))}
            </div>
          ))
        ) : (
          <p className="muted">해당 기간에 집계할 점검 건이 없습니다.</p>
        )}
      </div>
      <p className="trend-note">
        해당 날짜의 상태 변경 횟수가 아닙니다. 기간 필터를 적용하며
        종류·위치·검색 필터는 이 차트에 적용되지 않습니다. 반복 관측을 대표 점검
        건으로 묶어 집계합니다.
      </p>
    </section>
  );
}
