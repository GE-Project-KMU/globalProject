import { Route, ChevronRight } from "lucide-react";
import type { Observation } from "../types";
import { date, id, typeName, originLabel } from "../labels";
import { Badge } from "./UI";

export default function ObservationTable({
  items,
  onSelect,
}: {
  items: Observation[];
  onSelect: (o: Observation) => void;
}) {
  return (
    <div className="table-scroll">
      <table className="observation-table">
        <thead>
          <tr>
            <th>관측 대상</th>
            <th>관측 시각</th>
            <th>검토 상태</th>
            <th>위치 품질</th>
            <th>인식 지표</th>
            <th>
              <span className="sr-only">상세 보기</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {items.map((o) => (
            <tr key={id(o)} onClick={() => onSelect(o)}>
              <td>
                <button
                  className="row-target"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelect(o);
                  }}
                >
                  <span className={`type-icon ${o.type}`}>
                    <Route size={17} />
                  </span>
                  <span>
                    <b>{typeName(o)}</b>
                    <small>
                      {id(o).slice(0, 8)}
                      <em>{originLabel(o)}</em>
                    </small>
                  </span>
                </button>
              </td>
              <td>{date(o.captured_at)}</td>
              <td>
                <Badge status={o.review?.status || "unreviewed"} />
                {o.review?.priority === "urgent" && (
                  <span className="urgent-dot" title="긴급 우선순위" />
                )}
              </td>
              <td>
                <span
                  className={`quality ${o.location?.quality === "valid" ? "ok" : ""}`}
                >
                  <i />
                  {(
                    {
                      valid: "유효",
                      degraded: "품질 저하",
                      unavailable: "위치 없음",
                    } as any
                  )[o.location?.quality || ""] || "미기록"}
                </span>
              </td>
              <td>
                {o.type === "pothole" ? (
                  <>
                    <strong>
                      {((o.pothole?.confidence || 0) * 100).toFixed(0)}%
                    </strong>
                    <small className="metric-caption">confidence</small>
                  </>
                ) : o.lane?.retention_pct == null ? (
                  <span className="muted">판정 불가</span>
                ) : (
                  <>
                    <strong>{o.lane.retention_pct.toFixed(1)}%</strong>
                    <small className="metric-caption">도막 잔존율</small>
                  </>
                )}
              </td>
              <td>
                <ChevronRight size={17} className="muted" />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
