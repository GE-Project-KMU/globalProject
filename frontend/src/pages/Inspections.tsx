import { Search, X, LoaderCircle } from "lucide-react";
import type { Observation } from "../types";
import { id, date, typeName, statusLabels } from "../labels";
import { Badge, Empty } from "../components/UI";
import ObservationMap from "../components/ObservationMap";
import ObservationTable from "../components/ObservationTable";
interface Props {
  page: string;
  items: Observation[];
  filtered: Observation[];
  search: string;
  setSearch: (v: string) => void;
  type: string;
  setType: (v: string) => void;
  status: string;
  setStatus: (v: string) => void;
  quality: string;
  setQuality: (v: string) => void;
  clear: () => void;
  loading: boolean;
  onSelect: (o: Observation) => void;
  nextCursor: string | null;
  moreLoading: boolean;
  more: () => void;
}
export default function Inspections({
  page,
  items,
  filtered,
  search,
  setSearch,
  type,
  setType,
  status,
  setStatus,
  quality,
  setQuality,
  clear,
  loading,
  onSelect,
  nextCursor,
  moreLoading,
  more,
}: Props) {
  return (
    <section className="panel inspection-panel">
      <div className="panel-heading">
        <div>
          <h2>
            {page === "overview" ? "최근 관측" : "관측 자료"}
            <span className="count-label">{filtered.length}</span>
          </h2>
          <p>포트홀·마모차선의 검토 상태와 근거</p>
        </div>
      </div>
      <div className="filter-bar">
        <div className="search-field">
          <Search size={16} />
          <input
            aria-label="관측 검색"
            placeholder="관측 ID, 종류, 상태 검색"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <label className="filter-select">
          <span className="sr-only">관측 종류</span>
          <select value={type} onChange={(e) => setType(e.target.value)}>
            <option value="">모든 종류</option>
            <option value="pothole">포트홀</option>
            <option value="lane_segment">차선 마모</option>
          </select>
        </label>
        <label className="filter-select">
          <span className="sr-only">검토 상태 필터</span>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">모든 상태</option>
            {Object.entries(statusLabels).map(([v, l]) => (
              <option key={v} value={v}>
                {l}
              </option>
            ))}
          </select>
        </label>
        <label className="filter-select">
          <span className="sr-only">위치 품질 필터</span>
          <select value={quality} onChange={(e) => setQuality(e.target.value)}>
            <option value="">모든 위치 품질</option>
            <option value="valid">유효</option>
            <option value="degraded">품질 저하</option>
            <option value="unavailable">위치 없음</option>
          </select>
        </label>
        {(search || type || status || quality) && (
          <button
            className="icon-button"
            onClick={clear}
            aria-label="필터 초기화"
          >
            <X size={17} />
          </button>
        )}
      </div>
      {loading ? (
        <div className="loading">
          <LoaderCircle className="spin" size={21} />
          관측 자료를 불러오는 중입니다
        </div>
      ) : page === "map" ? (
        <div className="full-map">
          <ObservationMap items={filtered} onSelect={onSelect} />
          <div className="map-list">
            {filtered.length ? (
              filtered.map((o) => (
                <button key={id(o)} onClick={() => onSelect(o)}>
                  <b>{typeName(o)}</b>
                  <Badge status={o.review.status} />
                  <small>
                    {date(o.captured_at)}
                    {o.location?.lat == null ? " · 위치 없음" : ""}
                  </small>
                </button>
              ))
            ) : (
              <Empty title="표시할 관측이 없습니다" action={clear} />
            )}
          </div>
        </div>
      ) : filtered.length ? (
        <ObservationTable items={filtered} onSelect={onSelect} />
      ) : (
        <Empty
          title={
            search || type || status || quality
              ? "조건에 맞는 관측이 없습니다"
              : undefined
          }
          text={
            search || type || status || quality
              ? "검색어와 필터를 바꾸어 다시 확인해 주세요."
              : undefined
          }
          action={search || type || status || quality ? clear : undefined}
        />
      )}
      <div className="table-footer">
        <span>현재 불러온 관측 {items.length}건 · KST 기준</span>
        {nextCursor && (
          <button
            className="button secondary"
            onClick={more}
            disabled={moreLoading}
          >
            {moreLoading ? <LoaderCircle size={15} className="spin" /> : null}더
            불러오기
          </button>
        )}
        <span>위치는 차량 관측점입니다</span>
      </div>
    </section>
  );
}
