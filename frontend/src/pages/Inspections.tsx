import { Search, X, LoaderCircle } from "lucide-react";
import type { Observation } from "../types";
import { id, date, typeName, statusLabels } from "../labels";
import { Badge, Empty } from "../components/UI";
import ObservationMap from "../components/ObservationMap";
import ObservationTable from "../components/ObservationTable";
import { useState } from "react";
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
  dataScope: string;
  setDataScope: (v: string) => void;
  setQuality: (v: string) => void;
  period: { from: string; to: string; bbox: string };
  setPeriod: (v: { from: string; to: string; bbox: string }) => void;
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
  dataScope,
  setDataScope,
  setQuality,
  period,
  setPeriod,
  clear,
  loading,
  onSelect,
  nextCursor,
  moreLoading,
  more,
}: Props) {
  const [draft, setDraft] = useState(period),
    [filterError, setFilterError] = useState("");
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
        <label className="filter-select">
          <span className="sr-only">자료 출처 범위</span>
          <select
            aria-label="자료 출처 범위"
            value={dataScope}
            onChange={(e) => setDataScope(e.target.value)}
          >
            <option value="presentation">시연 목록 · HTTP 테스트 제외</option>
            <option value="all">전체 기록 · HTTP 테스트 포함</option>
            <option value="test">합성 HTTP 테스트만</option>
            <option value="synthetic">합성 시연 자료만</option>
            <option value="field">일반 관측 · 출처 미확인</option>
          </select>
        </label>
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
          <select
            aria-label="관측 종류"
            value={type}
            onChange={(e) => setType(e.target.value)}
          >
            <option value="">모든 종류</option>
            <option value="pothole">포트홀</option>
            <option value="lane_segment">차선 마모</option>
          </select>
        </label>
        <label className="filter-select">
          <span className="sr-only">검토 상태 필터</span>
          <select
            aria-label="검토 상태 필터"
            value={status}
            onChange={(e) => setStatus(e.target.value)}
          >
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
          <select
            aria-label="위치 품질 필터"
            value={quality}
            onChange={(e) => setQuality(e.target.value)}
          >
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
      <details className="advanced-filters">
        <summary>
          기간·구역 필터
          {period.from || period.to || period.bbox ? " · 적용 중" : ""}
        </summary>
        <form
          className="management-form"
          onSubmit={(e) => {
            e.preventDefault();
            setFilterError("");
            if (
              draft.from &&
              draft.to &&
              new Date(draft.from) >= new Date(draft.to)
            ) {
              setFilterError("종료 시각은 시작 시각보다 뒤여야 합니다.");
              return;
            }
            if (draft.bbox) {
              const values = draft.bbox.split(",").map(Number);
              if (
                values.length !== 4 ||
                values.some((x) => !Number.isFinite(x)) ||
                values[0] < -180 ||
                values[2] > 180 ||
                values[1] < -90 ||
                values[3] > 90 ||
                values[0] > values[2] ||
                values[1] > values[3]
              ) {
                setFilterError(
                  "구역은 서경도,남위도,동경도,북위도 순의 유효한 좌표 4개를 입력하세요.",
                );
                return;
              }
              if (quality === "unavailable") {
                setFilterError(
                  "위치 없음 필터와 좌표 구역은 함께 적용할 수 없습니다.",
                );
                return;
              }
            }
            setPeriod({ ...draft });
          }}
        >
          <div className="form-grid">
            <label>
              촬영 시작 (현지 시각)
              <input
                type="datetime-local"
                value={draft.from}
                onChange={(e) => setDraft({ ...draft, from: e.target.value })}
              />
            </label>
            <label>
              촬영 종료 (미포함)
              <input
                type="datetime-local"
                value={draft.to}
                onChange={(e) => setDraft({ ...draft, to: e.target.value })}
              />
            </label>
          </div>
          <label>
            구역 WGS84 경계 (서경도,남위도,동경도,북위도)
            <input
              value={draft.bbox}
              placeholder="126.9,37.5,127.1,37.7"
              onChange={(e) => setDraft({ ...draft, bbox: e.target.value })}
            />
          </label>
          {filterError && <p role="alert">{filterError}</p>}
          <div className="form-actions">
            <button className="button secondary">기간·구역 적용</button>
            <button
              type="button"
              className="button secondary"
              onClick={() => {
                setDraft({ from: "", to: "", bbox: "" });
                setPeriod({ from: "", to: "", bbox: "" });
                setFilterError("");
              }}
            >
              기간·구역 해제
            </button>
          </div>
        </form>
      </details>
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
