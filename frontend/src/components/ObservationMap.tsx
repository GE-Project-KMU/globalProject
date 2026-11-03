import { useEffect, useRef } from "react";
import L from "leaflet";
import type { Observation } from "../types";
import { id, typeName, statusLabels } from "../labels";

export default function ObservationMap({
  items,
  onSelect,
  selected,
}: {
  items: Observation[];
  onSelect: (o: Observation) => void;
  selected?: string;
}) {
  const host = useRef<HTMLDivElement>(null),
    map = useRef<L.Map | null>(null),
    layer = useRef<L.LayerGroup | null>(null);
  useEffect(() => {
    if (!host.current) return;
    const m = L.map(host.current, {
      attributionControl: false,
      scrollWheelZoom: false,
    }).setView([37.55, 126.98], 12);
    map.current = m;
    layer.current = L.layerGroup().addTo(m);
    let mounted = true;
    const observer = new ResizeObserver(() => {
      if (mounted) m.invalidateSize({ animate: false, pan: false });
    });
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      mounted = false;
      m.stop();
      m.remove();
      map.current = null;
      layer.current = null;
    };
  }, []);
  useEffect(() => {
    const m = map.current,
      g = layer.current;
    if (!m || !g) return;
    g.clearLayers();
    const points: L.LatLngTuple[] = [];
    items.forEach((o) => {
      if (o.location?.lat == null || o.location?.lon == null) return;
      const point: L.LatLngTuple = [o.location.lat, o.location.lon];
      points.push(point);
      const color =
        o.type === "pothole"
          ? "#db743c"
          : o.lane?.grade === "unassessable"
            ? "#84928c"
            : "#257e70";
      L.circleMarker(point, {
        radius: selected === id(o) ? 11 : 7,
        color: "#fff",
        weight: 3,
        fillColor: color,
        fillOpacity: 1,
      })
        .addTo(g)
        .bindTooltip(
          `${typeName(o)} · ${statusLabels[o.review?.status] || "미확인"}`,
          { direction: "top" },
        )
        .on("click", () => onSelect(o));
    });
    if (points.length)
      m.fitBounds(L.latLngBounds(points).pad(0.25), {
        maxZoom: 16,
        animate: false,
      });
  }, [items, onSelect, selected]);
  return (
    <div className="map-wrap">
      <div
        ref={host}
        className="map-canvas"
        role="img"
        aria-label={`관측 위치 지도, 좌표가 있는 관측 ${items.filter((o) => o.location?.lat != null).length}건`}
      />
      <div className="map-key">
        <span>
          <i className="dot orange" />
          포트홀
        </span>
        <span>
          <i className="dot green" />
          차선
        </span>
        <span>차량 관측 위치 · WGS84</span>
      </div>
      <div className="map-caption">
        배경지도 미연결 · 관측 좌표만 표시합니다
      </div>
      {!items.some((o) => o.location?.lat != null) && (
        <div className="map-no-points">표시할 관측 좌표가 없습니다</div>
      )}
    </div>
  );
}
