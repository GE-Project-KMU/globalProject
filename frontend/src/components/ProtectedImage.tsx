import { useEffect, useState } from "react";
import { Eye } from "lucide-react";

export default function ProtectedImage({
  token,
  path,
  bbox,
  mask,
  outline,
}: {
  token: string;
  path?: string;
  bbox?: number[];
  mask?: { size: number[]; counts: number[] };
  outline?: number[][];
}) {
  const [url, setUrl] = useState(""),
    [error, setError] = useState(false);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [showOverlay, setShowOverlay] = useState(true),
    [maskUrl, setMaskUrl] = useState("");
  useEffect(() => {
    setMaskUrl("");
    if (!mask) return;
    const [h, w] = mask.size;
    if (
      !Number.isInteger(w) ||
      !Number.isInteger(h) ||
      w < 1 ||
      h < 1 ||
      w > 512 ||
      h > 512 ||
      mask.counts.length > 8192 ||
      mask.counts.some((x) => !Number.isInteger(x) || x < 0) ||
      mask.counts.reduce((a, b) => a + b, 0) !== w * h
    )
      return;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(w, h);
    let pos = 0,
      one = false;
    for (const n of mask.counts) {
      for (let i = 0; i < n; i++, pos++)
        if (one) {
          image.data.set([65, 214, 160, 130], pos * 4);
        }
      one = !one;
    }
    ctx.putImageData(image, 0, 0);
    setMaskUrl(canvas.toDataURL("image/png"));
  }, [mask]);
  useEffect(() => {
    setUrl("");
    setError(false);
    if (!path) return;
    let alive = true,
      objectUrl = "";
    const controller = new AbortController();
    fetch(path.startsWith("/api/") ? path : `/api/v1${path}`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
      signal: controller.signal,
    })
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.blob();
      })
      .then((b) => {
        if (!alive) return;
        objectUrl = URL.createObjectURL(b);
        setUrl(objectUrl);
      })
      .catch(() => {
        if (alive) setError(true);
      });
    return () => {
      alive = false;
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [token, path]);
  return (
    <>
      <div className="evidence-image">
        {url ? (
          <div className="image-overlay-stage">
            <img
              src={url}
              alt="비식별화된 노면 관측 사진"
              onLoad={(e) =>
                setSize({
                  w: e.currentTarget.naturalWidth,
                  h: e.currentTarget.naturalHeight,
                })
              }
            />
            {showOverlay && size.w > 0 && (
              <svg
                className="evidence-overlay"
                viewBox={`0 0 ${size.w} ${size.h}`}
                aria-label="검출 근거 오버레이"
              >
                {maskUrl && (
                  <image
                    href={maskUrl}
                    width={size.w}
                    height={size.h}
                    preserveAspectRatio="none"
                  />
                )}
                {bbox?.length === 4 && (
                  <rect
                    x={bbox[0]}
                    y={bbox[1]}
                    width={bbox[2]}
                    height={bbox[3]}
                    fill="none"
                    stroke="#ffb35d"
                    strokeWidth={Math.max(2, size.w / 300)}
                  />
                )}{" "}
                {outline && outline.length >= 3 && (
                  <polygon
                    points={outline.map((p) => p.join(",")).join(" ")}
                    fill="none"
                    stroke="#f3c460"
                    strokeWidth={Math.max(2, size.w / 300)}
                  />
                )}
              </svg>
            )}
          </div>
        ) : (
          <div>
            <Eye size={28} />
            <p>
              {error
                ? "사진을 불러오지 못했습니다"
                : path
                  ? "사진을 불러오는 중입니다"
                  : "등록된 근거 사진이 없습니다"}
            </p>
          </div>
        )}
      </div>
      {(bbox || mask || outline) && (
        <label className="overlay-toggle">
          <input
            type="checkbox"
            checked={showOverlay}
            onChange={(e) => setShowOverlay(e.target.checked)}
          />
          검출 박스·도막·외곽 표시
        </label>
      )}
    </>
  );
}
