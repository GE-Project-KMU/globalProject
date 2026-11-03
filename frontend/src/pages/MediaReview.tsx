import { useEffect, useRef, useState } from "react";
import {
  Undo2,
  ImagePlus,
  CheckCircle2,
  LoaderCircle,
  ShieldCheck,
} from "lucide-react";
import { request } from "../api";
import type { User } from "../types";
import { ErrorBox } from "../components/UI";

function localDateTime(value: Date) {
  return new Date(value.getTime() - value.getTimezoneOffset() * 60000)
    .toISOString()
    .slice(0, 16);
}

export default function MediaReview({
  token,
  user,
}: {
  token: string;
  user: User;
}) {
  const [file, setFile] = useState<File | null>(null),
    [preview, setPreview] = useState(""),
    [rects, setRects] = useState<
      { x: number; y: number; w: number; h: number }[]
    >([]),
    [drag, setDrag] = useState<{ x: number; y: number } | null>(null),
    [checked, setChecked] = useState(false),
    [device, setDevice] = useState(""),
    [deviceToken, setDeviceToken] = useState(""),
    [capturedAt, setCapturedAt] = useState(""),
    [decoded, setDecoded] = useState(false),
    [dimensions, setDimensions] = useState({ width: 0, height: 0 }),
    [maskInput, setMaskInput] = useState({ x: "0", y: "0", w: "", h: "" }),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [result, setResult] = useState("");
  const canvas = useRef<HTMLCanvasElement>(null),
    image = useRef<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    setRects([]);
    setChecked(false);
    setResult("");
    setCapturedAt(localDateTime(new Date(file.lastModified)));
    setDecoded(false);
    setDimensions({ width: 0, height: 0 });
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    if (!preview) return;
    let active = true;
    const img = new Image();
    img.onload = () => {
      if (!active) return;
      if (!img.width || !img.height || img.width > 4096 || img.height > 4096) {
        setError("가로와 세로가 각각 4,096픽셀 이하인 이미지를 선택해 주세요.");
        setDecoded(false);
        image.current = null;
        return;
      }
      image.current = img;
      setDecoded(true);
      setDimensions({ width: img.width, height: img.height });
      paint();
    };
    img.onerror = () => {
      if (!active) return;
      setError(
        "이미지를 읽지 못했습니다. 유효한 JPEG 또는 PNG 파일을 선택해 주세요.",
      );
      setDecoded(false);
      image.current = null;
      const c = canvas.current;
      if (c) c.getContext("2d")?.clearRect(0, 0, c.width, c.height);
    };
    img.src = preview;
    function paint() {
      const c = canvas.current;
      if (!c) return;
      c.width = img.width;
      c.height = img.height;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      ctx.fillStyle = "#111827";
      rects.forEach((r) => ctx.fillRect(r.x, r.y, r.w, r.h));
    }
    return () => {
      active = false;
    };
  }, [preview, rects]);
  function point(e: React.PointerEvent<HTMLCanvasElement>) {
    const c = e.currentTarget,
      b = c.getBoundingClientRect();
    return {
      x: Math.max(
        0,
        Math.min(c.width, ((e.clientX - b.left) * c.width) / b.width),
      ),
      y: Math.max(
        0,
        Math.min(c.height, ((e.clientY - b.top) * c.height) / b.height),
      ),
    };
  }
  async function submit() {
    if (!file || !canvas.current || !decoded || !checked) return;
    setBusy(true);
    setError("");
    setResult("");
    try {
      const capturedDate = new Date(capturedAt);
      if (
        !Number.isFinite(capturedDate.getTime()) ||
        capturedDate.getTime() > Date.now()
      )
        throw new Error(
          "실제 촬영 시각을 확인해 주세요. 미래 시각은 사용할 수 없습니다.",
        );
      // Build from the current immutable source + rectangles at submission,
      // rather than relying on an asynchronously repainted preview canvas.
      const source = image.current;
      if (!source) throw new Error("검수할 이미지를 다시 선택해 주세요.");
      const output = document.createElement("canvas");
      output.width = source.width;
      output.height = source.height;
      const context = output.getContext("2d")!;
      context.drawImage(source, 0, 0);
      context.fillStyle = "#111827";
      rects.forEach((rect) => context.fillRect(rect.x, rect.y, rect.w, rect.h));
      const blob = await new Promise<Blob>((resolve, reject) =>
        output.toBlob(
          (b) => (b ? resolve(b) : reject(new Error("이미지 변환 실패"))),
          "image/png",
        ),
      );
      if (blob.size > 5000000)
        throw new Error(
          "가림 처리한 파일이 5 MB를 넘습니다. 해상도를 줄인 파일을 선택해 주세요.",
        );
      const hash = Array.from(
        new Uint8Array(
          await crypto.subtle.digest("SHA-256", await blob.arrayBuffer()),
        ),
      )
        .map((x) => x.toString(16).padStart(2, "0"))
        .join("");
      const fid = crypto.randomUUID(),
        captured = capturedDate.toISOString();
      const approval = await request<any>(token, "/image-approvals", {
        method: "POST",
        body: JSON.stringify({
          file_id: fid,
          device_id: device,
          sha256: hash,
          size_bytes: blob.size,
          mime: "image/png",
          captured_at: captured,
          reviewed_at: new Date().toISOString(),
          redaction_model_version: "manual-opaque-v1",
        }),
      });
      const form = new FormData();
      form.set("file", blob, "redacted.png");
      form.set(
        "metadata",
        JSON.stringify({
          file_id: fid,
          approval_id: approval.approval_id || approval.id,
          sha256: hash,
          captured_at: captured,
        }),
      );
      await request(deviceToken, "/images", { method: "POST", body: form });
      setResult(`공유 검수와 전송을 완료했습니다. 파일 ID: ${fid}`);
      setChecked(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="media-layout">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>로컬 사진 검수</h2>
            <p>
              사진은 선택 시 서버로 전송되지 않습니다. 드래그해 가릴 영역을
              추가하세요.
            </p>
          </div>
          <button
            className="icon-button"
            disabled={!rects.length}
            onClick={() => {
              setRects((r) => r.slice(0, -1));
              setChecked(false);
            }}
            aria-label="마지막 가림 취소"
          >
            <Undo2 size={18} />
          </button>
        </div>
        <div className="media-canvas-wrap">
          {preview ? (
            <canvas
              ref={canvas}
              aria-label="검수 이미지, 드래그하여 개인정보 가림"
              onPointerDown={(e) => {
                if (!decoded) return;
                setDrag(point(e));
                e.currentTarget.setPointerCapture(e.pointerId);
              }}
              onPointerUp={(e) => {
                if (!drag) return;
                const end = point(e),
                  r = {
                    x: Math.min(drag.x, end.x),
                    y: Math.min(drag.y, end.y),
                    w: Math.abs(end.x - drag.x),
                    h: Math.abs(end.y - drag.y),
                  };
                if (r.w > 2 && r.h > 2) setRects((x) => [...x, r]);
                setDrag(null);
                setChecked(false);
              }}
              onPointerCancel={() => setDrag(null)}
            />
          ) : (
            <div className="empty">
              <ImagePlus size={32} />
              <h3>검수할 이미지를 선택하세요</h3>
              <p>얼굴·번호판과 불필요한 식별 정보를 가린 뒤 공유합니다.</p>
            </div>
          )}
        </div>
        <div className="panel-padding">
          <label className="file-picker">
            <ImagePlus size={17} />
            사진 선택
            <input
              type="file"
              accept="image/jpeg,image/png"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) {
                  if (
                    f.size > 5000000 ||
                    !["image/jpeg", "image/png"].includes(f.type)
                  ) {
                    setFile(null);
                    setPreview("");
                    setDecoded(false);
                    setChecked(false);
                    setError("5 MB 이하의 JPEG/PNG를 선택해 주세요.");
                    return;
                  }
                  setError("");
                  setDecoded(false);
                  setFile(f);
                }
              }}
            />
          </label>
          <span className="help-text">
            JPEG/PNG · 5 MB 이하 · 가림 영역 {rects.length}개
          </span>
          {decoded && (
            <fieldset className="mask-coordinates">
              <legend>
                좌표로 가림 영역 추가 · {dimensions.width} × {dimensions.height}
                px
              </legend>
              <div className="form-row">
                {(
                  [
                    ["x", "왼쪽 X"],
                    ["y", "위쪽 Y"],
                    ["w", "너비"],
                    ["h", "높이"],
                  ] as const
                ).map(([key, label]) => (
                  <label key={key}>
                    {label}
                    <input
                      type="number"
                      min={key === "w" || key === "h" ? 1 : 0}
                      step="1"
                      value={maskInput[key]}
                      onChange={(e) =>
                        setMaskInput((v) => ({ ...v, [key]: e.target.value }))
                      }
                    />
                  </label>
                ))}
              </div>
              <button
                className="button secondary"
                type="button"
                onClick={() => {
                  const r = {
                    x: Number(maskInput.x),
                    y: Number(maskInput.y),
                    w: Number(maskInput.w),
                    h: Number(maskInput.h),
                  };
                  if (
                    Object.values(r).some((v) => !Number.isFinite(v)) ||
                    r.x < 0 ||
                    r.y < 0 ||
                    r.w <= 0 ||
                    r.h <= 0 ||
                    r.x + r.w > dimensions.width ||
                    r.y + r.h > dimensions.height
                  ) {
                    setError(
                      "가림 영역의 좌표와 크기를 이미지 범위 안에서 입력해 주세요.",
                    );
                    return;
                  }
                  setRects((v) => [...v, r]);
                  setChecked(false);
                  setError("");
                }}
              >
                가림 영역 추가
              </button>
            </fieldset>
          )}
        </div>
      </section>
      <section className="panel media-form">
        <span className="eyebrow">SHARING APPROVAL</span>
        <h2>확인 후 공유</h2>
        <p className="muted">
          최종 이미지의 해시와 검수자를 기록하고 기기 권한으로 전송합니다.
        </p>
        {error && <ErrorBox message={error} />}{" "}
        {result && (
          <div className="success-box" role="status">
            <CheckCircle2 size={18} />
            {result}
          </div>
        )}
        <label>
          실제 촬영 시각
          <input
            type="datetime-local"
            value={capturedAt}
            onChange={(e) => {
              setCapturedAt(e.target.value);
              setChecked(false);
            }}
            required
          />
        </label>
        <p className="help-text">
          초기값은 파일 수정 시각입니다. 실제 촬영 시각과 다르면 수정해 주세요.
          서버에는 UTC로 변환해 전송합니다.
        </p>
        <label>
          연결할 기기 ID
          <input
            value={device}
            onChange={(e) => setDevice(e.target.value)}
            placeholder="등록된 기기 UUID"
          />
        </label>
        <label>
          기기 전송 토큰
          <input
            type="password"
            value={deviceToken}
            onChange={(e) => setDeviceToken(e.target.value)}
            autoComplete="off"
            placeholder="전송 시에만 사용 · 저장하지 않음"
          />
        </label>
        <label className="check-label">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => setChecked(e.target.checked)}
          />
          얼굴·번호판·개인 식별 요소를 확인했고 이 최종 파일의 공유를
          승인합니다.
        </label>
        <button
          className="button primary full"
          disabled={
            !file ||
            !decoded ||
            !capturedAt ||
            !checked ||
            !device ||
            !deviceToken ||
            busy ||
            user.role === "viewer"
          }
          onClick={submit}
        >
          {busy ? (
            <LoaderCircle size={16} className="spin" />
          ) : (
            <ShieldCheck size={16} />
          )}
          승인 및 전송
        </button>
        <p className="help-text">
          직접 가린 이미지는 수동 검수 자료입니다. AI 비식별화 모델의 검출
          성능을 나타내지 않습니다.
        </p>
      </section>
    </div>
  );
}
