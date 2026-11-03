import { useState, type FormEvent } from "react";
import { request } from "../api";
import type { User } from "../types";
import {
  Collection,
  Feedback,
  JsonDetails,
  useCollection,
} from "../components/Management";

const fields: Record<string, string[]> = {
  profiles: [
    "name",
    "version",
    "camera_serial",
    "lens_label",
    "calibration",
    "config_hash",
  ],
  "model-bundles": [
    "version",
    "weights_hashes",
    "preprocess",
    "thresholds",
    "config_hash",
  ],
};
export default function Registry({
  token,
  user,
  kind,
}: {
  token: string;
  user: User;
  kind: "profiles" | "model-bundles";
}) {
  const state = useCollection(token, `/${kind}`),
    [source, setSource] = useState(""),
    [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false);
  const profile = kind === "profiles";
  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setMessage("");
    setBusy(true);
    try {
      const b = JSON.parse(source);
      if (
        !b ||
        Array.isArray(b) ||
        typeof b !== "object" ||
        fields[kind].some((k) => !(k in b))
      )
        throw new Error(`필수 항목: ${fields[kind].join(", ")}`);
      await request(token, `/${kind}`, {
        method: "POST",
        body: JSON.stringify(b),
      });
      setSource("");
      setMessage(
        profile
          ? "불변 장착 프로파일을 등록했습니다."
          : "모델 manifest를 등록했습니다. 가중치 준비·검증과 정확도 입증은 별도입니다.",
      );
      await state.load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Collection
        title={profile ? "장착·보정 프로파일" : "모델 버전 레지스트리"}
        description={
          profile
            ? "차종별 실제 보정값과 ZED 2i 카메라 정보를 버전으로 고정합니다."
            : "가중치 해시·전처리·임계값·코드 버전을 연결합니다. 등록만으로 배포 준비를 뜻하지 않습니다."
        }
        state={state}
      >
        <div className="management-list">
          {state.items.map((x) => (
            <article key={x.id || x.profile_id || x.model_bundle_id}>
              <div>
                <h3>
                  {profile ? x.name : x.version}
                  {profile ? ` · v${x.version}` : ""}
                </h3>
                <p>
                  {profile
                    ? `${x.camera_serial} · ${x.lens_label}`
                    : "manifest 등록 · 실제 가중치 검증 상태는 분석 운영에서 확인"}
                </p>
                <code>{x.id || x.profile_id || x.model_bundle_id}</code>
                <JsonDetails value={x} />
              </div>
            </article>
          ))}
        </div>
      </Collection>
      {user.role === "admin" && (
        <section className="panel management-panel">
          <h2>{profile ? "보정 프로파일 등록" : "모델 manifest 등록"}</h2>
          <p>
            {profile
              ? "실측 보정 산출 JSON을 입력하거나 가져오세요. 예시 보정값을 실제 장착값으로 등록하지 마세요. 같은 이름·버전은 변경할 수 없습니다."
              : "로컬 학습·내보내기 도구의 manifest를 가져오세요. 원본 영상이나 가중치 바이너리는 서버로 보내지 않습니다. 버전은 불변입니다."}
          </p>
          <details className="json-details">
            <summary>필수 필드와 작성 기준</summary>
            <p>{fields[kind].join(" · ")}</p>
            {profile ? (
              <p>
                calibration: image_size [너비,높이], height_m, pitch_deg,
                camera_matrix (3×3), distortion, homography (3×3),
                roi_polygon_px, visible_distance_m [근거리,원거리],
                calibrated_at (UTC). config_hash는 실제 설정의 SHA-256입니다.
              </p>
            ) : (
              <p>
                weights_hashes: pothole/lane/redaction SHA-256. preprocess: 실제
                전처리 설정. thresholds: pothole_confidence, pothole_nms_iou,
                single_frame_confidence, lane_monitor, lane_good,
                min_valid_ratio, min_segment_support_ratio.
                code_commit·environment_hash도 함께 기록하세요.
              </p>
            )}
          </details>
          <form className="management-form" onSubmit={submit}>
            <label>
              JSON 파일 가져오기
              <input
                type="file"
                accept=".json,application/json"
                onChange={async (e) => {
                  const f = e.target.files?.[0];
                  if (!f) return;
                  if (f.size > 262144) {
                    setError("설정 JSON은 256KB 이하로 가져오세요.");
                    return;
                  }
                  setSource(await f.text());
                  setError("");
                }}
              />
            </label>
            <label>
              {profile ? "프로파일 JSON" : "모델 manifest JSON"}
              <textarea
                required
                rows={14}
                value={source}
                onChange={(e) => setSource(e.target.value)}
                spellCheck={false}
              />
            </label>
            <Feedback error={error} message={message} />
            <button
              className="button primary"
              disabled={busy || !source.trim()}
            >
              {busy ? "등록 중…" : "새 버전 등록"}
            </button>
          </form>
        </section>
      )}
    </>
  );
}
