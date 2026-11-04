"""Batch analysis. Real detectors are mandatory; geometry uncertainty is explicit."""
from datetime import datetime, timezone, timedelta
import json
import uuid
import time
import cv2
import numpy as np
from .capture import replay
from .models import OnnxModel, ModelNotReady, frame_quality, redact,runtime_metadata
from .lane import measure_patch
from .inference_runner import create_models,InferenceRunner,close_models
from .paths import workspace_path, atomic_json, sha256


def infer_potholes(source, manifest_path, output):
    """Single-image real inference, with explicit external-baseline provenance."""
    source = workspace_path(source, exists=True)
    model = OnnxModel(manifest_path)
    if model.manifest["task"] != "pothole":
        raise ValueError("pothole_model_required")
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError("image_decode_failed")
    started = time.perf_counter()
    detections = model.predict(image)
    result = {"status": "inference_completed", "source_sha256": sha256(source),
              "source_size": [image.shape[1], image.shape[0]], "detections": detections,
              "model_architecture": model.manifest.get("architecture", "YOLOX"),
              "model_sha256": model.manifest["sha256"], "dataset_release": model.manifest["dataset_release"],
              "production_approved": model.manifest.get("production_approved", False),
              "runtime": model.backend,"runtime_version":model.backend_version,"runtime_details":runtime_metadata({"pothole":model}), "elapsed_seconds": time.perf_counter() - started,
              "inferred_at": datetime.now(timezone.utc).isoformat(),
              "sharing_state": "local_result_only_no_image_approval"}
    atomic_json(output, result)
    return result


def intersection_over_union(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0


class Tracker:
    def __init__(self, iou_threshold=.3):
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.last_at = None

    def update(self, detections, at):
        if not np.isfinite(at) or self.last_at is not None and at<=self.last_at:raise ValueError("tracking_time_must_increase")
        self.last_at=at
        self.tracks=[track for track in self.tracks if at-track["last_at"]<=1]
        available = {i for i, track in enumerate(self.tracks) if at - track["last_at"] <= 1}
        updated = []
        for detection in sorted(detections, key=lambda x: x["confidence"], reverse=True):
            matches=[]
            for i in available:
                track=self.tracks[i]
                a,b=track["bbox_xywh_px"],detection["bbox_xywh_px"]
                movement=np.array([b[0]+b[2]/2-a[0]-a[2]/2,b[1]+b[3]/2-a[1]-a[3]/2])
                velocity=np.asarray(track.get("direction",[0.,0.]))
                if np.linalg.norm(movement)>2 and np.linalg.norm(velocity)>2 and np.dot(movement,velocity)<-.5*np.linalg.norm(movement)*np.linalg.norm(velocity):continue
                matches.append((intersection_over_union(a,b),i))
            best = max(matches, default=(0, None))
            if best[0] >= self.iou_threshold:
                track = self.tracks[best[1]]
                available.remove(best[1])
                a,b=track["bbox_xywh_px"],detection["bbox_xywh_px"]
                track["direction"]=[b[0]+b[2]/2-a[0]-a[2]/2,b[1]+b[3]/2-a[1]-a[3]/2]
            else:
                track = {"event_id": str(uuid.uuid4()), "observations_at": []}
                self.tracks.append(track)
            track.update(detection, last_at=at)
            track["observations_at"] = [t for t in track["observations_at"] if at - t <= 1] + [at]
            count = len(track["observations_at"])
            state = "multi_frame" if count >= 3 else "single_frame_review" if detection["confidence"] >= .8 else "held"
            updated.append(dict(track, track_observations=count, candidate_state=state))
        return updated


def straight_solid_outline(paint, nominal_width_px):
    """Conservative BEV straight-solid geometry, never guesses dashed gaps.

    Returns unsupported on sparse/curved/ambiguous paint. A calibrated nominal
    width is mandatory and must not be inferred from the worn paint itself.
    """
    nominal_width_px = float(nominal_width_px)
    if not np.isfinite(nominal_width_px) or nominal_width_px < 2:
        raise ValueError("calibrated_lane_width_required")
    y, x = np.nonzero(paint)
    h, w = paint.shape
    if len(x) < 50 or np.ptp(y) < h * .35:
        return None, "insufficient_visible_boundary"
    centers = []
    for row in range(h):
        points = x[y == row]
        if len(points) >= 2:
            centers.append((row, float(np.median(points))))
    if len(centers) < h * .35:
        return None, "insufficient_rows"
    ys, xs = np.array(centers).T
    slope, intercept = np.polyfit(ys, xs, 1)
    error = float(np.sqrt(np.mean((xs - (slope * ys + intercept)) ** 2)))
    if error > nominal_width_px * .35 or abs(slope) > .8:
        return None, "curved_or_multiple_lane_geometry"
    outline = np.zeros((h, w), np.uint8)
    for row in range(h):
        center = slope * row + intercept
        left, right = int(round(center - nominal_width_px / 2)), int(round(center + nominal_width_px / 2))
        if left < 0 or right > w:
            return None, "outline_clipped"
        outline[row, left:right] = 1
    # Paint far outside one inferred stripe means multiple lanes/incorrect geometry.
    inside = int((paint.astype(bool) & outline.astype(bool)).sum())
    if inside / len(x) < .8:
        return None, "multiple_lane_geometry"
    return outline, None


def analyze_video(source, output, *, pothole_manifest, lane_manifest, redaction_manifest,
                  stereo=False, profile_path=None, sample_fps=10, cpu_profile="default", inference_backend="cpu",gpu_profile=None):
    # All real models must load before any processing; no heuristic potholes.
    models = create_models({"pothole":pothole_manifest,"lane":lane_manifest,"redaction":redaction_manifest},cpu_profile,OnnxModel,inference_backend=inference_backend,gpu_profile=gpu_profile)
    try:
        output = workspace_path(output)
        output.mkdir(parents=True, exist_ok=True)
        profile = json.loads(workspace_path(profile_path, exists=True).read_text(encoding="utf-8")) if profile_path else {}
        h_matrix = profile.get("homography")
        if h_matrix is not None:
            h_matrix = np.asarray(h_matrix, dtype=np.float64)
            if h_matrix.shape != (3, 3) or not np.isfinite(h_matrix).all() or abs(np.linalg.det(h_matrix)) < 1e-12:
                raise ValueError("invalid_homography")
        tracker, observations, started = Tracker(), [], datetime.now(timezone.utc)
        run_id = str(uuid.uuid4())
        with InferenceRunner(models,cpu_profile) as runner:
            for sample in replay(source, sample_fps=sample_fps):
                frame = sample["frame"]
                if stereo:
                    if frame.shape[1] % 2:
                        raise ValueError("side_by_side_even_width_required")
                    frame = frame[:, :frame.shape[1] // 2]
                quality = frame_quality(frame)
                predictions = runner.predict(frame)
                potholes = tracker.update(predictions["pothole"], sample["pts_seconds"])
                paint = predictions["lane"]
                redacted = redact(frame, predictions["redaction"])
                image_path = output / f"{sample['frame_id']:09d}.redacted.png"
                if not cv2.imwrite(str(image_path), redacted):
                    raise RuntimeError("image_write_failed")
                lane = {"grade": "unassessable", "retention_pct": None, "valid_area_ratio": None,
                        "segment_support_ratio": None, "outline_supported": False,
                        "unassessable_reason": "missing_calibrated_geometry"}
                if h_matrix is not None and profile.get("line_type") == "solid" and profile.get("nominal_lane_width_px"):
                    size = profile.get("bev_size", [512, 512])
                    bev = cv2.warpPerspective(paint, h_matrix, tuple(size), flags=cv2.INTER_NEAREST)
                    outline, reason = straight_solid_outline(bev, profile["nominal_lane_width_px"])
                    if outline is not None:
                        result = measure_patch(outline, np.zeros_like(outline), bev, np.ones_like(outline))
                        lane.update(result)
                        # A single frame is a patch, not a supported 10m segment.
                        lane["patch_grade"] = result["grade"]
                        lane["patch_retention_pct"] = result["retention_pct"]
                        lane.update(grade="unassessable", retention_pct=None, segment_support_ratio=None,
                                    unassessable_reason="segment_distance_support_not_available")
                    else:
                        lane["unassessable_reason"] = reason
                elif profile.get("line_type") == "dashed":
                    lane["unassessable_reason"] = "intentional_gap_geometry_required"
                item = {"run_id": run_id, "frame_id": sample["frame_id"], "pts_seconds": sample["pts_seconds"],
                        "quality": quality, "potholes": potholes, "lane": lane,
                        "image": {"path": image_path.relative_to(workspace_path(".")).as_posix(), "sha256": sha256(image_path),
                                  "redacted": True, "reviewed_for_sharing": False},
                        "expires_at": (started + timedelta(days=90)).isoformat(),
                        "status": "awaiting_human_review"}
                observations.append(item)
        result = {"run_id": run_id, "started_at": started.isoformat(), "source": str(workspace_path(source)),
                  "analyzed_frames": len(observations), "observations": observations,
                  "runtime":runtime_metadata(models)["engine"],"runtime_details":runtime_metadata(models), "cpu_profile":cpu_profile, "limitations": ["No validated target accuracy is claimed.",
                  "Single-frame lane patch values are not 10m segment judgments.",
                  "Outputs need human image review and complete acquisition metadata before API ingestion."]}
        atomic_json(output / "analysis.json", result)
        return result
    finally:close_models(models)
