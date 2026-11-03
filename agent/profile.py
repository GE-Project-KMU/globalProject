"""Validated local installation geometry; measurements require operator calibration."""
from datetime import datetime
import json
import uuid
import cv2
import numpy as np
from .paths import workspace_path, atomic_json


def load_profile(path):
    value = json.loads(workspace_path(path, exists=True).read_text(encoding="utf-8"))
    str(uuid.UUID(value["profile_id"]))
    c = value["calibration"]
    w, h = c["image_size"]
    if type(w) is not int or type(h) is not int or not 0 < min(w, h) <= max(w, h) <= 16384:
        raise ValueError("invalid_image_size")
    for key in ("camera_matrix", "homography"):
        matrix = np.asarray(c[key], float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("invalid_" + key)
    distortion = np.asarray(c["distortion"], float)
    if distortion.ndim != 1 or len(distortion) not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError("invalid_distortion")
    roi = np.asarray(c["roi_polygon_px"], float)
    if roi.ndim != 2 or roi.shape[1] != 2 or len(roi) < 3 or not np.isfinite(roi).all() or (roi < 0).any() or (roi[:, 0] > w).any() or (roi[:, 1] > h).any() or abs(cv2.contourArea(roi.astype(np.float32))) < 1:
        raise ValueError("invalid_roi")
    near, far = c["visible_distance_m"]
    if not 0 <= near < far or not np.isfinite([near, far, c["height_m"], c["pitch_deg"]]).all() or c["height_m"] <= 0:
        raise ValueError("invalid_mount_geometry")
    if datetime.fromisoformat(c["calibrated_at"].replace("Z", "+00:00")).tzinfo is None or not value.get("calibrated_by"):
        raise ValueError("calibration_review_required")
    metric = value["measurement"]
    bw, bh = metric["bev_size"]
    scale = float(metric["meters_per_pixel"])
    if type(bw) is not int or type(bh) is not int or not 16 <= min(bw, bh) <= max(bw, bh) <= 4096 or not np.isfinite(scale) or not 0 < scale <= .1:
        raise ValueError("invalid_bev_geometry")
    if abs(bh * scale - (far - near)) > scale * 2:
        raise ValueError("bev_distance_scale_mismatch")
    tracks = metric.get("lane_tracks", [])
    seen = set()
    for track in tracks:
        identifier = str(uuid.UUID(track["lane_track_id"]))
        if identifier in seen: raise ValueError("duplicate_lane_track")
        seen.add(identifier)
        if track["line_type"] not in ("solid", "dashed") or track["color"] not in ("white", "yellow"):
            raise ValueError("unsupported_lane_type")
        left, right = track["bev_x_range"]
        if type(left) is not int or type(right) is not int or not 0 <= left < right <= bw or track.get("outline_verified") is not True:
            raise ValueError("verified_lane_outline_required")
        if track["line_type"] == "dashed":
            dash, gap, phase = [float(track[k]) for k in ("dash_length_m", "gap_length_m", "phase_m")]
            if not np.isfinite([dash, gap, phase]).all() or min(dash, gap) <= 0 or track.get("phase_verified") is not True:
                raise ValueError("verified_dash_phase_required")
    return value


def geometry(frame, profile):
    c, metric = profile["calibration"], profile["measurement"]
    if list(frame.shape[1::-1]) != c["image_size"]: raise ValueError("profile_image_size_mismatch")
    image = cv2.undistort(frame, np.asarray(c["camera_matrix"], float), np.asarray(c["distortion"], float))
    roi = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(roi, [np.asarray(c["roi_polygon_px"], np.int32)], 1)
    return image, roi, np.asarray(c["homography"], float), tuple(metric["bev_size"])


def calibrate_homography(request, output):
    value = json.loads(workspace_path(request, exists=True).read_text(encoding="utf-8"))
    source, target = np.asarray(value["image_points"], np.float32), np.asarray(value["bev_points"], np.float32)
    if source.shape != (4, 2) or target.shape != (4, 2) or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("four_calibration_correspondences_required")
    matrix = cv2.getPerspectiveTransform(source, target)
    if abs(np.linalg.det(matrix)) < 1e-12: raise ValueError("degenerate_calibration_points")
    result = {"homography": matrix.tolist(), "operator_measurements_required": True,
              "image_points": source.tolist(), "bev_points": target.tolist()}
    atomic_json(output, result)
    return result


def lane_evidence(bev_image, paint, valid, profile, distance=None):
    """Check visible color/alignment against each independently verified outline."""
    hsv=cv2.cvtColor(bev_image,cv2.COLOR_BGR2HSV)
    answer={}; usable=valid.copy()
    height=paint.shape[0]
    for track in profile["measurement"]["lane_tracks"]:
        left,right=track["bev_x_range"]
        selected=(paint[:,left:right]>0)&(valid[:,left:right]>0)
        pixels=hsv[:,left:right][selected]
        reason=None;color="unknown"
        if len(pixels)>=20:
            yellow=float(((pixels[:,0]>=15)&(pixels[:,0]<=40)&(pixels[:,1]>=70)&(pixels[:,2]>=70)).mean())
            white=float(((pixels[:,1]<90)&(pixels[:,2]>=80)).mean())
            color="yellow" if yellow>=.6 else "white" if white>=.6 else "unknown"
        if color=="unknown": reason="insufficient_visible_color_or_boundary"
        elif color!=track["color"]:reason="lane_color_profile_mismatch"
        # Visible paint centers must agree with the independently calibrated stripe.
        y,x=np.nonzero(selected)
        if len(x)>=20 and np.ptp(y)>=height*.2:
            centers=[float(np.median(x[y==row])) for row in np.unique(y)]
            if abs(float(np.median(centers))-(right-left-1)/2)>(right-left)*.3:
                reason="lane_outline_alignment_mismatch"
        if reason:usable[:,left:right]=0
        answer[track["lane_track_id"]]={"color":color,"line_type":track["line_type"],"outline_supported":reason is None,"reason":reason}
    return usable,answer
