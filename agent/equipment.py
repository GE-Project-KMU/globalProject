"""Optional stereo/SDK adapters and stationary preflight, with explicit capability errors."""
from datetime import datetime, timezone
import shutil
import uuid
import time
import cv2
import numpy as np
from .paths import workspace_path, atomic_json
from .paths import sha256
from .profile import load_profile, geometry
from .gnss import build_fixes

CHECKS = {"camera_preview_ok", "mount_secure", "lens_clear", "profile_valid", "gnss_device_connected", "storage_ge_10gb", "notice_present"}


def collection_plan(path):
    import json
    path=workspace_path(path,exists=True)
    value=json.loads(path.read_text(encoding="utf-8"))
    required={"plan_id","organization","controller_contact","operator","legal_basis","purpose","collection_scope","notice_message",
              "raw_retention_days","masked_retention_days","audit_retention_days"}
    if set(value)!=required:raise ValueError("collection_plan_fields_required")
    str(uuid.UUID(value["plan_id"]))
    for key in required-{"plan_id","raw_retention_days","masked_retention_days","audit_retention_days"}:
        if not isinstance(value[key],str) or not 3<=len(value[key].strip())<=2000 or "REPLACE" in value[key].upper():
            raise ValueError("collection_plan_value_required:"+key)
    if any(type(value[k]) is not int or value[k]!=days for k,days in
           (("raw_retention_days",7),("masked_retention_days",90),("audit_retention_days",180))):
        raise ValueError("collection_plan_retention_mismatch")
    return {"path":path.relative_to(workspace_path(".")).as_posix(),"sha256":sha256(path),**value}


def preflight(source, profile_path, checklist, output, *, nmea_records):
    """Human mount/lens/notice decisions are never inferred from camera pixels."""
    profile = load_profile(profile_path)
    human = {"mount_secure", "lens_clear", "notice_present", "daylight", "dry_asphalt"}
    if set(checklist) != human or any(checklist[k] is not True for k in human): raise ValueError("human_preflight_checks_required")
    source = source if isinstance(source, int) else str(workspace_path(source, exists=True))
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened(): raise RuntimeError("camera_unavailable")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,2560);capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720);capture.set(cv2.CAP_PROP_FPS,30)
        ok, frame = capture.read()
        if not ok or frame.shape[1] % 2: raise RuntimeError("stereo_preview_unavailable")
        geometry(frame[:, :frame.shape[1] // 2], profile)
    finally: capture.release()
    fixes = build_fixes(nmea_records)["fixes"]
    if not fixes or min(abs((datetime.now(timezone.utc)-datetime.fromisoformat(f["utc"])).total_seconds()) for f in fixes)>2:
        raise ValueError("fresh_gnss_valid_nmea_preview_required")
    if shutil.disk_usage(workspace_path(".")).free < 10_000_000_000: raise RuntimeError("insufficient_disk_space")
    result = {"precheck": {k: True for k in CHECKS}, "profile_id": profile["profile_id"],
              "conditions": {"daylight":True,"dry_asphalt":True},
              "checked_at": datetime.now(timezone.utc).isoformat(), "preview_size": list(frame.shape[1::-1])}
    atomic_json(output, result)
    return result


def probe_gnss(port, seconds=3,baudrate=115200):
    import serial
    if type(baudrate) is not int or baudrate<=0 or not 0<seconds<=30:raise ValueError("invalid_gnss_probe_options")
    rows = []
    with serial.Serial(port,baudrate=baudrate,timeout=.2) as stream:
        deadline = time.monotonic()+seconds
        while time.monotonic()<deadline:
            line = stream.readline(4096)
            if line:
                rows.append({"sentence":line.decode("ascii",errors="replace"),"received_at":datetime.now(timezone.utc).isoformat()})
    return rows


def stereo_depth(left, right, calibration):
    """Rectified stereo disparity, metric Z and point cloud; invalid pixels remain NaN."""
    if left.shape != right.shape or left.ndim != 3: raise ValueError("stereo_shape_mismatch")
    h, w = left.shape[:2]
    matrices = [np.asarray(calibration[k], float) for k in ("left_camera_matrix", "right_camera_matrix", "rotation")]
    if any(m.shape != (3, 3) or not np.isfinite(m).all() for m in matrices): raise ValueError("invalid_stereo_calibration")
    translation = np.asarray(calibration["translation_m"], float)
    if translation.shape != (3,) or not np.isfinite(translation).all() or np.linalg.norm(translation) <= 0: raise ValueError("stereo_baseline_required")
    k1, k2, rotation = matrices
    d1, d2 = np.asarray(calibration["left_distortion"], float), np.asarray(calibration["right_distortion"], float)
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(k1, d1.reshape(-1,1), k2, d2.reshape(-1,1), (w,h), rotation, translation.reshape(3,1))
    rectified = []
    for image, k, d, r, p in ((left,k1,d1,r1,p1),(right,k2,d2,r2,p2)):
        mx, my = cv2.initUndistortRectifyMap(k,d,r,p,(w,h),cv2.CV_32FC1)
        rectified.append(cv2.cvtColor(cv2.remap(image,mx,my,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY))
    disparities = min(128, max(16, (w // 3 // 16) * 16))
    matcher = cv2.StereoSGBM_create(minDisparity=0,numDisparities=disparities,blockSize=5,P1=8*25,P2=32*25,
                                  disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=50,speckleRange=2)
    disparity = matcher.compute(*rectified).astype(np.float32) / 16
    points = cv2.reprojectImageTo3D(disparity,q)
    bad = (disparity <= 0) | ~np.isfinite(points).all(axis=2) | (points[:,:,2] <= 0)
    points[bad] = np.nan
    return {"disparity_px": disparity, "points_m": points, "depth_m": points[:,:,2], "valid": ~bad}


def depth_summary(points_m, bbox):
    x,y,w,h = map(int,bbox)
    roi = points_m[max(0,y):max(0,y+h),max(0,x):max(0,x+w)]
    valid = np.isfinite(roi).all(axis=2)
    if not valid.size or valid.sum() < 20 or valid.mean() < .7:
        return {"status": "unavailable", "reason": "insufficient_stereo_support", "camera_point_m": None}
    return {"status": "available_unvalidated", "camera_point_m": np.median(roi[valid],axis=0).tolist(),
            "valid_ratio": float(valid.mean()), "physical_severity": None}


def road_relative_step(points_m,bbox):
    """Experimental surrounding-plane residual, never a physical severity grade."""
    x,y,w,h=map(int,bbox);height,width=points_m.shape[:2]
    x0,y0,x1,y1=max(0,x-w//2),max(0,y-h//2),min(width,x+w+w//2),min(height,y+h+h//2)
    patch=points_m[y0:y1,x0:x1]
    ring=np.ones(patch.shape[:2],bool)
    ring[max(0,y-y0):min(y+h-y0,ring.shape[0]),max(0,x-x0):min(x+w-x0,ring.shape[1])]=False
    support=patch[ring&np.isfinite(patch).all(axis=2)]
    inside=points_m[max(0,y):min(height,y+h),max(0,x):min(width,x+w)]
    target=inside[np.isfinite(inside).all(axis=2)]
    if len(support)<100 or len(target)<20:return {"status":"unavailable","reason":"insufficient_surrounding_depth","offset_m":None}
    center=np.median(support,axis=0)
    _,singular,vectors=np.linalg.svd(support-center,full_matrices=False)
    if singular[1]<1e-6:return {"status":"unavailable","reason":"degenerate_road_plane","offset_m":None}
    normal=vectors[-1]
    if normal[2]<0:normal=-normal
    residual=(support-center)@normal
    scatter=float(np.median(abs(residual)))
    if scatter>.03:return {"status":"unavailable","reason":"road_plane_not_supported","offset_m":None}
    return {"status":"experimental_unvalidated","offset_m":float(np.median((target-center)@normal)),
            "plane_median_residual_m":scatter,"physical_severity":None,"reference":"surrounding_stereo_plane"}


def object_coordinate(location, camera_point_m, camera_to_vehicle, heading_deg):
    """Optional object estimate; never replaces the API vehicle-observation location."""
    if location["quality"] != "valid" or heading_deg is None: return {"status": "unavailable", "reason": "position_or_heading_missing"}
    transform = np.asarray(camera_to_vehicle,float)
    point = np.asarray(camera_point_m,float)
    if transform.shape != (4,4) or point.shape != (3,) or not np.isfinite(transform).all() or not np.isfinite(point).all() or not np.isfinite(heading_deg):
        raise ValueError("invalid_object_transform")
    lateral, forward, _, _ = transform @ np.r_[point,1.]
    angle = np.deg2rad(heading_deg)
    east, north = lateral*np.cos(angle)+forward*np.sin(angle), forward*np.cos(angle)-lateral*np.sin(angle)
    lat = float(location["lat"])
    if abs(lat) >= 89: return {"status": "unavailable", "reason": "local_projection_polar_limit"}
    return {"status": "estimated_unvalidated", "lat": lat + float(north)/111320,
            "lon": float(location["lon"]) + float(east)/(111320*np.cos(np.deg2rad(lat))),
            "source": "stereo_heading_offset", "accuracy_m": None}


class ZedSDK:
    """Optional SDK adapter; importing this module never requires ZED SDK."""
    def __init__(self):
        try: import pyzed.sl as sl
        except ImportError as exc: raise RuntimeError("zed_sdk_not_installed_use_uvc_capture") from exc
        self.sl, self.camera = sl, sl.Camera()
        settings = sl.InitParameters(); settings.coordinate_units = sl.UNIT.METER
        # Classical mode avoids implicitly downloading neural SDK weights/caches.
        settings.depth_mode = sl.DEPTH_MODE.ULTRA
        error = self.camera.open(settings)
        if error != sl.ERROR_CODE.SUCCESS:
            self.camera.close(); raise RuntimeError("zed_sdk_open_failed:" + str(error))
    def read(self):
        sl = self.sl
        if self.camera.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS: return None
        left,right,depth = sl.Mat(),sl.Mat(),sl.Mat()
        self.camera.retrieve_image(left,sl.VIEW.LEFT); self.camera.retrieve_image(right,sl.VIEW.RIGHT)
        self.camera.retrieve_measure(depth,sl.MEASURE.DEPTH)
        return {"left": left.get_data()[:,:,:3].copy(), "right": right.get_data()[:,:,:3].copy(),
                "depth_m": depth.get_data().copy(), "timestamp_ns": self.camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()}
    def close(self): self.camera.close()
