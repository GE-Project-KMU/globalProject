"""Acquisition lineage -> real inference -> review package -> encrypted outbox."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import uuid
import re
import time
from urllib.parse import urlparse
import httpx
import cv2
import numpy as np
from .capture import EncryptedStore, replay, replay_bytes
from .gnss import build_fixes, associate_location
from .models import OnnxModel, frame_quality, redact,runtime_metadata
from .inference_runner import create_models,InferenceRunner,profile_options,close_models
from .pipeline import Tracker
from .profile import load_profile, geometry, lane_evidence
from .segments import Distance, Segments
from .paths import workspace_path, atomic_json, sha256
from .outbox import Outbox
from .tls import gateway_tls


def read_json(path):
    return json.loads(workspace_path(path, exists=True).read_text(encoding="utf-8"))


def overlay_mask(mask):
    h,w = mask.shape
    scale = min(1.,512/max(h,w))
    small = (cv2.resize(mask,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_NEAREST)>0).astype(np.uint8)
    values = small.ravel(order="C")
    changes = np.flatnonzero(values[1:] != values[:-1])+1
    counts = np.diff(np.r_[0,changes,len(values)]).tolist()
    if values[0]: counts.insert(0,0)
    return {"size": list(small.shape), "counts": counts} if len(counts)<=8192 else None


def run_samples(manifest_path, *, key_path="data/agent/local.key", sample_fps=10):
    """Authenticate/decrypt bounded chunks in memory; no plaintext temp files."""
    manifest = read_json(manifest_path)
    if datetime.fromisoformat(manifest["expires_at"]) <= datetime.now(timezone.utc):
        raise ValueError("raw_run_expired")
    store = EncryptedStore(key_path)
    indexes, records = [], []
    for chunk in manifest["chunks"]:
        for asset in [chunk["index"], *chunk["eyes"]]:
            if workspace_path(asset["path"],exists=True).stat().st_size>128*1024*1024+64:
                raise ValueError("encrypted_chunk_exceeds_memory_limit")
            if sha256(workspace_path(asset["path"], exists=True)) != asset["sha256"]:
                raise ValueError("capture_chunk_digest_mismatch")
        index=json.loads(store.decrypt_bytes(chunk["index"]["path"]))
        indexes.append(index);records.extend(index["nmea"])
    fixes=build_fixes(records)
    for chunk,index in zip(manifest["chunks"],indexes):
        left=next(asset for asset in chunk["eyes"] if ".left.mp4.aesgcm" in asset["path"])
        for sample in replay_bytes(store.decrypt_bytes(left["path"]),sample_fps=sample_fps):
            metadata=index["frames"][sample["frame_id"]]
            yield sample|metadata|{"fixes":fixes["fixes"],"gnss_errors":fixes["errors"]}


def video_samples(source, metadata_path, *, sample_fps=10, stereo=False):
    metadata = read_json(metadata_path)
    by_frame = {row["frame_id"]: row for row in metadata["frames"]}
    fixes = build_fixes(metadata.get("nmea", []))
    for sample in replay(source, sample_fps=sample_fps):
        if sample["frame_id"] not in by_frame: raise ValueError("original_frame_timestamp_required")
        if stereo:
            w = sample["frame"].shape[1]
            if w % 2: raise ValueError("side_by_side_even_width_required")
            sample["frame"] = sample["frame"][:, :w // 2]
        yield sample | by_frame[sample["frame_id"]] | {"fixes": fixes["fixes"], "gnss_errors": fixes["errors"]}


def analyze_samples(samples, output, *, run, profile_path, device_id, bundle, model_factory=OnnxModel, cpu_profile="default",inference_backend="cpu",gpu_profile=None):
    """The injectable factory is for contract tests; CLI always uses real ONNX."""
    started=time.perf_counter()
    profile = load_profile(profile_path)
    run_id, device_id = str(uuid.UUID(run["run_id"])), str(uuid.UUID(device_id))
    if run.get("source_kind") == "video_replay" and not run.get("original_capture_verified"):
        raise ValueError("video_replay_requires_verified_original_acquisition")
    if not isinstance(bundle.get("version"), str) or not bundle["version"]:
        raise ValueError("model_bundle_version_required")
    models = create_models({task:bundle[task] for task in ("pothole","lane","redaction")},cpu_profile,model_factory,inference_backend=inference_backend,gpu_profile=gpu_profile)
    try:
        for task, model in models.items():
            if model.manifest["task"] != task: raise ValueError("bundle_task_mismatch")
            if model.manifest["sha256"] != bundle["weights_hashes"][task]: raise ValueError("bundle_weight_hash_mismatch")
        output = workspace_path(output); output.mkdir(parents=True, exist_ok=True)
        config_hash = bundle.get("config_hash", "")
        if not re.fullmatch("[a-f0-9]{64}", config_hash): raise ValueError("registered_bundle_config_hash_required")
        tracker, segments = Tracker(), Segments(run_id, profile)
        motion = Distance(profile["measurement"]["meters_per_pixel"])
        frame_items, candidates, held = [], {}, {}
        generated_images=[]
        last_at = None
        with InferenceRunner(models,cpu_profile) as runner:
            for sample in samples:
                at = sample["captured_at"]
                moment = datetime.fromisoformat(at.replace("Z", "+00:00"))
                if moment.tzinfo is None or (last_at and moment <= last_at): raise ValueError("monotonic_capture_utc_required")
                last_at = moment
                if moment + timedelta(days=7) <= datetime.now(timezone.utc): raise ValueError("raw_frame_expired")
                image, roi, matrix, size = geometry(sample["frame"], profile)
                quality = frame_quality(image)
                if run.get("conditions") != {"daylight":True,"dry_asphalt":True}:
                    quality = {"usable":False,"reasons":quality["reasons"]+["daylight_dry_asphalt_not_verified"]}
                defect = run.get("postcheck",{}).get("defect_from")
                if defect and moment>=datetime.fromisoformat(defect):
                    quality = {"usable":False,"reasons":quality["reasons"]+["postcheck_mount_or_lens_defect"]}
                location = associate_location(at, sample["fixes"])
                near_fixes = [(abs((datetime.fromisoformat(f["utc"]) - moment).total_seconds()), f) for f in sample["fixes"] if f.get("fix_valid")]
                nearest = min(near_fixes, default=(99, {}), key=lambda item: item[0])
                speed = nearest[1].get("speed_kmh") if nearest[0] <= .5 else None
                if speed is not None and speed > 20:
                    quality = {"usable": False, "reasons": quality["reasons"] + ["speed_out_of_scope"]}
                predictions = runner.predict(image)
                detections = predictions["pothole"]
                # Candidate center must be in calibrated visible road ROI.
                detections = [d for d in detections if roi[min(image.shape[0]-1, max(0, int(d["bbox_xywh_px"][1] + d["bbox_xywh_px"][3]/2))),
                                                           min(image.shape[1]-1, max(0, int(d["bbox_xywh_px"][0] + d["bbox_xywh_px"][2]/2)))]]
                tracks = tracker.update(detections, moment.timestamp())
                paint = predictions["lane"]
                privacy = predictions["redaction"]
                safe = redact(image, privacy)
                image_path = output / f"{sample['frame_id']:09d}.redacted.png"
                if not cv2.imwrite(str(image_path), safe): raise RuntimeError("image_write_failed")
                generated_images.append(image_path)
                valid = roi.copy()
                luminance = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                valid[(luminance<=5)|(luminance>=250)] = 0
                for detection in privacy + detections:
                    x, y, w, h = map(int, detection["bbox_xywh_px"])
                    valid[max(0,y):min(valid.shape[0],y+h),max(0,x):min(valid.shape[1],x+w)] = 0
                if not quality["usable"]: valid[:] = 0
                bev = cv2.warpPerspective(paint, matrix, size, flags=cv2.INTER_NEAREST)
                valid = cv2.warpPerspective(valid, matrix, size, flags=cv2.INTER_NEAREST)
                gray = cv2.warpPerspective(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), matrix, size)
                distance, reason = motion.update(gray, at, speed, location["quality"] == "valid" and quality["usable"])
                bev_image=cv2.warpPerspective(image,matrix,size)
                valid,lane_status=lane_evidence(bev_image,bev,valid,profile,distance)
                evidence = {"frame_id": sample["frame_id"], "captured_at": at, "location": location,
                            "quality": quality, "image_path": image_path.relative_to(workspace_path(".")).as_posix(),
                            "image_sha256": sha256(image_path)}
                evidence["lineage"]={key:sample[key] for key in ("sample_id","source_id") if sample.get(key)}
                evidence["overlays"] = {}
                for track in profile["measurement"]["lane_tracks"]:
                    left,right = track["bev_x_range"]
                    only = np.zeros_like(bev); only[:,int(left):int(right)] = bev[:,int(left):int(right)] & valid[:,int(left):int(right)]
                    mask = cv2.warpPerspective(only,np.linalg.inv(matrix),(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
                    polygon = cv2.perspectiveTransform(np.array([[[left,0],[right,0],[right,size[1]-1],[left,size[1]-1]]],np.float32),np.linalg.inv(matrix))[0]
                    polygon[:,0] = np.clip(polygon[:,0],0,image.shape[1]); polygon[:,1] = np.clip(polygon[:,1],0,image.shape[0])
                    evidence["overlays"][track["lane_track_id"]] = {"paint_mask_rle": overlay_mask(mask), "outline_polygon_px": polygon.tolist()}
                if distance is not None:
                    segments.add(bev, valid, distance, motion.epoch, evidence, float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                else:
                    for track in profile["measurement"]["lane_tracks"]:held[(motion.epoch,track["lane_track_id"])]=(track,evidence)
                for track in tracks:
                    # One representative observation per local track/event, with strongest evidence.
                    rank = (track["track_observations"], track["confidence"])
                    if track["event_id"] not in candidates or rank > candidates[track["event_id"]][0]:
                        candidates[track["event_id"]] = (rank, track.copy(), evidence)
                frame_items.append({"frame_id": sample["frame_id"], "captured_at": at, "location": location,
                                    "motion_distance_m": distance, "motion_reason": reason, "quality": quality,"lane_evidence":lane_status})
        if not frame_items: raise ValueError("no_analysis_frames")
        observations = []
        def observation(kind, event, detail, evidence):
            return {"schema_version": "0.4", "observation_id": str(uuid.uuid4()), "event_id": event,
                    "run_id": run_id, "device_id": device_id, "type": kind, "captured_at": evidence["captured_at"],
                    "frame_id": evidence["frame_id"], "location": evidence["location"], "profile_id": profile["profile_id"],
                    "model_version": bundle["version"], "config_hash": config_hash,
                    "pothole": detail if kind == "pothole" else None, "lane": detail if kind == "lane_segment" else None,
                    "quality": evidence["quality"], "image": {"file_id": str(uuid.uuid4()), "sha256": evidence["image_sha256"],
                    "redacted": True, "reviewed_for_sharing": False}, "review": {"status": "unreviewed", "priority": "normal"}}
        for _, track, evidence in candidates.values():
            detail = {k: track[k] for k in ("bbox_xywh_px", "confidence", "track_observations", "candidate_state")}
            detail["bbox_xywh_px"] = [float(x) for x in detail["bbox_xywh_px"]]
            detail["occluded"] = False
            observations.append({"observation": observation("pothole", track["event_id"], detail, evidence), "image_path": evidence["image_path"],"lineage":evidence["lineage"]})
        lane_results = segments.finish()
        represented={(item["distance_epoch"],item["lane_track_id"]) for item in lane_results}
        for (epoch,identifier),(track,evidence) in held.items():
            if (epoch,identifier) in represented:continue
            segment_id=str(uuid.uuid5(uuid.UUID(run_id),f"held:{epoch}:{identifier}"))
            lane_results.append({"lane":{"segment_id":segment_id,"estimated_length_m":None,"retention_pct":None,
                "grade":"unassessable","color":track["color"],"line_type":track["line_type"],"valid_area_ratio":None,
                "segment_support_ratio":None,"outline_supported":False,"patch_ids":[],"unassessable_reason":"missing_distance_basis"},
                "evidence":evidence,"lane_track_id":identifier,"distance_epoch":epoch,"start_m":None,"end_m":None})
        for item in lane_results:
            overlay = item["evidence"]["overlays"][item["lane_track_id"]]
            item["lane"]["outline_polygon_px"] = overlay["outline_polygon_px"]
            if overlay["paint_mask_rle"] is not None: item["lane"]["paint_mask_rle"] = overlay["paint_mask_rle"]
            else: item["overlay_omitted_reason"] = "rle_count_limit"
            observations.append({"observation": observation("lane_segment", item["lane"]["segment_id"], item["lane"], item["evidence"]),
                                 "image_path": item["evidence"]["image_path"],"lineage":item["evidence"]["lineage"]})
        retained={workspace_path(item["image_path"]) for item in observations}
        removed_images=0
        for image_path in generated_images:
            if image_path not in retained:
                image_path.unlink();removed_images+=1
        result = {"status": "awaiting_human_review", "run": run, "device_id": device_id,
                  "profile_id": profile["profile_id"], "model_version": bundle["version"], "config_hash": config_hash,
                  "frames": frame_items, "items": observations, "lane_segments": lane_results,
                  "runtime":{**runtime_metadata(models),"cpu_profile":cpu_profile,"model_workers":profile_options(cpu_profile)["model_workers"],"elapsed_seconds":time.perf_counter()-started,
                             "measured_processing_fps":len(frame_items)/max(time.perf_counter()-started,1e-9)},
                  "model_acceptance":{task:model.manifest.get("production_approved",False) for task,model in models.items()},
                  "retained_photos":len(retained),"removed_unreferenced_photos":removed_images,
                  "review_instructions": "Inspect every selected image; supply observation IDs to approve-review. No automatic approval."}
        atomic_json(output / "review.json", result)
        return result

    finally:close_models(models)


def execute_config(config_path,*,cpu_profile=None,inference_backend=None,gpu_profile=None):
    config = read_json(config_path)
    keys = {"run_id", "device_id", "profile_id", "model_bundle_id", "run_manifest", "profile_path", "bundle_path", "output"}
    if not keys<=set(config) or set(config)-keys-{"ca_bundle","cpu_profile","inference_backend","gpu_profile"}: raise ValueError("workflow_config_keys_mismatch")
    chosen_profile=cpu_profile if cpu_profile is not None else config.get("cpu_profile","default")
    profile_options(chosen_profile)
    chosen_backend=inference_backend if inference_backend is not None else config.get("inference_backend","cpu")
    chosen_gpu=gpu_profile if gpu_profile is not None else config.get("gpu_profile")
    if config.get("ca_bundle"):gateway_tls(config["ca_bundle"])
    run, profile, bundle = read_json(config["run_manifest"]), load_profile(config["profile_path"]), read_json(config["bundle_path"])
    from .equipment import CHECKS
    if set(run.get("precheck",{}))!=CHECKS or any(value is not True for value in run["precheck"].values()):
        raise ValueError("verified_run_precheck_required")
    for key in ("run_id", "device_id", "profile_id", "model_bundle_id"): str(uuid.UUID(config[key]))
    if run["run_id"] != config["run_id"] or profile["profile_id"] != config["profile_id"] or bundle["model_bundle_id"] != config["model_bundle_id"]:
        raise ValueError("workflow_identity_mismatch")
    if run.get("device_id") != config["device_id"] or run.get("profile_id") != config["profile_id"]:
        raise ValueError("capture_identity_mismatch")
    if run.get("source_kind")=="camera":
        from .equipment import collection_plan
        declared=run.get("collection_plan")
        if not declared or run.get("stationary_start_confirmed") is not True:raise ValueError("capture_collection_plan_required")
        if collection_plan(declared["path"])!=declared:raise ValueError("capture_collection_plan_changed")
    samples = run_samples(config["run_manifest"])
    try:
        result = analyze_samples(samples, config["output"], run=run, profile_path=config["profile_path"],
                                 device_id=config["device_id"], bundle=bundle,cpu_profile=chosen_profile,inference_backend=chosen_backend,gpu_profile=chosen_gpu)
        if config.get("ca_bundle"):
            result["ca_bundle"]=workspace_path(config["ca_bundle"]).relative_to(workspace_path(".")).as_posix()
            atomic_json(workspace_path(config["output"])/"review.json",result)
    finally: samples.close()
    return {"status": "completed", "run_id": config["run_id"], "model_bundle_id": config["model_bundle_id"],
            "result_path": (workspace_path(config["output"]) / "review.json").relative_to(workspace_path(".")).as_posix(),
            "observation_count": len(result["items"])}


def derive_path(package,max_points=1000,max_segments=500):
    """Never bridge absent fixes, UTC gaps, or quality/assessment transitions."""
    if type(max_points) is not int or type(max_segments) is not int or not 2<=max_points<=1000 or not 1<=max_segments<=500:raise ValueError("invalid_path_limits")
    max_segments=min(max_segments,max_points//2)
    segments=[];current=[];current_key=None
    def flush():
        nonlocal current
        if len(current)>=2:
            first,last=current[0],current[-1]
            segments.append({"from":first[0],"to":last[0],"coordinates":[row[1] for row in current],
                "quality":current_key[0],"visible_scope":{"profile_id":package["profile_id"],"roi_id":"main",
                "assessment_state":current_key[1],"reason":current_key[2]}})
        current=[]
    for frame in package.get("frames",[]):
        location=frame["location"];lat,lon=location.get("lat"),location.get("lon")
        at=datetime.fromisoformat(frame["captured_at"].replace("Z","+00:00"))
        if at.tzinfo is None:raise ValueError("path_utc_required")
        if (location.get("quality") not in ("valid","degraded") or lat is None or lon is None or
            not np.isfinite([lat,lon]).all() or not -90<=lat<=90 or not -180<=lon<=180 or
            location.get("age_ms") is not None and location["age_ms"]>2000):
            flush();current_key=None;continue
        reasons=list(frame["quality"].get("reasons",[]))
        lanes=list(frame.get("lane_evidence",{}).values())
        if not lanes:reasons.append("lane_evidence_unavailable")
        else:reasons.extend(row.get("reason") or "lane_outline_unsupported" for row in lanes if not row.get("outline_supported"))
        assessed=frame["quality"]["usable"] and bool(lanes) and all(row.get("outline_supported") for row in lanes)
        key=(location["quality"],"assessed" if assessed else "unassessable",None if assessed else ";".join(sorted(set(reasons)))[:500] or "quality_or_lane_evidence_insufficient")
        if current and (key!=current_key or not 0<(at-datetime.fromisoformat(current[-1][0].replace("Z","+00:00"))).total_seconds()<=2):flush()
        current_key=key
        current.append((frame["captured_at"],[float(lon),float(lat)]))
    flush()
    omitted=max(0,len(segments)-max_segments)
    if omitted:
        indices=np.linspace(0,len(segments)-1,max_segments,dtype=int)
        segments=[segments[int(index)] for index in indices]
    available=max_points-2*len(segments)
    interiors=[(i,j) for i,segment in enumerate(segments) for j in range(1,len(segment["coordinates"])-1)]
    selected=set(interiors)
    if len(interiors)>available:
        selected={interiors[int(i)] for i in np.linspace(0,len(interiors)-1,available,dtype=int)} if available else set()
    removed=0
    for i,segment in enumerate(segments):
        coords=segment["coordinates"]
        reduced=[point for j,point in enumerate(coords) if j in (0,len(coords)-1) or (i,j) in selected]
        removed+=len(coords)-len(reduced);segment["coordinates"]=reduced
    return {"segments":segments,"omitted_segments":omitted,"removed_interior_points":removed}


def run_registration(run,device_id=None,profile_id=None):
    manifest_hash=hashlib.sha256(json.dumps(run,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return {"run_id":run["run_id"],"device_id":device_id or run["device_id"],"profile_id":profile_id or run["profile_id"],
            "started_at":run["started_at"],"manifest_hash":manifest_hash,"precheck":run["precheck"]}


def register_capture(manifest_path,gateway,device_token,*,transport=None,ca_bundle=None):
    run=read_json(manifest_path)
    if run.get("status") not in ("completed","interrupted") or not run.get("ended_at") or not run.get("postcheck"):
        raise ValueError("finished_postchecked_capture_required")
    parsed=urlparse(gateway)
    if parsed.scheme not in ("http","https") or parsed.scheme=="http" and parsed.hostname not in ("localhost","127.0.0.1","::1"):
        raise ValueError("https_or_loopback_gateway_required")
    body=run_registration(run)
    with httpx.Client(base_url=gateway.rstrip("/")+"/",headers={"Authorization":"Bearer "+device_token},transport=transport,
                      verify=gateway_tls(ca_bundle),trust_env=False,timeout=30) as client:
        response=client.post("runs",json=body);response.raise_for_status()
    return {"run_id":run["run_id"],"status":"registered_for_analysis","manifest_hash":body["manifest_hash"]}


def sync_run(package_path, gateway, device_token, *, transport=None,ca_bundle=None):
    package = read_json(package_path); run = package["run"]
    parsed = urlparse(gateway)
    if parsed.scheme not in ("http", "https") or parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("https_or_loopback_gateway_required")
    body=run_registration(run,package["device_id"],package["profile_id"])
    manifest_hash=body["manifest_hash"]
    path=derive_path(package)
    with httpx.Client(base_url=gateway.rstrip("/") + "/", headers={"Authorization": "Bearer " + device_token}, transport=transport,
                      verify=gateway_tls(ca_bundle or package.get("ca_bundle")),trust_env=False,timeout=30) as client:
        response = client.post("runs", json=body); response.raise_for_status()
        response=client.put("runs/"+run["run_id"]+"/path",json={"segments":path["segments"],"manifest_hash":manifest_hash},
                            headers={"If-Match":'"'+manifest_hash+'"'})
        response.raise_for_status()
        response = client.patch("runs/" + run["run_id"], json={"ended_at": run["ended_at"], "status": run["status"],
                               "manifest_hash": manifest_hash}, headers={"If-Match": '"' + manifest_hash + '"'})
        response.raise_for_status()
    return {"run_id": run["run_id"], "status": "synchronized","path_segments":len(path["segments"]),
            "omitted_segments":path["omitted_segments"],"removed_interior_points":path["removed_interior_points"]}


def approve_review(package_path, decisions_path, *, reviewed_by, queue_path="data/agent/outbox.sqlite", key_path="data/agent/local.key"):
    package, decisions = read_json(package_path), read_json(decisions_path)
    if not reviewed_by.strip(): raise ValueError("reviewer_required")
    selected = {row["observation_id"]: row for row in decisions}
    known = {row["observation"]["observation_id"] for row in package["items"]}
    if set(selected) - known or len(selected) != len(decisions): raise ValueError("invalid_review_observation_ids")
    outbox, queued = Outbox(queue_path, key_path), []
    # Validate entire review before enqueueing any item.
    ready = []
    from services.inspection.schemas import Observation
    for item in package["items"]:
        value = json.loads(json.dumps(item["observation"]))
        decision = selected.get(value["observation_id"])
        if not decision or decision.get("approve") is not True: continue
        if decision.get("image_sha256") != value["image"]["sha256"] or sha256(workspace_path(item["image_path"], exists=True)) != value["image"]["sha256"]:
            raise ValueError("reviewed_image_hash_mismatch")
        value["image"]["reviewed_for_sharing"] = True
        Observation.model_validate(value)
        ready.append((value, item["image_path"],item.get("lineage")))
    for value, path,lineage in ready:
        queued.append(outbox.enqueue(value, path, reviewed_by=reviewed_by, confirmed_human_review=True,lineage=lineage))
    return {"queued": queued, "reviewed_by": reviewed_by}
