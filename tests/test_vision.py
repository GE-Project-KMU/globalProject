"""Deterministic local tooling tests; synthetic fixtures are NOT accuracy evidence."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import uuid
import os
import subprocess
import sys
import cv2
import httpx
import numpy as np
import pytest
from cryptography.exceptions import InvalidTag
from agent.paths import ROOT, workspace_path, sha256, atomic_json
from agent.capture import EncryptedStore, record, replay
from agent.dataset import decode_rle, split_connected, validate_release, validate_annotation
from agent.gnss import parse_nmea, associate_location, unavailable, build_fixes
from agent.lane import measure_patch, aggregate_segment, api_lane, grade
from agent.models import readiness, OnnxModel, ModelNotReady, redact, decode_yolox_raw
from agent.outbox import Outbox
from agent.pipeline import Tracker, straight_solid_outline
from agent.retention import purge_expired_runs


@pytest.fixture
def folder(monkeypatch):
    path = ROOT / ".cache" / "vision-tests" / str(uuid.uuid4())
    path.mkdir(parents=True)
    from agent import governance
    monkeypatch.setattr(governance,"REGISTRY",path/"registry")
    monkeypatch.setattr(governance,"REVOCATIONS",path/"revocations.json")
    return path


def uid():
    return str(uuid.uuid4())


def sentence(body):
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    return f"${body}*{checksum:02X}"


def synthetic_video(path, frames=12, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (128, 48))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((48, 128, 3), np.uint8)
        frame[:, :64] = (10 + index, 90, 180)
        frame[:, 64:] = (180, 90, 10 + index)
        writer.write(frame)
    writer.release()
    return path


def fixture_observation(folder):
    image = folder / "redacted.png"
    assert cv2.imwrite(str(image), np.zeros((48, 64, 3), np.uint8))
    captured = datetime.now(timezone.utc).isoformat()
    observation = {"schema_version": "0.4", "observation_id": uid(), "event_id": uid(),
                   "run_id": uid(), "device_id": uid(), "profile_id": uid(), "captured_at": captured,
                   "frame_id": 1, "model_version": "trained-test", "config_hash": "a" * 64,
                   "type": "pothole", "location": unavailable(),
                   "pothole": {"bbox_xywh_px": [1., 2., 8., 8.], "confidence": .9, "track_observations": 3,
                               "candidate_state": "multi_frame", "occluded": False}, "lane": None,
                   "quality": {"usable": True, "reasons": []},
                   "image": {"file_id": uid(), "sha256": sha256(image), "redacted": True, "reviewed_for_sharing": True},
                   "review": {"status": "unreviewed", "priority": "normal"}}
    return observation, image


def test_workspace_boundary_rejects_parent_and_absolute_escape():
    with pytest.raises(ValueError):
        workspace_path("../outside")
    with pytest.raises(ValueError):
        workspace_path(ROOT.parent / "outside")


def test_encryption_round_trip_and_tamper(folder):
    store = EncryptedStore(folder / "key")
    source = folder / "private.txt"
    source.write_bytes(b"not-to-be-shared")
    item = store.encrypt(source)
    assert not source.exists()
    encrypted = workspace_path(item["path"])
    assert b"not-to-be-shared" not in encrypted.read_bytes()
    assert store.decrypt(encrypted, folder / "opened.txt").read_bytes() == b"not-to-be-shared"
    raw = bytearray(encrypted.read_bytes())
    raw[-1] ^= 1
    encrypted.write_bytes(raw)
    with pytest.raises(InvalidTag):
        store.decrypt(encrypted, folder / "tampered.txt")


def test_record_encrypted_stereo_and_replay(folder):
    source = synthetic_video(folder / "source.mp4", frames=12)
    result = record(str(source), folder / "runs", seconds=1.2, chunk_seconds=1,
                    minimum_free_bytes=0, key_path=folder / "key")
    assert result["status"] == "completed" and result["frames"] == 12
    assert len(result["chunks"]) == 2
    chunk = result["chunks"][0]
    assert len(chunk["eyes"]) == 2
    assert all(item["encrypted"] for item in chunk["eyes"])
    store = EncryptedStore(folder / "key")
    left = store.decrypt(chunk["eyes"][0]["path"], folder / "left.mp4")
    frames = list(replay(left, sample_fps=5))
    assert len(frames) == 5 and frames[0]["frame"].shape == (48, 64, 3)
    assert frames[0]["frame"][10, 10, 2] > frames[0]["frame"][10, 10, 0]
    index = store.decrypt(chunk["index"]["path"], folder / "index.json")
    stamps = json.loads(index.read_text())["frames"]
    assert len(stamps) == 10 and stamps[0]["frame_id"] == 0
    assert not list((folder / "runs").glob("**/*.mp4"))


def test_record_disk_gate_and_invalid_source(folder):
    source = synthetic_video(folder / "source.mp4")
    with pytest.raises(RuntimeError, match="insufficient_disk"):
        record(str(source), folder / "runs", minimum_free_bytes=10**30)
    with pytest.raises(FileNotFoundError):
        record(str(folder / "absent.mp4"), folder / "runs")
    with pytest.raises(ValueError):
        list(replay(source, sample_fps=0))


def test_nmea_checksum_coordinates_and_invalid_fix():
    row = parse_nmea(sentence("GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W"))
    assert row["lat"] == pytest.approx(48.1173)
    assert row["lon"] == pytest.approx(11.5166667)
    assert row["speed_kmh"] == pytest.approx(41.4848)
    assert row["utc"].startswith("1994-03-23T12:35:19")
    nofix = parse_nmea(sentence("GPRMC,123519,V,,,,,,,230394,,"))
    assert nofix["lat"] is None and not nofix["fix_valid"]
    with pytest.raises(ValueError, match="checksum"):
        parse_nmea("$GPRMC,123519*00")


def test_gnss_gap_quality_and_strict_api_location():
    from services.inspection.schemas import Location
    now = datetime.now(timezone.utc)
    row = {"fix_valid": True, "utc": now.isoformat(), "lat": 37., "lon": 127., "hdop": 1.2}
    result = associate_location(now.isoformat(), [row])
    assert result["quality"] == "valid" and result["accuracy_m"] is None
    Location.model_validate(result)
    degraded = associate_location(now.isoformat(), [row | {"hdop": 4.}])
    assert degraded["quality"] == "degraded"
    missing = associate_location((now + timedelta(seconds=3)).isoformat(), [row])
    assert missing["quality"] == "unavailable" and missing["lat"] is None
    Location.model_validate(missing)


def test_gnss_pairs_gga_quality_and_rejects_non_finite():
    received = datetime.now(timezone.utc).isoformat()
    values = [{"sentence": sentence("GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W"), "received_at": received},
              {"sentence": sentence("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"), "received_at": received}]
    result = build_fixes(values)
    assert not result["errors"] and result["fixes"][0]["hdop"] == .9
    assert associate_location(result["fixes"][0]["utc"], result["fixes"])["quality"] == "valid"
    with pytest.raises(ValueError, match="numeric"):
        parse_nmea(sentence("GPRMC,123519,A,4807.038,N,01131.000,E,nan,084.4,230394,003.1,W"))


@pytest.mark.parametrize("ratio,expected", [(0,"worn_candidate"),(59.999,"worn_candidate"),(60,"monitor"),(79.999,"monitor"),(80,"visible_good"),(100,"visible_good")])
def test_lane_grade_boundaries(ratio, expected):
    assert grade(ratio) == expected


def test_mask_holes_gaps_invalid_and_no_denominator():
    outline = np.ones((4, 4), np.uint8)
    paint = np.zeros_like(outline)
    paint[:, :2] = 1
    result = measure_patch(outline, np.zeros_like(outline), paint, outline)
    assert result["retention_pct"] == 50 and result["grade"] == "worn_candidate"
    gap = np.zeros_like(outline); gap[:, 2:] = 1
    result = measure_patch(outline, gap, paint, outline)
    assert result["retention_pct"] == 100
    valid = np.zeros_like(outline); valid[:2] = 1
    assert measure_patch(outline, np.zeros_like(outline), paint, valid)["retention_pct"] is None
    missing = measure_patch(np.zeros_like(outline), gap, paint, valid)
    assert missing["valid_area_ratio"] is None and missing["retention_pct"] is None
    with pytest.raises(ValueError, match="shape"):
        measure_patch(outline, outline[:2], paint, valid)
    with pytest.raises(ValueError):
        measure_patch(outline, gap, paint, np.full_like(outline, 255))


def test_unique_patch_weighted_segment_and_strict_api():
    from services.inspection.schemas import Lane
    patch_id = uid()
    patch = {"patch_id": patch_id, "effective_area": 20, "paint_area": 10, "grade": "worn_candidate"}
    result = aggregate_segment([patch, patch], segment_support_ratio=.8, estimated_length_m=10.)
    assert result["retention_pct"] == 50 and len(result["patch_ids"]) == 1
    mapped = api_lane(result, segment_id=uid(), color="white", line_type="solid", outline_supported=True, valid_area_ratio=.9)
    Lane.model_validate(mapped)
    missing = aggregate_segment([patch], segment_support_ratio=None, estimated_length_m=None)
    Lane.model_validate(api_lane(missing, segment_id=uid()))
    assert missing["retention_pct"] is None
    with pytest.raises(ValueError, match="conflicting"):
        aggregate_segment([patch, patch | {"paint_area": 9}], segment_support_ratio=.9, estimated_length_m=10)
    assert aggregate_segment([patch], segment_support_ratio=.79999, estimated_length_m=10)["retention_pct"] is None


def test_coco_rle_and_annotation_bounds():
    mask = decode_rle({"size": [4, 4], "counts": [0, 8, 8]})
    assert int(mask.sum()) == 8 and np.all(mask[:, :2] == 1)
    with pytest.raises(ValueError):
        decode_rle({"size": [4, 4], "counts": [0, 15]})
    sample = {"dataset_schema_version": "1.0", "sample_id": uid(), "image": {"width": 1280, "height": 720},
              "potholes": [{"annotation_id": uid(), "physical_object_id": uid(), "bbox_xywh_px": [1279, 1, 2, 2], "certainty": "confirmed"}],
              "annotation_status": "draft"}
    assert "bbox_out_of_bounds" in validate_annotation(sample)
    sample["potholes"][0]["bbox_xywh_px"] = [0, 0, 1, 1]
    assert validate_annotation(sample) == []


def test_connected_split_links_parent_run_and_nested_object():
    obj = uid()
    samples = [{"sample_id": "a", "run_id": "run-a", "potholes": [{"physical_object_id": obj}]},
               {"sample_id": "b", "run_id": "run-b", "potholes": [{"physical_object_id": obj}]},
               {"sample_id": "crop", "parent_sample_id": "a"},
               {"sample_id": "c"}, {"sample_id": "d"}, {"sample_id": "e"}]
    split = split_connected(samples)
    group = [row for row in split if row["sample_id"] in ("a", "b", "crop")]
    assert len({row["split"] for row in group}) == 1
    assert len({row["leakage_group_id"] for row in group}) == 1
    assert split == split_connected(samples)
    with pytest.raises(ValueError):
        split_connected(samples[:3])


def test_release_leakage_expiry_and_hash(folder):
    image = folder / "frame.bin"; image.write_bytes(b"frame")
    now = datetime.now(timezone.utc)
    base = {"sample_id": uid(), "source_id": uid(), "leakage_group_id": "same-location", "annotation_version": "1.0", "path": str(image), "sha256": sha256(image), "privacy_status": "approved", "qc_status": "qc_passed", "allowed_use": "test-authorized", "expires_at": (now + timedelta(days=1)).isoformat(), "split": "train", "source_hash": "shared"}
    assert validate_release([base], now=now)["valid"]
    second = base | {"sample_id": uid(), "split": "test"}
    assert any(e["code"] == "split_leakage" for e in validate_release([base, second], now=now)["errors"])
    assert not validate_release([base | {"expires_at": now.isoformat()}], now=now)["valid"]
    image.write_bytes(b"changed")
    assert any(e["code"] == "file_hash_mismatch" for e in validate_release([base], now=now)["errors"])


def test_model_missing_class_and_digest_gate(folder):
    assert readiness(folder / "missing.json")["status"] == "model_not_ready"
    with pytest.raises(ModelNotReady):
        OnnxModel(folder / "missing.json")
    weights = folder / "target.onnx"; weights.write_bytes(b"not-real-onnx")
    manifest = {"format": "onnx", "task": "pothole", "classes": ["person", "car"], "target_trained": True,
                "dataset_release": "synthetic-test", "weights": str(weights), "sha256": sha256(weights),
                "input_size": [416, 416], "output_layout": "yolox_decoded_xywh"}
    path = folder / "model.json"; atomic_json(path, manifest)
    assert "class_mapping" in readiness(path)["reason"]
    manifest["classes"] = ["pothole"]; manifest["sha256"] = "0" * 64; atomic_json(path, manifest)
    assert "digest" in readiness(path)["reason"]
    manifest["sha256"] = sha256(weights); manifest["target_trained"] = "yes"; atomic_json(path, manifest)
    assert "provenance" in readiness(path)["reason"]
    manifest["target_trained"] = True; atomic_json(path, manifest)
    with pytest.raises(ModelNotReady, match="onnx_load_failed"):
        OnnxModel(path)


def test_redaction_and_tracker_do_not_create_extra_objects():
    frame = np.full((40, 40, 3), 255, np.uint8)
    censored = redact(frame, [[10, 10, 10, 10]], padding=0)
    assert not censored[10:20, 10:20].any() and frame.all()
    tracker = Tracker()
    detection = {"bbox_xywh_px": [1., 1., 10., 10.], "confidence": .9}
    first = tracker.update([detection], 0)[0]
    second = tracker.update([detection], .1)[0]
    third = tracker.update([detection], .2)[0]
    assert first["event_id"] == third["event_id"] and third["candidate_state"] == "multi_frame"
    assert tracker.update([], .3) == []
    assert tracker.update([detection], 2)[0]["event_id"] != first["event_id"]


def test_conservative_outline_does_not_invent_missing_line():
    blank = np.zeros((100, 100), np.uint8)
    assert straight_solid_outline(blank, 10)[0] is None
    blank[:, 45:55] = 1
    outline, reason = straight_solid_outline(blank, 10)
    assert outline is not None and reason is None


def test_outbox_requires_human_hash_and_encrypts(folder):
    observation, image = fixture_observation(folder)
    queue = Outbox(folder / "queue.sqlite", folder / "key")
    with pytest.raises(ValueError, match="human"):
        queue.enqueue(observation, image, reviewed_by="reviewer")
    queue.enqueue(observation, image, reviewed_by="reviewer", confirmed_human_review=True)
    assert observation["captured_at"].encode() not in (folder / "queue.sqlite").read_bytes()
    assert queue.enqueue(observation, image, reviewed_by="reviewer", confirmed_human_review=True) == observation["observation_id"]
    with pytest.raises(ValueError, match="conflict"):
        queue.enqueue(observation | {"frame_id": 2}, image, reviewed_by="reviewer", confirmed_human_review=True)


def test_outbox_real_request_shapes_ack_and_restart(folder):
    observation, image = fixture_observation(folder)
    queue = Outbox(folder / "queue.sqlite", folder / "key")
    queue.enqueue(observation, image, reviewed_by="reviewer", confirmed_human_review=True)
    calls = []
    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("image-approvals"):
            assert request.headers["Authorization"] == "Bearer reviewer"
            body = json.loads(request.content); assert body["sha256"] == sha256(image)
            return httpx.Response(201, json={"data": {"approval_id": uid()}})
        if request.url.path.endswith("images"):
            assert "multipart/form-data" in request.headers["content-type"]
            return httpx.Response(201, json={"data": {"image_id": observation["image"]["file_id"]}})
        assert json.loads(request.content) == observation
        return httpx.Response(201, json={"data": {"observation_id": observation["observation_id"]}})
    result = queue.drain("http://127.0.0.1:8000/api/v1", device_token="device", reviewer_token="reviewer", transport=httpx.MockTransport(handler))
    assert result[0]["state"] == "acked" and len(calls) == 3
    assert Outbox(folder / "queue.sqlite", folder / "key").list()[0]["state"] == "acked"
    assert queue.drain("http://127.0.0.1:8000/api/v1", device_token="device", transport=httpx.MockTransport(handler)) == []


def test_outbox_retries_only_transient_and_lease_recovery(folder):
    observation, image = fixture_observation(folder)
    queue = Outbox(folder / "queue.sqlite", folder / "key")
    queue.enqueue(observation, image, reviewed_by="reviewer", confirmed_human_review=True)
    transport = httpx.MockTransport(lambda request: httpx.Response(503, json={"error": "unavailable"}))
    queue.drain("http://127.0.0.1:8000/api/v1", device_token="d", reviewer_token="r", transport=transport)
    item = queue.list()[0]
    assert item["state"] == "pending" and item["attempts"] == 1 and item["next_at"] > 0
    with queue.connect() as db:
        db.execute("UPDATE queue SET state='uploading',lease_until=0,next_at=0")
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "invalid"}))
    queue.drain("http://127.0.0.1:8000/api/v1", device_token="d", reviewer_token="r", transport=transport)
    assert queue.list()[0]["state"] == "blocked"


def test_outbox_expiry_and_changed_file_do_not_upload(folder):
    observation, image = fixture_observation(folder)
    queue = Outbox(folder / "queue.sqlite", folder / "key")
    queue.enqueue(observation, image, reviewed_by="r", confirmed_human_review=True)
    def forbidden(request):
        raise AssertionError("Must not upload")
    image.write_bytes(image.read_bytes() + b"changed")
    queue.drain("http://127.0.0.1:8000/api/v1", device_token="d", reviewer_token="r", transport=httpx.MockTransport(forbidden))
    assert queue.list()[0]["error"] == "reviewed_file_changed"
    expired = observation | {"observation_id": uid(), "captured_at": (datetime.now(timezone.utc) - timedelta(days=91)).isoformat(), "image": observation["image"] | {"sha256": sha256(image)}}
    with pytest.raises(ValueError, match="expired"):
        queue.enqueue(expired, image, reviewed_by="r", confirmed_human_review=True)
    with pytest.raises(ValueError):
        queue.drain("http://example.com/api/v1", device_token="d")


def test_retention_dry_run_then_verified_delete(folder):
    root = folder / "runs"; target = root / uid(); target.mkdir(parents=True)
    atomic_json(target / "manifest.json", {"run_id": target.name, "expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})
    assert purge_expired_runs(root)[0]["deleted"] is False and target.exists()
    assert purge_expired_runs(root, dry_run=False)[0]["deleted"] is True and not target.exists()


def test_outbox_expires_previously_queued_item_without_http(folder, monkeypatch):
    import agent.outbox as module
    observation, image = fixture_observation(folder)
    queue = Outbox(folder / "queue.sqlite", folder / "key")
    queue.enqueue(observation, image, reviewed_by="reviewer", confirmed_human_review=True)
    future = datetime.now(timezone.utc) + timedelta(days=91)
    class FutureDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return future
    monkeypatch.setattr(module, "datetime", FutureDateTime)
    def forbidden(request):
        raise AssertionError("Expired queue must not send HTTP")
    result = queue.drain("http://127.0.0.1:8000/api/v1", device_token="d", transport=httpx.MockTransport(forbidden))
    assert result[0]["state"] == "expired" and queue.list()[0]["state"] == "expired"


def test_release_invalid_shape_and_missing_lineage_are_rejected():
    assert not validate_release({})["valid"]
    assert not validate_release([None, {"sample_id": []}])["valid"]
    result = validate_release([{"sample_id": uid(), "split": "test", "privacy_status": "approved", "qc_status": "qc_passed", "allowed_use": "test", "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}])
    assert any(error["code"] == "missing_path" for error in result["errors"])


def test_cli_missing_model_exit_code_is_explicit(folder):
    process = subprocess.run([sys.executable, "-m", "agent", "model-status", str(folder / "absent.json")],
                             cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert process.returncode == 2
    assert json.loads(process.stdout)["status"] == "model_not_ready"


def test_lane_adapter_letterbox_crops_padding_back_to_original_size():
    model = object.__new__(OnnxModel)
    model.manifest = {"task": "lane", "input_size": [32, 32]}
    class FakeNetwork:
        def setInput(self, value):
            assert value.shape == (1, 3, 32, 32)
        def forward(self):
            output = np.zeros((1, 2, 32, 32), np.float32)
            output[:, 1, :16, :] = 1
            return output
    model.net = FakeNetwork()
    result = model.predict(np.ones((20, 40, 3), np.uint8))
    assert result.shape == (20, 40) and result.all()


@pytest.mark.parametrize("boxes", [
    [[-100., 10., 150., 40.], [0., 10., 50., 40.]],
    [[50., 10., 150., 40.], [50., 10., 50., 40.]],
    [[10., -100., 40., 150.], [10., 0., 40., 50.]],
    [[10., 50., 40., 150.], [10., 50., 40., 50.]],
])
def test_detector_nms_preserves_boundary_boxes_distinct_before_clipping(boxes):
    # Unclipped IoU=1/3, but clipped IoU=1: clipping first wrongly suppresses
    # one candidate. Native YOLOX applies class NMS before source-image clipping.
    confidences = [.95, .9]
    rows = [[x+w/2, y+h/2, w, h, 1., score]
            for (x,y,w,h),score in zip(boxes,confidences)]
    model = object.__new__(OnnxModel)
    model.manifest = {"task":"pothole", "classes":["pothole"], "input_size":[100,100],
                      "output_layout":"yolox_decoded_xywh", "confidence":.35, "nms_iou":.5}
    class FakeNetwork:
        def setInput(self,value): pass
        def forward(self): return np.array([rows], np.float32)
    model.net = FakeNetwork()
    expected=[]
    for x,y,w,h in boxes:
        left,top=max(0.,x),max(0.,y)
        expected.append([left,top,min(100.,x+w)-left,min(100.,y+h)-top])
    assert len(cv2.dnn.NMSBoxes(boxes,confidences,.35,.5)) == 2
    assert len(cv2.dnn.NMSBoxes(expected,confidences,.35,.5)) == 1
    actual=model.predict(np.zeros((100,100,3),np.uint8))
    assert len(actual)==2
    for item,box,confidence in zip(actual,expected,confidences):
        assert item["bbox_xywh_px"]==pytest.approx(box)
        assert item["confidence"]==pytest.approx(confidence)
        assert item["class"]=="pothole"


@pytest.mark.parametrize("boxes", [
    [[-100.,10.,150.,40.],[-90.,10.,110.,40.]],
    [[50.,10.,150.,40.],[80.,10.,110.,40.]],
])
def test_detector_nms_suppresses_raw_overlap_before_clipping_and_rejects_empty(boxes):
    # Unclipped IoU=110/150, clipped IoU=20/50: clipping first would retain
    # an extra lower-confidence box. Fully out-of-image selections are removed.
    rows=[[x+w/2,y+h/2,w,h,1.,score] for (x,y,w,h),score in zip(boxes,[.95,.9])]
    rows.append([-50.,-50.,10.,10.,1.,.99])
    model=object.__new__(OnnxModel)
    model.manifest={"task":"redaction","classes":["face","license_plate"],"input_size":[100,100],
                    "output_layout":"yolox_decoded_xywh","confidence":.35,"nms_iou":.5}
    class FakeNetwork:
        def setInput(self,value): pass
        def forward(self): return np.array([[row+[0.] for row in rows]],np.float32)
    model.net=FakeNetwork()
    actual=model.predict(np.zeros((100,100,3),np.uint8))
    assert len(actual)==1
    x,y,w,h=boxes[0];left,top=max(0.,x),max(0.,y)
    assert actual[0]["bbox_xywh_px"]==pytest.approx([left,top,min(100.,x+w)-left,min(100.,y+h)-top])
    assert actual[0]["class"]=="face" and actual[0]["confidence"]==pytest.approx(.95)


def test_raw_yolox_decoding_grid_strides_and_count():
    raw = np.zeros((21, 6), np.float32)  # 32x32: 4x4 + 2x2 + 1x1
    result = decode_yolox_raw(raw, 32, 32)
    assert result[0, :4].tolist() == [0., 0., 8., 8.]
    assert result[1, :4].tolist() == [8., 0., 8., 8.]
    assert result[-1, :4].tolist() == [0., 0., 32., 32.]
    with pytest.raises(ModelNotReady, match="grid_count"):
        decode_yolox_raw(raw[:20], 32, 32)
    with pytest.raises(ModelNotReady, match="stride"):
        decode_yolox_raw(raw, 31, 32)


def test_real_external_baseline_load_and_inference_when_downloaded():
    manifest = ROOT / "data" / "models" / "external" / "pothole_yolox_s.json"
    if not manifest.exists():
        pytest.skip("Optional pinned external baseline not downloaded")
    state = readiness(manifest)
    assert state["ready"] and state["manifest"]["production_approved"] is False
    model = OnnxModel(manifest)
    detections = model.predict(np.full((360, 640, 3), 110, np.uint8))
    assert isinstance(detections, list)
    for detection in detections:
        assert detection["class"] == "pothole"
        assert 0 <= detection["confidence"] <= 1
        assert len(detection["bbox_xywh_px"]) == 4


def test_training_pair_preprocessing_matches_runtime_letterbox(folder):
    from agent.train_lane import prepare_pair
    image = folder / "train-image.png"; mask = folder / "train-mask.png"
    cv2.imwrite(str(image), np.full((20, 40, 3), 114, np.uint8))
    labels = np.zeros((20, 40), np.uint8); labels[:, :20] = 1
    cv2.imwrite(str(mask), labels)
    pixels, target = prepare_pair(image, mask, size=32)
    assert pixels.shape == (3, 32, 32) and target.shape == (32, 32)
    assert np.all(target[16:] == 255) and set(np.unique(target)) == {0, 1, 255}


def test_baseline_download_rejects_corrupt_payload_and_removes_partial(folder, monkeypatch):
    from agent import baseline
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"corrupt")))
    monkeypatch.setattr(baseline.httpx, "stream", client.stream)
    with pytest.raises(ValueError, match="checksum_mismatch"):
        baseline.download_external_baseline(folder)
    assert not (folder / baseline.MODEL_FILE).exists()
    assert not (folder / (baseline.MODEL_FILE + ".download")).exists()
    client.close()


def installation_profile(folder):
    value = {"profile_id":uid(),"calibrated_by":"fixture-operator","calibration":{
        "image_size":[64,48],"height_m":.4,"pitch_deg":30.,"camera_matrix":[[100.,0,32],[0,100.,24],[0,0,1]],
        "distortion":[0.,0.,0.,0.,0.],"homography":np.eye(3).tolist(),"roi_polygon_px":[[0,0],[63,0],[63,47],[0,47]],
        "visible_distance_m":[0.,2.4],"calibrated_at":datetime.now(timezone.utc).isoformat()},
        "measurement":{"bev_size":[64,48],"meters_per_pixel":.05,"lane_tracks":[{
            "lane_track_id":uid(),"color":"white","line_type":"solid","bev_x_range":[20,30],"outline_verified":True}]}}
    path=folder/"profile.json"; atomic_json(path,value)
    return path,value


def test_profile_geometry_homography_and_rejection(folder):
    from agent.profile import load_profile,geometry,calibrate_homography
    path,p=installation_profile(folder)
    image=np.zeros((48,64,3),np.uint8)
    assert geometry(image,load_profile(path))[1].sum()>1000
    with pytest.raises(ValueError,match="image_size"):
        geometry(image[:20],p)
    p["calibration"]["homography"]=[[0,0,0]]*3; atomic_json(path,p)
    with pytest.raises(ValueError,match="homography"): load_profile(path)
    request=folder/"points.json"
    atomic_json(request,{"image_points":[[0,0],[10,0],[10,10],[0,10]],"bev_points":[[0,0],[20,0],[20,20],[0,20]]})
    assert np.allclose(calibrate_homography(request,folder/"matrix.json")["homography"],[[2,0,0],[0,2,0],[0,0,1]])


def test_metric_segments_deduplicate_and_exclude_verified_dash_gaps(folder):
    from agent.segments import Segments
    _,p=installation_profile(folder)
    p["measurement"]["lane_tracks"][0].update(line_type="dashed",dash_length_m=1.,gap_length_m=1.,phase_m=0.,phase_verified=True)
    accumulator=Segments(uid(),p)
    paint=np.ones((48,64),np.uint8); valid=paint.copy()
    evidence={"frame_id":1}
    for distance in np.arange(0,9,.2):
        accumulator.add(paint,valid,float(distance),0,evidence,100.)
        accumulator.add(paint,valid,float(distance),0,evidence,100.)
    results=accumulator.finish()
    first=next(r for r in results if r["start_m"]==0)
    assert first["lane"]["grade"]=="visible_good"
    assert first["lane"]["retention_pct"]==100
    assert first["lane"]["segment_support_ratio"]==1
    assert len(first["lane"]["patch_ids"])==50


def test_motion_requires_gnss_visual_agreement_and_resets_gap():
    from agent.segments import Distance
    rng=np.random.default_rng(2026)
    gray=rng.integers(20,220,(64,64),dtype=np.uint8)
    shifted=np.roll(gray,2,axis=0)
    at=datetime.now(timezone.utc)
    motion=Distance(.05)
    assert motion.update(gray,at.isoformat(),3.6,True)[0] is None
    distance,reason=motion.update(shifted,(at+timedelta(seconds=.1)).isoformat(),3.6,True)
    assert reason is None and abs(distance-.1)<.01
    assert motion.update(shifted,(at+timedelta(seconds=.2)).isoformat(),3.6,False)[0] is None
    assert motion.epoch==1


@pytest.mark.parametrize('cpu_profile',['default','fast'])
def test_capture_run_analysis_review_outbox_contract(folder,cpu_profile):
    from agent.workflow import analyze_samples,run_samples,approve_review,sync_run,overlay_mask
    video=folder/"stereo.mp4"; synthetic_video(video,frames=6)
    key=folder/"key"; captured=record(video,folder/"runs",seconds=.6,chunk_seconds=1,minimum_free_bytes=0,key_path=key)
    captured.update(original_capture_verified=True,precheck={k:True for k in __import__("agent.equipment",fromlist=["CHECKS"]).CHECKS})
    manifest=folder/"run.json";atomic_json(manifest,captured)
    profile_path,p=installation_profile(folder)
    class ContractModel:
        def __init__(self,task): self.manifest={"task":task,"sha256":"a"*64}
        def predict(self,image):
            if self.manifest["task"]=="lane": return np.ones(image.shape[:2],np.uint8)
            if self.manifest["task"]=="redaction": return []
            return [{"class":"pothole","bbox_xywh_px":[10.,10.,8.,8.],"confidence":.9}]
    bundle={"version":"fixture-contract","config_hash":"b"*64,"weights_hashes":{k:"a"*64 for k in ("pothole","lane","redaction")},
            "pothole":"pothole","lane":"lane","redaction":"redaction"}
    result=analyze_samples(run_samples(manifest,key_path=key),folder/"analysis",run=captured,profile_path=profile_path,
                           device_id=uid(),bundle=bundle,model_factory=ContractModel,cpu_profile=cpu_profile)
    assert len(result["frames"])==6 and len(result["items"])==2
    assert result["items"][1]["observation"]["lane"]["estimated_length_m"] is None
    item=result["items"][0]; obs=item["observation"]
    assert obs["pothole"]["track_observations"]>=3 and obs["location"]["quality"]=="unavailable"
    assert obs["image"]["reviewed_for_sharing"] is False
    decisions=folder/"decisions.json";atomic_json(decisions,[{"observation_id":obs["observation_id"],"approve":True,"image_sha256":obs["image"]["sha256"]}])
    reviewed=approve_review(folder/"analysis"/"review.json",decisions,reviewed_by="contract-reviewer",queue_path=folder/"queue.sqlite",key_path=key)
    assert reviewed["queued"]==[obs["observation_id"]]
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(201 if request.method=="POST" else 200,json={"data":{}})
    sync_run(folder/"analysis"/"review.json","http://localhost:8000/api/v1","device",transport=httpx.MockTransport(handler))
    assert [r.method for r in requests]==["POST","PUT","PATCH"] and "if-match" in requests[2].headers
    assert json.loads(requests[1].content)["segments"]==[]
    assert len(list((folder/"analysis").glob("*.redacted.png")))==result["retained_photos"]
    assert result["removed_unreferenced_photos"]>0
    assert overlay_mask(np.array([[0,1],[1,0]],np.uint8))=={"size":[2,2],"counts":[1,2,1]}
    def ingest(request):
        if request.url.path.endswith("image-approvals"):
            return httpx.Response(201,json={"data":{"approval_id":uid()}})
        if request.url.path.endswith("images"):
            assert b"image/png" in request.content
            return httpx.Response(201,json={"data":{"file_id":obs["image"]["file_id"]}})
        from services.inspection.schemas import Observation
        accepted=Observation.model_validate(json.loads(request.content))
        assert accepted.run_id==captured["run_id"] and accepted.image.reviewed_for_sharing
        return httpx.Response(201,json={"data":{"observation_id":accepted.observation_id}})
    queue=Outbox(folder/"queue.sqlite",key)
    queue.drain("http://localhost:8000/api/v1",device_token="contract-device",reviewer_token="contract-reviewer",transport=httpx.MockTransport(ingest))
    assert queue.list()[0]["state"]=="acked"


def test_optional_stereo_summary_and_object_coordinate_never_invent_depth():
    from agent.equipment import depth_summary,object_coordinate
    points=np.full((10,10,3),np.nan)
    assert depth_summary(points,[0,0,10,10])["status"]=="unavailable"
    points[:]=[0,0,5]
    assert depth_summary(points,[0,0,10,10])["camera_point_m"]==[0.,0.,5.]
    location={"quality":"valid","lat":37.,"lon":127.}
    estimate=object_coordinate(location,[0,10,0],np.eye(4).tolist(),0.)
    assert estimate["lat"]>37 and estimate["accuracy_m"] is None
    assert object_coordinate({"quality":"unavailable"},[0,0,5],np.eye(4).tolist(),None)["status"]=="unavailable"


def test_actual_synthetic_stereo_disparity_and_metric_depth():
    from agent.equipment import stereo_depth
    rng=np.random.default_rng(96)
    left=rng.integers(20,230,(96,192,3),dtype=np.uint8)
    right=np.roll(left,-8,axis=1)
    matrix=[[100.,0,96],[0,100.,48],[0,0,1]]
    calibration={"left_camera_matrix":matrix,"right_camera_matrix":matrix,"rotation":np.eye(3).tolist(),
                 "translation_m":[-.12,0,0],"left_distortion":[0.]*5,"right_distortion":[0.]*5}
    result=stereo_depth(left,right,calibration)
    assert result["valid"].sum()>1000
    assert abs(float(np.nanmedian(result["depth_m"]))-1.5)<.1


def test_independent_metrics_count_duplicates_and_unassessable_fn(folder):
    from agent.evaluate import detection_metrics,lane_metrics,segmentation_iou,evaluate
    gt=[{"sample_id":"a","class":"pothole","bbox_xywh_px":[0,0,10,10]}]
    predictions=[gt[0]|{"confidence":.9},gt[0]|{"confidence":.8}]
    result=detection_metrics(gt,predictions,["pothole"])
    assert result["map50"]==1 and result["classes"]["pothole"]["fp"]==1
    lanes=lane_metrics([{"segment_id":"one","retention_pct":30.},{"segment_id":"two","retention_pct":90.}],
                       [{"segment_id":"one","retention_pct":None},{"segment_id":"two","retention_pct":80.}])
    assert lanes["fn"]==1 and lanes["assessable_ratio"]==.5 and lanes["retention_mae_pp"]==10.
    assert segmentation_iou([[1,0,255]],[[1,1,1]])["paint_iou"]==.5
    request=folder/"metrics.json"
    atomic_json(request,{"independent_locked_truth":True,"synthetic":True,"detection":{"truth":gt,"predictions":predictions,"classes":["pothole"]}})
    assert evaluate(request,folder/"metrics-result.json")["acceptance_approved"] is False


def test_detector_coco_export_uses_only_reviewed_train_validation(folder):
    from agent.train_detector import export_coco
    rows=[]
    for number,split in enumerate(("train","validation","test")):
        path=folder/f"image-{number}.png";cv2.imwrite(str(path),np.full((20,30,3),number*50,np.uint8))
        rows.append({"sample_id":uid(),"source_id":uid(),"source_hash":sha256(path),"leakage_group_id":uid(),"annotation_version":"1.0",
                     "path":str(path),"sha256":sha256(path),"privacy_status":"approved","qc_status":"qc_passed","allowed_use":"synthetic-test",
                     "expires_at":(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),"split":split,"detection_labels_verified":True,
                     "objects":[{"class":"pothole","certainty":"confirmed","bbox_xywh_px":[1,2,3,4]}]})
    release=folder/"release.json";atomic_json(release,rows)
    output=export_coco(release,folder/"coco","pothole")
    train=json.loads((output/"annotations/instances_train2017.json").read_text())
    assert len(train["annotations"])==1 and not (output/"test2017").exists()
    rows[0]["objects"][0]["certainty"]="uncertain";atomic_json(release,rows)
    with pytest.raises(ValueError,match="uncertain"):
        export_coco(release,folder/"rejected-coco","pothole")


def test_lane_visible_color_alignment_and_surrounding_plane(folder):
    from agent.profile import lane_evidence
    from agent.equipment import road_relative_step
    _,profile=installation_profile(folder)
    image=np.full((48,64,3),100,np.uint8);paint=np.zeros((48,64),np.uint8)
    image[:,20:30]=220;paint[:,20:30]=1
    valid,status=lane_evidence(image,paint,np.ones_like(paint),profile)
    key=profile["measurement"]["lane_tracks"][0]["lane_track_id"]
    assert status[key]["color"]=="white" and valid[:,20:30].all()
    image[:,20:30]=[0,220,220]
    valid,status=lane_evidence(image,paint,np.ones_like(paint),profile)
    assert status[key]["color"]=="yellow" and not valid[:,20:30].any()
    y,x=np.mgrid[:40,:40]
    points=np.stack((x*.01,y*.01,np.ones_like(x)),axis=-1).astype(float)
    points[15:25,15:25,2]+=.05
    result=road_relative_step(points,[15,15,10,10])
    assert result["physical_severity"] is None and abs(result["offset_m"]-.05)<1e-6


def test_workflow_identity_rejection_precedes_model_execution(folder):
    from agent.workflow import execute_config
    from agent.equipment import CHECKS
    profile_path,p=installation_profile(folder)
    run_id,device_id,bundle_id=uid(),uid(),uid()
    manifest=folder/"manifest.json";bundle=folder/"bundle.json"
    atomic_json(manifest,{"run_id":run_id,"device_id":device_id,"profile_id":p["profile_id"],"precheck":{k:True for k in CHECKS}})
    atomic_json(bundle,{"model_bundle_id":bundle_id})
    config={"run_id":uid(),"device_id":device_id,"profile_id":p["profile_id"],"model_bundle_id":bundle_id,
            "run_manifest":str(manifest),"profile_path":str(profile_path),"bundle_path":str(bundle),"output":str(folder/"result")}
    path=folder/"config.json";atomic_json(path,config)
    with pytest.raises(ValueError,match="identity_mismatch"):execute_config(path)
    assert not (folder/"result").exists()


def test_local_retention_removes_only_expired_known_artifacts(folder):
    from agent.retention import purge_local_material
    work=folder/"work";temporary=work/uid();temporary.mkdir(parents=True)
    atomic_json(temporary/"retention.json",{"expires_at":(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()})
    (temporary/"chunk.mp4").write_bytes(b"test-raw")
    results=folder/"results";job=results/uid();job.mkdir(parents=True)
    atomic_json(job/"review.json",{"run":{"started_at":(datetime.now(timezone.utc)-timedelta(days=91)).isoformat()}})
    (job/"000.redacted.png").write_bytes(b"test-derived");(job/"unrelated.txt").write_text("preserve")
    assert len(purge_local_material(work=work,results=results))==2 and temporary.exists()
    purge_local_material(work=work,results=results,dry_run=False)
    assert not temporary.exists() and not (job/"000.redacted.png").exists() and (job/"unrelated.txt").exists()


def test_capture_and_run_replay_never_write_plaintext_mp4_or_index(folder,monkeypatch):
    from agent.capture import replay_bytes
    from agent.workflow import run_samples
    source=synthetic_video(folder/"source.mp4",frames=12)
    original_open=Path.open
    def guarded_open(path,mode="r",*args,**kwargs):
        if any(flag in mode for flag in ("w","a","x")) and (path.suffix==".mp4" or path.name.endswith(".pts.json")):
            raise AssertionError("plaintext_raw_disk_write")
        return original_open(path,mode,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded_open)
    monkeypatch.setattr(cv2,"VideoWriter",lambda *a,**k: (_ for _ in ()).throw(AssertionError("disk_encoder_forbidden")))
    key=folder/"key"
    result=record(source,folder/"runs",seconds=1.2,chunk_seconds=1,minimum_free_bytes=0,key_path=key)
    store=EncryptedStore(key)
    item=result["chunks"][0]["eyes"][0]
    frames=list(replay_bytes(store.decrypt_bytes(item["path"]),sample_fps=5))
    assert len(frames)==5 and frames[0]["frame"].shape==(48,64,3)
    manifest=folder/"runs"/result["run_id"]/"manifest.json"
    monkeypatch.setattr(store.__class__,"decrypt",lambda *a,**k: (_ for _ in ()).throw(AssertionError("disk_decryption_forbidden")))
    assert len(list(run_samples(manifest,key_path=key)))==12
    assert not list(manifest.parent.glob("*.mp4")) and not list(manifest.parent.glob("*.pts.json"))


def test_collection_plan_and_stationary_gate(folder):
    from agent.equipment import collection_plan
    value={"plan_id":uid(),"organization":"fixture organization","controller_contact":"fixture@example.invalid",
           "operator":"fixture operator","legal_basis":"fixture declared authorization; not legal guidance",
           "purpose":"synthetic contract testing","collection_scope":"consented test fixture only",
           "notice_message":"fixture recording notice and contact","raw_retention_days":7,"masked_retention_days":90,"audit_retention_days":180}
    path=folder/"plan.json";atomic_json(path,value)
    assert collection_plan(path)["sha256"]==sha256(path)
    with pytest.raises(ValueError,match="stationary"):
        record(0,folder/"runs",minimum_free_bytes=0,collection_plan_path=path)
    value["raw_retention_days"]=30;atomic_json(path,value)
    with pytest.raises(ValueError,match="retention"):collection_plan(path)


def test_project_ca_keeps_certificate_and_hostname_verification(folder,monkeypatch):
    import ssl
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes,serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from agent.tls import gateway_tls
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,"vision-test-ca")])
    now=datetime.now(timezone.utc)
    certificate=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                 .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(days=1))
                 .add_extension(x509.BasicConstraints(ca=True,path_length=None),critical=True).sign(key,hashes.SHA256()))
    path=folder/"ca.pem";path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    monkeypatch.setenv("ROAD_CA_BUNDLE",str(path))
    context=gateway_tls()
    assert context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED and context.cert_store_stats()["x509_ca"]==1
    with pytest.raises(ValueError,match="outside_workspace"):gateway_tls(ROOT.parent/"outside.pem")


@pytest.mark.parametrize("operator_stop",[True,False])
def test_camera_operator_stop_and_disconnect_preserve_encrypted_metadata(folder,monkeypatch,operator_stop):
    import serial
    from agent.equipment import probe_gnss
    class FakeSerial:
        in_waiting=0
        def __init__(self,port,baudrate,timeout):assert baudrate==9600
        def close(self):pass
        def readline(self,*args):return b""
        def __enter__(self):return self
        def __exit__(self,*args):self.close()
    monkeypatch.setattr(serial,"Serial",FakeSerial)
    assert probe_gnss("FIXTURE",seconds=.001,baudrate=9600)==[]
    plan=folder/"plan.json"
    atomic_json(plan,{"plan_id":uid(),"organization":"fixture organization","controller_contact":"fixture@example.invalid",
        "operator":"fixture operator","legal_basis":"explicit synthetic fixture authorization","purpose":"contract test only",
        "collection_scope":"generated frames only","notice_message":"fixture collection notice","raw_retention_days":7,
        "masked_retention_days":90,"audit_retention_days":180})
    class FakeCamera:
        def __init__(self):self.count=0;self.closed=False
        def isOpened(self):return True
        def set(self,*args):return True
        def get(self,*args):return 10.
        def read(self):
            self.count+=1
            if self.count==1:return True,np.full((48,128,3),100,np.uint8)
            if operator_stop:raise KeyboardInterrupt()
            return False,None
        def release(self):self.closed=True
    camera=FakeCamera();monkeypatch.setattr(cv2,"VideoCapture",lambda *args:camera)
    metadata={"device_id":uid(),"profile_id":uid(),"precheck":{"notice_present":True},"conditions":{"daylight":True,"dry_asphalt":True}}
    options=dict(seconds=60,minimum_free_bytes=0,key_path=folder/"key",collection_plan_path=plan,confirmed_stationary=True,capture_metadata=metadata,
                 serial_port="FIXTURE",baudrate=9600)
    if operator_stop:
        result=record(0,folder/"runs",**options)
        assert result["status"]=="completed" and result["stop_reason"]=="operator"
    else:
        with pytest.raises(RuntimeError,match="disconnected"):record(0,folder/"runs",**options)
        result=json.loads(next((folder/"runs").glob("*/manifest.json")).read_text())
        assert result["status"]=="interrupted"
    assert camera.closed and result["device_id"]==metadata["device_id"] and result["profile_id"]==metadata["profile_id"]
    assert result["persisted_frames"]==1 and result["chunks"][0]["index"]["encrypted"]
    assert result["collection_plan"]["sha256"]==sha256(plan)
    assert result["gnss_baudrate"]==9600


def training_release_fixture(folder):
    rows=[]
    now=datetime.now(timezone.utc)
    for number,split in enumerate(("train","validation")):
        image=folder/f"source-{number}.png";cv2.imwrite(str(image),np.full((20,30,3),number*70,np.uint8))
        rows.append({"sample_id":uid(),"source_id":uid(),"source_hash":sha256(image),"leakage_group_id":uid(),"annotation_version":"1.0",
            "path":str(image),"sha256":sha256(image),"privacy_status":"approved","qc_status":"qc_passed","allowed_use":"synthetic-contract",
            "expires_at":(now+timedelta(days=10)).isoformat(),"source_expires_at":(now+timedelta(days=1 if number==0 else 10)).isoformat(),
            "split":split,"detection_labels_verified":True,"objects":[{"class":"pothole","certainty":"confirmed","bbox_xywh_px":[1,2,3,4]}]})
    release=folder/"release.json";atomic_json(release,rows)
    return release,rows


def test_training_expiry_removes_copies_labels_and_blocks_model(folder,monkeypatch):
    from agent.train_detector import export_coco
    from agent.governance import link_model,revoke_known
    release,rows=training_release_fixture(folder);original_hash=sha256(release)
    output=export_coco(release,folder/"training"/"run","pothole")
    provenance=json.loads((output/"provenance.json").read_text())
    assert provenance["assets"][0]["expires_at"]==rows[0]["source_expires_at"]
    assert provenance["assets"][0]["files"][0]["sha256"]==rows[0]["sha256"]
    weights=output/"model.onnx";weights.write_bytes(b"mocked-loader-test-only")
    manifest=output/"model.json"
    atomic_json(manifest,{"format":"onnx","task":"pothole","classes":["pothole"],"weights":str(weights),"sha256":sha256(weights),
        "target_trained":True,"dataset_release":"fixture","input_size":[32,32],"output_layout":"yolox_decoded_xywh",
        "source_release_path":str(release),"training_provenance":str(output/"provenance.json")})
    link_model(output/"provenance.json",manifest)
    from agent import models as model_module
    monkeypatch.setattr(model_module,"load_network",lambda *args,**kwargs:object())
    assert readiness(manifest)["ready"]
    now=datetime.now(timezone.utc)+timedelta(days=2)
    before=readiness(manifest)
    result=revoke_known(now=now,dry_run=False,training_root=folder/"training",results_root=folder/"results")
    assert result["affected"][0]["assets"]==1
    assert not workspace_path(provenance["assets"][0]["files"][0]["path"]).exists()
    assert workspace_path(provenance["assets"][1]["files"][0]["path"]).exists()
    train=json.loads((output/"annotations/instances_train2017.json").read_text())
    assert train["images"]==[] and train["annotations"]==[]
    assert "revoked" in readiness(manifest)["reason"]
    assert sha256(release)==original_hash and workspace_path(rows[0]["path"]).exists()
    from agent import governance
    log=(governance.REGISTRY.parent/"revocation-audit.jsonl").read_text()
    assert rows[0]["sample_id"] not in log and rows[0]["path"] not in log
    with pytest.raises(ValueError,match="revoked"):
        export_coco(release,folder/"new-training","pothole")


def test_explicit_source_and_observation_revoke_scoped_copies_and_queue(folder):
    from agent.train_detector import export_coco
    from agent.governance import revoke_known,identity_hash
    release,rows=training_release_fixture(folder)
    output=export_coco(release,folder/"training"/"run","pothole")
    observation,image=fixture_observation(folder)
    queue=Outbox(folder/"queue.sqlite",folder/"key")
    queue.enqueue(observation,image,reviewed_by="reviewer",confirmed_human_review=True,lineage={"source_id":rows[1]["source_id"]})
    report=revoke_known(request={"source_ids":[rows[1]["source_id"]],"reason":"fixture consent withdrawal"},
                        dry_run=False,training_root=folder/"training",results_root=folder/"results")
    assert report["affected"][0]["assets"]==1
    queue.revoke_observations({identity_hash("source_id",rows[1]["source_id"])},dry_run=False)
    assert queue.list()==[] and workspace_path(rows[1]["path"]).exists()
    with pytest.raises(ValueError,match="revoked"):
        queue.enqueue(observation,image,reviewed_by="reviewer",confirmed_human_review=True,lineage={"source_id":rows[1]["source_id"]})
    # Tampered provenance cannot authorize deletion of an unrelated project file.
    unrelated=folder/"unrelated.png";unrelated.write_bytes(b"preserve")
    provenance=json.loads((output/"provenance.json").read_text())
    provenance["assets"][0]["files"][0]["path"]=str(unrelated);atomic_json(output/"provenance.json",provenance)
    with pytest.raises(ValueError,match="unowned_training"):
        revoke_known(request={"sample_ids":[rows[0]["sample_id"]],"reason":"fixture"},dry_run=False,training_root=folder/"training",results_root=folder/"results")
    assert unrelated.exists()


def test_lane_staged_masks_follow_source_ttl_and_observation_revoke_clears_shared_review(folder):
    from agent.train_lane import stage_lane_pairs
    from agent.governance import revoke_known
    release,rows=training_release_fixture(folder)
    for index,row in enumerate(rows):
        mask=folder/f"source-mask-{index}.png";cv2.imwrite(str(mask),np.ones((20,30),np.uint8))
        row.update(mask_path=str(mask),mask_sha256=sha256(mask))
    atomic_json(release,rows)
    copied,provenance=stage_lane_pairs(folder/"training"/"lane",release,rows)
    revoke_known(now=datetime.now(timezone.utc)+timedelta(days=2),dry_run=False,training_root=folder/"training",results_root=folder/"results")
    assert not workspace_path(copied[0]["mask_path"]).exists() and not workspace_path(copied[0]["path"]).exists()
    assert workspace_path(copied[1]["mask_path"]).exists() and workspace_path(rows[0]["mask_path"]).exists()
    job=folder/"results"/uid();job.mkdir(parents=True)
    image=job/"000.redacted.png";cv2.imwrite(str(image),np.zeros((20,30,3),np.uint8))
    first,second=uid(),uid()
    atomic_json(job/"review.json",{"items":[{"observation":{"observation_id":identifier},"image_path":str(image)} for identifier in (first,second)],
                "frames":[{"frame_id":0}],"lane_segments":[{"private_evidence":"remove"}]})
    unrelated=job/"unrelated.txt";unrelated.write_text("preserve")
    revoke_known(request={"observation_ids":[first],"reason":"fixture deletion request"},dry_run=False,
                 training_root=folder/"training",results_root=folder/"results")
    value=json.loads((job/"review.json").read_text())
    assert value["items"]==[] and value["frames"]==[] and value["lane_segments"]==[]
    assert not image.exists() and unrelated.exists()


def test_path_sync_preserves_gaps_scopes_limits_and_capture_registration(folder):
    from agent.workflow import derive_path,sync_run,register_capture
    from agent.equipment import CHECKS
    from services.contracts import RunPath
    now=datetime.now(timezone.utc)-timedelta(hours=1)
    profile_id,device_id,run_id=uid(),uid(),uid()
    def frame(seconds,quality="valid",usable=True,lat=37.):
        return {"captured_at":(now+timedelta(seconds=seconds)).isoformat(),"location":{"lat":lat,"lon":127. if lat is not None else None,"quality":quality,"age_ms":0},
                "quality":{"usable":usable,"reasons":[] if usable else ["blur_or_low_texture"]},
                "lane_evidence":{"lane":{"outline_supported":usable,"reason":None if usable else "outline_not_supported"}}}
    frames=[frame(0),frame(.1),frame(.2,"unavailable",lat=None),frame(.3),frame(.4),
            frame(3),frame(3.1),frame(3.2,"degraded",False),frame(3.3,"degraded",False)]
    run={"run_id":run_id,"device_id":device_id,"profile_id":profile_id,"started_at":now.isoformat(),
         "ended_at":(now+timedelta(seconds=4)).isoformat(),"status":"completed","precheck":{k:True for k in CHECKS},
         "postcheck":{"mount_secure":True,"lens_clear":True,"defect_from":None}}
    package={"run":run,"device_id":device_id,"profile_id":profile_id,"frames":frames}
    path=derive_path(package)
    assert len(path["segments"])==4
    assert path["segments"][-1]["quality"]=="degraded" and path["segments"][-1]["visible_scope"]["assessment_state"]=="unassessable"
    assert path["segments"][0]["to"]==frames[1]["captured_at"] and path["segments"][1]["from"]==frames[3]["captured_at"]
    long_package=package|{"frames":[frame(i*.01) for i in range(3000)]}
    reduced=derive_path(long_package)
    assert len(reduced["segments"][0]["coordinates"])==1000 and reduced["removed_interior_points"]==2000
    RunPath.model_validate({"segments":reduced["segments"],"manifest_hash":"a"*64})
    many=[]
    for i in range(600):many.extend([frame(i*3),frame(i*3+.1)])
    bounded=derive_path(package|{"frames":many})
    assert len(bounded["segments"])==500 and sum(len(s["coordinates"]) for s in bounded["segments"])==1000 and bounded["omitted_segments"]==100
    assert bounded["segments"][0]["from"]==many[0]["captured_at"] and bounded["segments"][-1]["to"]==many[-1]["captured_at"]
    manifest=folder/"finished.json";atomic_json(manifest,run)
    package_path=folder/"review.json";atomic_json(package_path,package)
    requests=[]
    def handler(request):
        requests.append(request)
        if request.method=="PUT":RunPath.model_validate(json.loads(request.content))
        return httpx.Response(200,json={"data":{}})
    register_capture(manifest,"http://localhost:8000/api/v1","device",transport=httpx.MockTransport(handler))
    synced=sync_run(package_path,"http://localhost:8000/api/v1","device",transport=httpx.MockTransport(handler))
    assert [r.method for r in requests]==["POST","POST","PUT","PATCH"]
    assert json.loads(requests[0].content)==json.loads(requests[1].content)
    assert json.loads(requests[2].content)["manifest_hash"]==json.loads(requests[3].content)["manifest_hash"]
    assert synced["path_segments"]==4


def test_research_detector_export_does_not_impersonate_deployment_approval(folder):
    from agent.train_detector import export_coco
    release,rows=training_release_fixture(folder)
    evidence=folder/"license.json"
    atomic_json(evidence,{"research_allowed":True,"license_id":"CC-BY-4.0","source_url":"https://publisher.invalid/dataset",
                          "license_url":"https://creativecommons.org/licenses/by/4.0/"})
    for row in rows:
        row.update(privacy_status="restricted_research",qc_status="publisher_annotations_machine_checked",release_scope="offline_research",
                   human_privacy_reviewed=False,not_for_deployment=True,license_evidence_path=str(evidence),license_evidence_sha256=sha256(evidence))
    atomic_json(release,rows)
    with pytest.raises(ValueError,match="approval_or_qc_missing"):
        export_coco(release,folder/"deployment","pothole")
    output=export_coco(release,folder/"research","pothole",research=True)
    provenance=json.loads((output/"provenance.json").read_text())
    assert provenance["research_only"] is True and provenance["production_approved"] is False
    assert json.loads(release.read_text())[0]["human_privacy_reviewed"] is False
    atomic_json(release.with_suffix(".quarantine.json"),{"reason":"semantic class mapping rejected in source QC"})
    with pytest.raises(ValueError,match="quarantined"):
        export_coco(release,folder/"quarantined-export","pothole",research=True)
