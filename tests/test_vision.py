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
def folder():
    path = ROOT / ".cache" / "vision-tests" / str(uuid.uuid4())
    path.mkdir(parents=True)
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
