"""Real HTTP contract/error tests. Start project services before running.

All image evidence generated here is SYNTHETIC: these tests do not measure AI
accuracy, real camera performance or privacy detector recall.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import httpx
from PIL import Image, ImageDraw
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.http


def utc(seconds=0):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def env_values():
    values = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k] = v.strip().strip('"').strip("'")
    return values


def data(response, expected=200):
    assert response.status_code == expected, (response.status_code, response.text[:1500])
    body = response.json()
    assert "data" in body and "meta" in body, body
    return body["data"]


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=os.getenv("ROAD_TEST_BASE_URL", "http://127.0.0.1:8000"), timeout=30) as c:
        health = c.get("/health")
        assert health.status_code == 200, "Gateway must be started before HTTP tests"
        yield c


@pytest.fixture(scope="session")
def auth(client):
    login = data(client.post("/api/v1/auth/login", json={"org_code": "demo", "login": "admin@road.local", "password": env_values()["ROAD_BOOTSTRAP_PASSWORD"]}))
    return {"Authorization": "Bearer " + login["access_token"]}


@pytest.fixture(scope="session")
def setup(client, auth):
    suffix = uuid4().hex[:10]
    device = data(client.post("/api/v1/devices", headers=auth, json={"name": "HTTP synthetic " + suffix}), 201)
    device_id = device.get("device_id", device.get("id"))
    dh = {"Authorization": "Bearer " + device["device_token"]}
    profile_body = {"name": "HTTP synthetic " + suffix, "version": 1,
                    "camera_serial": "SYNTHETIC-NOT-A-CAMERA", "lens_label": "synthetic test",
                    "calibration": {"image_size": [1280, 720], "height_m": 0.7, "pitch_deg": 20,
                                    "camera_matrix": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
                                    "distortion": [0, 0, 0, 0, 0], "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                    "roi_polygon_px": [[0, 0], [1279, 0], [1279, 719], [0, 719]],
                                    "visible_distance_m": [1.5, 6], "calibrated_at": utc(-120)},
                    "config_hash": "a" * 64}
    profile = data(client.post("/api/v1/profiles", headers=auth, json=profile_body), 201)
    profile_id = profile.get("profile_id", profile.get("id"))
    bundle_body = {"version": "http-" + suffix, "weights_hashes": {"pothole": "b"*64, "lane": "c"*64, "redaction": "d"*64},
                   "preprocess": {"pothole": {"input_size": [416, 416], "resize": "letterbox"}, "lane": {"long_edge": 512}, "redaction": {"input_size": [416, 416]}},
                   "thresholds": {"pothole_confidence": 0.35, "pothole_nms_iou": 0.5, "single_frame_confidence": 0.8, "lane_monitor": 60, "lane_good": 80, "min_valid_ratio": 0.7, "min_segment_support_ratio": 0.8},
                   "config_hash": "e"*64, "code_commit": "synthetic-contract-test", "environment_hash": "f"*64}
    bundle = data(client.post("/api/v1/model-bundles", headers=auth, json=bundle_body), 201)
    run_id = str(uuid4())
    manifest_hash = "1"*64
    run_body = {"run_id": run_id, "device_id": device_id, "profile_id": profile_id, "started_at": utc(-60), "manifest_hash": manifest_hash,
                "precheck": {k: True for k in ["camera_preview_ok", "mount_secure", "lens_clear", "profile_valid", "gnss_device_connected", "storage_ge_10gb", "notice_present"]}}
    data(client.post("/api/v1/runs", headers=dh, json=run_body), 201)
    image = Image.new("RGB", (1280, 720), (45, 54, 60))
    draw = ImageDraw.Draw(image)
    draw.text((30, 30), "SYNTHETIC HTTP TEST - NOT AI EVIDENCE", fill="white")
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    blob = stream.getvalue()
    sha = hashlib.sha256(blob).hexdigest()
    file_id, captured = str(uuid4()), utc(-10)
    approval_body = {"file_id": file_id, "device_id": device_id, "sha256": sha, "size_bytes": len(blob), "mime": "image/jpeg", "captured_at": captured, "reviewed_at": utc(-5), "redaction_model_version": "SYNTHETIC-TEST-NO-PEOPLE"}
    approval = data(client.post("/api/v1/image-approvals", headers=auth, json=approval_body), 201)
    metadata = {"file_id": file_id, "approval_id": approval.get("approval_id", approval.get("id")), "sha256": sha, "captured_at": captured}
    image_result = data(client.post("/api/v1/images", headers=dh, data={"metadata": json.dumps(metadata)}, files={"file": ("synthetic.jpg", blob, "image/jpeg")}), 201)
    observation = {"schema_version": "0.4", "observation_id": str(uuid4()), "event_id": str(uuid4()), "run_id": run_id, "device_id": device_id, "type": "pothole", "captured_at": captured, "frame_id": 1234,
                   "location": {"lat": 37.5, "lon": 127.0, "source": "vehicle_observation", "quality": "valid", "hdop": 1.2, "age_ms": 100, "accuracy_m": None},
                   "profile_id": profile_id, "model_version": bundle_body["version"], "config_hash": bundle_body["config_hash"],
                   "pothole": {"bbox_xywh_px": [100, 200, 160, 80], "confidence": 0.91, "track_observations": 5, "candidate_state": "multi_frame", "occluded": False}, "lane": None,
                   "quality": {"usable": True, "reasons": []}, "image": {"file_id": file_id, "sha256": sha, "redacted": True, "reviewed_for_sharing": True}, "review": {"status": "unreviewed", "priority": "normal"}}
    return {"device": device, "dh": dh, "run": run_body, "profile": profile_body, "profile_id": profile_id, "bundle": bundle_body, "observation": observation, "blob": blob, "metadata": metadata, "image": image_result, "approval_body": approval_body}


def test_authentication_and_no_header_spoofing(client):
    assert client.get("/api/v1/observations").status_code == 401
    assert client.get("/api/v1/observations", headers={"Authorization": "Bearer invalid", "X-Road-Principal": '{"role":"admin"}'}).status_code == 401
    bad = client.post("/api/v1/auth/login", json={"org_code": "demo", "login": "admin@road.local", "password": "wrong-password"})
    assert bad.status_code == 401
    assert "password" not in bad.text.lower() or "wrong-password" not in bad.text


def test_configuration_and_actual_image_upload(client, auth, setup):
    profile = data(client.get("/api/v1/profiles/" + setup["profile_id"], headers=auth))
    assert profile.get("name") == setup["profile"]["name"]
    file_id = setup["observation"]["image"]["file_id"]
    photo = client.get(f"/api/v1/images/{file_id}/content", headers=auth)
    assert photo.status_code == 200
    assert hashlib.sha256(photo.content).hexdigest() == setup["observation"]["image"]["sha256"]
    assert "no-store" in photo.headers.get("cache-control", "")


def test_observation_ingestion_retry_and_conflict(client, auth, setup):
    payload = setup["observation"]
    original = data(client.post("/api/v1/observations", headers=setup["dh"], json=payload), 201)
    for _ in range(3):
        replay = data(client.post("/api/v1/observations", headers=setup["dh"], json=dict(reversed(list(payload.items())))))
        assert replay["observation_id"] == original["observation_id"]
        assert replay["deduplicated"] is True
    changed = json.loads(json.dumps(payload))
    changed["pothole"]["confidence"] = 0.93
    assert client.post("/api/v1/observations", headers=setup["dh"], json=changed).status_code == 409
    view = data(client.get("/api/v1/observations/" + payload["observation_id"], headers=auth))
    assert view["pothole"]["confidence"] == 0.91


@pytest.mark.parametrize("change", ["bad_uuid", "half_coordinate", "opposite_type", "out_of_bounds", "unauthorized_review"])
def test_observation_validation_errors(client, setup, change):
    value = json.loads(json.dumps(setup["observation"]))
    value["observation_id"], value["event_id"] = str(uuid4()), str(uuid4())
    if change == "bad_uuid": value["observation_id"] = "not-a-uuid"
    if change == "half_coordinate": value["location"]["lat"] = None
    if change == "opposite_type": value["lane"] = {"grade": "monitor"}
    if change == "out_of_bounds": value["pothole"]["bbox_xywh_px"] = [1270, 200, 200, 80]
    if change == "unauthorized_review": value["review"]["status"] = "completed"
    response = client.post("/api/v1/observations", headers=setup["dh"], json=value)
    assert response.status_code == 422, response.text


def test_image_approval_and_content_mismatch(client, auth, setup):
    assert client.post("/api/v1/image-approvals", headers=setup["dh"], json=setup["approval_body"]).status_code == 403
    metadata = dict(setup["metadata"], file_id=str(uuid4()), sha256="0"*64)
    response = client.post("/api/v1/images", headers=setup["dh"], data={"metadata": json.dumps(metadata)}, files={"file": ("forged.jpg", setup["blob"], "image/jpeg")})
    assert response.status_code in (409, 422), response.text
    assert client.post("/api/v1/images", headers=setup["dh"], data={"metadata": json.dumps(metadata)}, files={"file": ("bad.txt", b"not an image", "text/plain")}).status_code in (415, 422)


def test_review_optimistic_locking_and_device_restriction(client, auth, setup):
    value = json.loads(json.dumps(setup["observation"]))
    value["observation_id"], value["event_id"] = str(uuid4()), str(uuid4())
    created = data(client.post("/api/v1/observations", headers=setup["dh"], json=value), 201)
    path = "/api/v1/observations/" + value["observation_id"] + "/review"
    body = {"expected_case_id": created["case_id"], "expected_version": created["case_version"], "status": "confirmed", "reason": "Synthetic HTTP state test"}
    assert client.patch(path, headers=setup["dh"], json=body).status_code == 403
    first = data(client.patch(path, headers=auth, json=body))
    assert first["version"] == created["case_version"] + 1
    assert client.patch(path, headers=auth, json=body).status_code == 409
    missing = dict(body)
    missing.pop("expected_version")
    assert client.patch(path, headers=auth, json=missing).status_code == 428


@pytest.mark.parametrize("retention,grade", [(59.999, "worn_candidate"), (60, "monitor"), (79.999, "monitor"), (80, "visible_good")])
def test_lane_boundary_values(client, setup, retention, grade):
    value = json.loads(json.dumps(setup["observation"]))
    value["observation_id"], value["event_id"] = str(uuid4()), str(uuid4())
    value["type"], value["pothole"] = "lane_segment", None
    value["lane"] = {"segment_id": str(uuid4()), "estimated_length_m": 10, "retention_pct": retention, "grade": grade, "color": "white", "line_type": "dashed", "valid_area_ratio": 0.7, "segment_support_ratio": 0.8, "outline_supported": True, "patch_ids": [str(uuid4())], "unassessable_reason": None}
    response = client.post("/api/v1/observations", headers=setup["dh"], json=value)
    data(response, 201)
    value["observation_id"], value["event_id"] = str(uuid4()), str(uuid4())
    value["lane"]["grade"] = "worn_candidate" if grade != "worn_candidate" else "visible_good"
    assert client.post("/api/v1/observations", headers=setup["dh"], json=value).status_code == 422


def test_unassessable_lane_nulls(client, auth, setup):
    value = json.loads(json.dumps(setup["observation"]))
    value["observation_id"], value["event_id"] = str(uuid4()), str(uuid4())
    value["type"], value["pothole"] = "lane_segment", None
    value["lane"] = {"segment_id": str(uuid4()), "estimated_length_m": None, "retention_pct": None, "grade": "unassessable", "color": "unknown", "line_type": "unknown", "valid_area_ratio": None, "segment_support_ratio": None, "outline_supported": False, "patch_ids": [], "unassessable_reason": "missing_outline_and_distance_basis"}
    data(client.post("/api/v1/observations", headers=setup["dh"], json=value), 201)
    saved = data(client.get("/api/v1/observations/" + value["observation_id"], headers=auth))
    assert saved["lane"]["retention_pct"] is None


def test_filter_errors_and_roles(client, auth, setup):
    assert client.get("/api/v1/observations?bbox=128,38,127,37", headers=auth).status_code == 422
    assert client.get("/api/v1/observations?bbox=126,36,128,38&quality=unavailable", headers=auth).status_code == 422
    assert client.get("/api/v1/observations?type=pothole&grade=monitor", headers=auth).status_code == 422
    assert client.get("/api/v1/observations/" + str(uuid4()), headers=auth).status_code == 404
    listing = data(client.get("/api/v1/observations?limit=2", headers=auth))
    assert len(listing["items"]) <= 2
    assert "counts_by_type" in listing


def test_report_job_and_download(client, auth):
    response = client.get("/api/v1/reports?format=csv", headers=auth)
    result = data(response, response.status_code)
    assert response.status_code in (200, 202)
    job_id = result.get("job_id", result.get("id"))
    assert job_id
    for _ in range(40):
        job = data(client.get("/api/v1/jobs/" + job_id, headers=auth))
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.25)
    assert job["status"] == "succeeded", job
    download = client.get("/api/v1/reports/" + job_id + "/download", headers=auth)
    assert download.status_code == 200, download.text[:500]
    assert "csv" in download.headers.get("content-type", "")
    assert len(download.content) > 10


def test_logout_revokes_session(client):
    login = data(client.post("/api/v1/auth/login", json={"org_code": "demo", "login": "admin@road.local", "password": env_values()["ROAD_BOOTSTRAP_PASSWORD"]}))
    headers = {"Authorization": "Bearer " + login["access_token"]}
    assert client.post("/api/v1/auth/logout", headers=headers).status_code == 204
    assert client.get("/api/v1/me", headers=headers).status_code == 401


def test_viewer_permissions_and_account_revocation(client, auth):
    login_name = "viewer-" + uuid4().hex[:10] + "@test.local"
    password = "Synthetic-Test-Password-2026"
    user = data(client.post("/api/v1/users", headers=auth, json={"login": login_name, "password": password, "role": "viewer"}), 201)
    token = data(client.post("/api/v1/auth/login", json={"org_code": "demo", "login": login_name, "password": password}))["access_token"]
    vh = {"Authorization": "Bearer " + token}
    assert client.get("/api/v1/observations", headers=vh).status_code == 200
    assert client.post("/api/v1/devices", headers=vh, json={"name": "not-allowed"}).status_code == 403
    assert client.post("/api/v1/users", headers=vh, json={"login": "not-allowed", "password": password, "role": "admin"}).status_code == 403
    data(client.patch("/api/v1/users/" + user["id"], headers=auth, json={"active": False}))
    assert client.get("/api/v1/me", headers=vh).status_code == 401


def test_cross_tenant_isolation(client, auth, setup):
    env = {**os.environ, **env_values(), "ROAD_DATA_DIR": str(ROOT / "data"), "PYTHONPATH": str(ROOT), "PYTHONPYCACHEPREFIX": str(ROOT / ".cache/pycache")}
    result = subprocess.run([sys.executable, "-m", "services.identity.bootstrap", "--org", "isolation", "--login", "admin@road.local", "--password-env", "ROAD_BOOTSTRAP_PASSWORD"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    token = data(client.post("/api/v1/auth/login", json={"org_code": "isolation", "login": "admin@road.local", "password": env["ROAD_BOOTSTRAP_PASSWORD"]}))["access_token"]
    foreign = {"Authorization": "Bearer " + token}
    assert client.get("/api/v1/profiles/" + setup["profile_id"], headers=foreign).status_code == 404
    assert client.get("/api/v1/images/" + setup["observation"]["image"]["file_id"] + "/content", headers=foreign).status_code == 404
    data(client.get("/api/v1/observations", headers=foreign))


def test_payload_limits(client, setup):
    too_big = client.post("/api/v1/images", headers=setup["dh"], data={"metadata": json.dumps(setup["metadata"])}, files={"file": ("oversize.jpg", b"x" * 5_000_001, "image/jpeg")})
    assert too_big.status_code == 413
