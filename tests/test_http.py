"""Real HTTP contract/error tests. Start project services before running.

All image evidence generated here is SYNTHETIC: these tests do not measure AI
accuracy, real camera performance or privacy detector recall.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
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


class PrivateFixture(dict):
    def __repr__(self):
        return "<synthetic HTTP fixture: generated credentials redacted>"


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=os.getenv("ROAD_TEST_BASE_URL", "http://127.0.0.1:8000"), timeout=30) as c:
        health = c.get("/health")
        assert health.status_code == 200, "Gateway must be started before HTTP tests"
        yield c


@pytest.fixture(scope="session")
def auth(client):
    login = data(client.post("/api/v1/auth/login", json={"org_code": "demo", "login": "admin@road.local", "password": env_values()["ROAD_BOOTSTRAP_PASSWORD"]}))
    return PrivateFixture({"Authorization": "Bearer " + login["access_token"]})


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
    return PrivateFixture({"device": device, "dh": dh, "run": run_body, "profile": profile_body, "profile_id": profile_id, "bundle": bundle_body, "observation": observation, "blob": blob, "metadata": metadata, "image": image_result, "approval_body": approval_body})


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


@pytest.mark.parametrize("raw", ["{", "null", "[]", '"text"', "", '{"name":"one","name":"two"}', '{"name":1e9999}'])
def test_malformed_or_non_object_json_is_422_not_500(client, auth, raw):
    response = client.post("/api/v1/devices", headers={**auth, "Content-Type": "application/json"}, content=raw)
    assert response.status_code == 422, response.text
    assert "traceback" not in response.text.lower()


def test_profile_and_bundle_reject_wrong_scalar_types(client, auth, setup):
    profile = json.loads(json.dumps(setup["profile"]))
    profile["name"] = {"invalid": "object instead of string"}
    assert client.post("/api/v1/profiles", headers=auth, json=profile).status_code == 422
    profile["name"] = "bad-image-size-" + uuid4().hex[:8]
    profile["calibration"]["image_size"] = [True, 720]
    assert client.post("/api/v1/profiles", headers=auth, json=profile).status_code == 422
    bundle = json.loads(json.dumps(setup["bundle"]))
    bundle["version"] = "bad-threshold-" + uuid4().hex[:8]
    bundle["thresholds"]["pothole_confidence"] = True
    assert client.post("/api/v1/model-bundles", headers=auth, json=bundle).status_code == 422


def test_run_path_put_and_manifest_concurrency(client, auth, setup):
    path = "/api/v1/runs/" + setup["run"]["run_id"]
    body = {"segments": [{"coordinates": [[127.0, 37.5], [127.0001, 37.5001]], "from": utc(-30), "to": utc(-20), "quality": "valid", "visible_scope": {"profile_id": setup["profile_id"], "roi_id": "main", "assessment_state": "assessed", "reason": None}}], "manifest_hash": setup["run"]["manifest_hash"]}
    data(client.put(path + "/path", headers=setup["dh"], json=body))
    data(client.put(path + "/path", headers=setup["dh"], json=body))
    complete = {"ended_at": utc(), "status": "completed", "manifest_hash": "2"*64, "summary": {"captured_frames": 1800, "dropped_frames": 0, "quality_failures": 0, "gnss_gap_seconds": 0}}
    assert client.patch(path, headers=setup["dh"], json=complete).status_code == 428
    assert client.patch(path, headers={**setup["dh"], "If-Match": '"' + "0"*64 + '"'}, json=complete).status_code == 409
    data(client.patch(path, headers={**setup["dh"], "If-Match": '"' + setup["run"]["manifest_hash"] + '"'}, json=complete))
    saved = data(client.get(path, headers=auth))
    assert saved["status"] == "completed"


def wait_job(client, auth, job_id):
    for _ in range(60):
        job = data(client.get("/api/v1/jobs/" + job_id, headers=auth))
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.25)
    pytest.fail("Background job did not reach a terminal state")


def test_pdf_report_has_readable_korean_content(client, auth):
    from pypdf import PdfReader
    response = client.get("/api/v1/reports", headers=auth, params={"format": "pdf", "snapshot_at": utc()})
    assert response.status_code in (200, 202), response.text
    result = data(response, response.status_code)
    assert wait_job(client, auth, result["job_id"])["status"] == "succeeded"
    pdf = client.get("/api/v1/reports/" + result["job_id"] + "/download", headers=auth)
    assert pdf.status_code == 200
    assert pdf.headers["content-type"].startswith("application/pdf")
    reader = PdfReader(io.BytesIO(pdf.content))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "도로점검 보고서" in text
    assert "재귀반사검사" in text
    assert len(reader.pages) >= 1


def test_privacy_delete_removes_image_and_observation_and_blocks_replay(client, auth, setup):
    privacy_run = json.loads(json.dumps(setup["run"]))
    privacy_run["run_id"], privacy_run["started_at"] = str(uuid4()), utc(-30)
    data(client.post("/api/v1/runs", headers=setup["dh"], json=privacy_run), 201)
    file_id, captured = str(uuid4()), utc(-5)
    sha = hashlib.sha256(setup["blob"]).hexdigest()
    approval = data(client.post("/api/v1/image-approvals", headers=auth, json={"file_id": file_id, "device_id": setup["observation"]["device_id"], "sha256": sha, "size_bytes": len(setup["blob"]), "mime": "image/jpeg", "captured_at": captured, "reviewed_at": utc(), "redaction_model_version": "SYNTHETIC-TEST-NO-PEOPLE"}), 201)
    metadata = {"file_id": file_id, "approval_id": approval["approval_id"], "sha256": sha, "captured_at": captured}
    data(client.post("/api/v1/images", headers=setup["dh"], data={"metadata": json.dumps(metadata)}, files={"file": ("privacy-synthetic.jpg", setup["blob"], "image/jpeg")}), 201)
    payload = json.loads(json.dumps(setup["observation"]))
    payload["observation_id"], payload["event_id"] = str(uuid4()), str(uuid4())
    payload["image"]["file_id"] = file_id
    payload["captured_at"] = captured
    payload["run_id"] = privacy_run["run_id"]
    data(client.post("/api/v1/observations", headers=setup["dh"], json=payload), 201)
    related = json.loads(json.dumps(payload))
    related["observation_id"], related["event_id"] = str(uuid4()), str(uuid4())
    data(client.post("/api/v1/observations", headers=setup["dh"], json=related), 201)
    result = data(client.post("/api/v1/privacy-requests", headers=auth, json={"request_type": "deletion", "target_refs": [{"kind": "observation", "id": payload["observation_id"]}], "reason": "Synthetic automated privacy deletion test"}), 202)
    assert wait_job(client, auth, result["job_id"])["status"] == "succeeded"
    state = data(client.get("/api/v1/privacy-requests/" + result["request_id"], headers=auth))
    assert state["status"] == "completed"
    assert client.get("/api/v1/observations/" + payload["observation_id"], headers=auth).status_code == 410
    assert client.get("/api/v1/observations/" + related["observation_id"], headers=auth).status_code == 410
    assert client.get("/api/v1/images/" + file_id + "/content", headers=auth).status_code in (404, 410)
    assert client.post("/api/v1/observations", headers=setup["dh"], json=payload).status_code == 410


def test_true_parallel_idempotent_ingestion(client, auth, setup):
    payload = json.loads(json.dumps(setup["observation"]))
    payload["observation_id"], payload["event_id"] = str(uuid4()), str(uuid4())
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: client.post("/api/v1/observations", headers=setup["dh"], json=payload), range(4)))
    assert sorted(r.status_code for r in responses) == [200, 200, 200, 201], [(r.status_code, r.text[:300]) for r in responses]
    assert {r.json()["data"]["case_id"] for r in responses}.__len__() == 1
    rows = data(client.get("/api/v1/observations", headers=auth, params={"run_id": payload["run_id"], "limit": 200}))["items"]
    assert sum(o["observation_id"] == payload["observation_id"] for o in rows) == 1


def test_true_parallel_review_conflict(client, auth, setup):
    payload = json.loads(json.dumps(setup["observation"]))
    payload["observation_id"], payload["event_id"] = str(uuid4()), str(uuid4())
    created = data(client.post("/api/v1/observations", headers=setup["dh"], json=payload), 201)
    patch = {"expected_case_id": created["case_id"], "expected_version": created["case_version"], "status": "confirmed", "reason": "Synthetic concurrent review"}
    endpoint = "/api/v1/observations/" + payload["observation_id"] + "/review"
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: client.patch(endpoint, headers=auth, json=patch), range(2)))
    assert sorted(r.status_code for r in responses) == [200, 409]
    history = data(client.get("/api/v1/cases/" + created["case_id"] + "/history", headers=auth))
    entries = history.get("entries", history.get("items", []))
    assert sum(e.get("to_status") == "confirmed" for e in entries) == 1
