"""Actual HTTP privacy lifecycle and evidence/export regressions."""
import copy
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import pytest
from PIL import Image
from test_http import client, auth, setup, data, utc, wait_job,env_values

pytestmark = pytest.mark.http


def fresh(client, auth, setup, blob=None, mime="image/jpeg"):
    run=copy.deepcopy(setup["run"]);run["run_id"]=str(uuid4());run["started_at"]=utc(-25)
    data(client.post("/api/v1/runs",headers=setup["dh"],json=run),201)
    blob=blob or setup["blob"]
    file_id=str(uuid4());captured=utc(-5);sha=hashlib.sha256(blob).hexdigest()
    approval=data(client.post("/api/v1/image-approvals",headers=auth,json={"file_id":file_id,"device_id":setup["observation"]["device_id"],"sha256":sha,"size_bytes":len(blob),"mime":mime,"captured_at":captured,"reviewed_at":utc(),"redaction_model_version":"SYNTHETIC-NO-PEOPLE"}),201)
    metadata={"file_id":file_id,"approval_id":approval["approval_id"],"sha256":sha,"captured_at":captured}
    data(client.post("/api/v1/images",headers=setup["dh"],data={"metadata":json.dumps(metadata)},files={"file":("synthetic",blob,mime)}),201)
    o=copy.deepcopy(setup["observation"]);o.update(observation_id=str(uuid4()),event_id=str(uuid4()),run_id=run["run_id"],captured_at=captured)
    o["image"].update(file_id=file_id,sha256=sha)
    data(client.post("/api/v1/observations",headers=setup["dh"],json=o),201)
    return o


def receipt(client,auth,identifier,kind="access",target_kind="observation"):
    j=data(client.post("/api/v1/privacy-requests",headers=auth,json={"request_type":kind,"target_refs":[{"kind":target_kind,"id":identifier}],"reason":"SYNTHETIC 내부 티켓 확인"}),202)
    assert wait_job(client,auth,j["job_id"])["status"]=="succeeded"
    return data(client.get("/api/v1/privacy-requests/"+j["request_id"],headers=auth))


def test_access_approval_list_download_and_conflict(client,auth,setup):
    o=fresh(client,auth,setup)
    j=receipt(client,auth,o["observation_id"])
    assert j["status"]=="in_progress" and j["resolved_at"] is None
    assert j["request_id"] in {x["request_id"] for x in data(client.get("/api/v1/privacy-requests",headers=auth))["items"]}
    path="/api/v1/privacy-requests/"+j["request_id"]
    assert client.get(path+"/download",headers=auth).status_code==404
    assert client.patch(path,headers=auth,json={"action":"approve_access","expected_version":j["version"]+1,"reason":"확인"}).status_code==409
    done=data(client.patch(path,headers=auth,json={"action":"approve_access","expected_version":j["version"],"reason":"관리자 본인 및 대상 확인"}))
    assert done["status"]=="completed" and done["resolved_at"]
    exported=client.get(done["download_path"],headers=auth)
    assert exported.status_code==200 and "no-store" in exported.headers["cache-control"]
    records=exported.json()["data"]["observations"]
    assert [x["observation_id"] for x in records]==[o["observation_id"]]
    assert client.patch(path,headers=auth,json={"action":"reject","expected_version":done["version"],"reason":"이미 처리됨"}).status_code==409


def test_corrections_append_to_original_and_reject_invalid(client,auth,setup):
    o=fresh(client,auth,setup)
    j=receipt(client,auth,o["observation_id"],"correction");path="/api/v1/privacy-requests/"+j["request_id"]
    correction={"observation_id":o["observation_id"],"field":"location","value":{"lat":95.0,"lon":127.0,"quality":"valid"}}
    body={"action":"apply_correction","expected_version":j["version"],"reason":"기준점 재확인","corrections":[correction]}
    assert client.patch(path,headers=auth,json=body).status_code==422
    correction["value"]["lat"]=37.51
    done=data(client.patch(path,headers=auth,json=body))
    assert done["result_code"]=="CORRECTION_APPLIED"
    detail=data(client.get("/api/v1/observations/"+o["observation_id"],headers=auth))
    assert detail["location"]["lat"]==o["location"]["lat"]
    assert detail["corrections"][0]["value"]["lat"]==37.51
    assert detail["corrections"][0]["reason"]=="기준점 재확인"


def test_privacy_request_can_be_rejected_with_stable_resolution(client,auth,setup):
    o=fresh(client,auth,setup);j=receipt(client,auth,o["observation_id"],"correction")
    result=data(client.patch("/api/v1/privacy-requests/"+j["request_id"],headers=auth,json={"expected_version":j["version"],"action":"reject","reason":"추가 근거 없음"}))
    assert result["status"]=="rejected"
    again=data(client.get("/api/v1/privacy-requests/"+j["request_id"],headers=auth))
    assert again["resolved_at"]==result["resolved_at"]


def test_run_deletion_blocks_paths_exports_and_replay(client,auth,setup):
    o=fresh(client,auth,setup)
    access=receipt(client,auth,o["observation_id"])
    access=data(client.patch("/api/v1/privacy-requests/"+access["request_id"],headers=auth,json={"expected_version":access["version"],"action":"approve_access","reason":"삭제 전 제한 열람"}))
    deletion=receipt(client,auth,o["run_id"],"deletion","run")
    assert deletion["status"]=="completed"
    assert client.get("/api/v1/runs/"+o["run_id"],headers=auth).status_code in (404,410)
    assert client.get(access["download_path"],headers=auth).status_code in (404,410)
    assert client.get("/api/v1/observations/"+o["observation_id"],headers=auth).status_code==410
    assert client.post("/api/v1/observations",headers=setup["dh"],json=o).status_code==410


def test_filtered_csv_includes_evidence_and_correction_columns(client,auth,setup):
    o=fresh(client,auth,setup)
    j=data(client.get("/api/v1/reports",headers=auth,params={"format":"csv","run_id":o["run_id"],"snapshot_at":utc()}),202)
    assert wait_job(client,auth,j["job_id"])["status"]=="succeeded"
    r=client.get("/api/v1/reports/"+j["job_id"]+"/download",headers=auth)
    rows=list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert len(rows)==1 and rows[0]["observation_id"]==o["observation_id"]
    assert rows[0]["location_source"]=="vehicle_observation"
    assert rows[0]["model_version"]==o["model_version"]
    assert "corrections" in rows[0] and "unassessable_reason" in rows[0]


def test_png_evidence_is_supported(client,auth,setup):
    out=io.BytesIO();Image.new("RGB",(1280,720),(45,55,60)).save(out,format="PNG")
    o=fresh(client,auth,setup,out.getvalue(),"image/png")
    assert client.get("/api/v1/images/"+o["image"]["file_id"]+"/content",headers=auth).status_code==200


def test_metadata_upload_and_private_admin_actions_are_rejected(client,auth,setup):
    out=io.BytesIO();im=Image.new("RGB",(1280,720));exif=Image.Exif();exif[270]="private descriptive metadata";im.save(out,format="JPEG",exif=exif)
    blob=out.getvalue();file_id=str(uuid4());sha=hashlib.sha256(blob).hexdigest();captured=utc(-2)
    approval=data(client.post("/api/v1/image-approvals",headers=auth,json={"file_id":file_id,"device_id":setup["observation"]["device_id"],"sha256":sha,"size_bytes":len(blob),"mime":"image/jpeg","captured_at":captured,"reviewed_at":utc(),"redaction_model_version":"SYNTHETIC"}),201)
    response=client.post("/api/v1/images",headers=setup["dh"],data={"metadata":json.dumps({"file_id":file_id,"approval_id":approval["approval_id"],"sha256":sha,"captured_at":captured})},files={"file":("exif.jpg",blob,"image/jpeg")})
    assert response.status_code==422 and response.json()["error"]["code"]=="IMAGE_METADATA_NOT_STRIPPED"
    assert client.get("/api/v1/privacy-requests",headers=setup["dh"]).status_code==403


@pytest.mark.parametrize("body",[{"request_type":[],"target_refs":[],"reason":"x"},{"request_type":"access","target_refs":[{"kind":"image","id":"wrong"}],"reason":"x"}])
def test_invalid_privacy_inputs_are_422(client,auth,body):
    assert client.post("/api/v1/privacy-requests",headers=auth,json=body).status_code==422


def test_complete_review_workflow_and_reopening_history(client,auth,setup):
    o=fresh(client,auth,setup);path="/api/v1/observations/"+o["observation_id"]
    current=data(client.get(path,headers=auth))
    for status in ("confirmed","action_planned","completed"):
        result=data(client.patch(path+"/review",headers=auth,json={"expected_case_id":current["case_id"],"expected_version":current["case_version"],"status":status,"reason":"SYNTHETIC 조치 근거 확인"}))
        current=data(client.get(path,headers=auth));assert current["review"]["status"]==status
    denied=client.patch(path+"/review",headers=auth,json={"expected_case_id":current["case_id"],"expected_version":current["case_version"],"status":"confirmed","reason":"재개 근거"})
    assert denied.status_code==409
    data(client.patch(path+"/review",headers=auth,json={"expected_case_id":current["case_id"],"expected_version":current["case_version"],"status":"confirmed","reason":"관리자 재검토 근거","override_transition":True}))
    history=data(client.get("/api/v1/cases/"+current["case_id"]+"/history",headers=auth))["entries"]
    assert len(history)==4 and history[-1]["to_status"]=="confirmed"


def test_manual_duplicate_merge_preserves_observations_and_revoke_restores_case(client,auth,setup):
    source=fresh(client,auth,setup);target=fresh(client,auth,setup)
    left=data(client.get("/api/v1/observations/"+source["observation_id"],headers=auth))
    right=data(client.get("/api/v1/observations/"+target["observation_id"],headers=auth))
    body={"source_observation_id":source["observation_id"],"target_observation_id":target["observation_id"],"decision":"approved","source_version":left["case_version"],"target_version":right["case_version"],"reason":"두 비식별 근거사진과 진행 방향 수동 확인"}
    link=data(client.post("/api/v1/duplicate-links",headers=auth,json=body),201)
    merged=data(client.get("/api/v1/observations/"+source["observation_id"],headers=auth))
    assert merged["case_id"]==right["case_id"] and merged["observation_id"]==source["observation_id"]
    data(client.patch("/api/v1/duplicate-links/"+link["id"],headers=auth,json={"decision":"revoked","expected_version":link["version"],"reason":"별개 객체로 재검토"}))
    restored=data(client.get("/api/v1/observations/"+source["observation_id"],headers=auth))
    assert restored["case_id"]==left["case_id"]


def test_public_demo_report_minimizes_identifiers_time_and_free_text(client,auth,setup):
    o=fresh(client,auth,setup)
    j=data(client.get("/api/v1/reports",headers=auth,params={"format":"csv","run_id":o["run_id"],"public_demo":"true","snapshot_at":utc()}),202)
    assert wait_job(client,auth,j["job_id"])["status"]=="succeeded"
    r=client.get("/api/v1/reports/"+j["job_id"]+"/download",headers=auth)
    rows=list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert len(rows)==1 and rows[0]["observation_id"]!=o["observation_id"]
    assert len(rows[0]["observation_id"])==16 and len(rows[0]["captured_at"])==10
    assert rows[0]["review_reason"]=="" and rows[0]["corrections"]=="[]"
    assert rows[0]["public_demo"]=="True"


def test_one_hundred_observations_three_sends_preserve_ids_without_loss(client,auth,setup):
    base=fresh(client,auth,setup);payloads=[]
    for _ in range(100):
        o=copy.deepcopy(base);o.update(observation_id=str(uuid4()),event_id=str(uuid4()));payloads.append(o)
    def send(o):
        response=client.post("/api/v1/observations",headers=setup["dh"],json=o)
        return response.status_code,response.json()
    with ThreadPoolExecutor(max_workers=6) as pool:
        initial=list(pool.map(send,payloads))
        assert all(status==201 for status,_ in initial)
        for _ in range(2):
            repeated=list(pool.map(send,payloads))
            assert all(status==200 and body["data"]["deduplicated"] is True for status,body in repeated)
            assert [body["data"]["case_id"] for _,body in repeated]==[body["data"]["case_id"] for _,body in initial]
    listed=data(client.get("/api/v1/observations",headers=auth,params={"run_id":base["run_id"],"limit":200}))["items"]
    assert {o["observation_id"] for o in payloads}<={o["observation_id"] for o in listed}
    assert len(listed)==101
    receipt(client,auth,base["run_id"],"deletion","run")


def test_parallel_image_uploads_are_idempotent_and_keep_approved_bytes(client,auth,setup):
    file_id=str(uuid4());captured=utc(-2);blob=setup["blob"];sha=hashlib.sha256(blob).hexdigest()
    approval=data(client.post("/api/v1/image-approvals",headers=auth,json={"file_id":file_id,"device_id":setup["observation"]["device_id"],"sha256":sha,"size_bytes":len(blob),"mime":"image/jpeg","captured_at":captured,"reviewed_at":utc(),"redaction_model_version":"SYNTHETIC-NO-PEOPLE"}),201)
    metadata={"file_id":file_id,"approval_id":approval["approval_id"],"sha256":sha,"captured_at":captured}
    def upload(_):
        return client.post("/api/v1/images",headers=setup["dh"],data={"metadata":json.dumps(metadata)},files={"file":("same.jpg",blob,"image/jpeg")})
    with ThreadPoolExecutor(max_workers=4) as pool:responses=list(pool.map(upload,range(4)))
    assert sorted(r.status_code for r in responses)==[200,200,200,201]
    assert all(r.json()["data"]["image_id"]==file_id for r in responses)
    content=client.get("/api/v1/images/"+file_id+"/content",headers=auth)
    assert hashlib.sha256(content.content).hexdigest()==sha


def test_other_organization_cannot_overwrite_an_existing_image_file_id(client,auth,setup):
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,**env_values(),"ROAD_DATA_DIR":str(root/"data"),"PYTHONPATH":str(root),"PYTHONPYCACHEPREFIX":str(root/".cache/pycache")}
    bootstrap=subprocess.run([sys.executable,"-m","services.identity.bootstrap","--org","isolation","--login","admin@road.local","--password-env","ROAD_BOOTSTRAP_PASSWORD"],cwd=root,env=env,capture_output=True,timeout=30)
    assert bootstrap.returncode==0
    token=data(client.post("/api/v1/auth/login",json={"org_code":"isolation","login":"admin@road.local","password":env["ROAD_BOOTSTRAP_PASSWORD"]}))["access_token"]
    foreign={"Authorization":"Bearer "+token}
    device=data(client.post("/api/v1/devices",headers=foreign,json={"name":"SYNTHETIC collision test"}),201)
    file_id=setup["image"]["image_id"];captured=utc(-1)
    out=io.BytesIO();Image.new("RGB",(1280,720),(130,20,10)).save(out,format="JPEG")
    blob=out.getvalue();sha=hashlib.sha256(blob).hexdigest()
    approval=data(client.post("/api/v1/image-approvals",headers=foreign,json={"file_id":file_id,"device_id":device["device_id"],"sha256":sha,"size_bytes":len(blob),"mime":"image/jpeg","captured_at":captured,"reviewed_at":utc(),"redaction_model_version":"SYNTHETIC-NO-PEOPLE"}),201)
    rejected=client.post("/api/v1/images",headers={"Authorization":"Bearer "+device["device_token"]},data={"metadata":json.dumps({"file_id":file_id,"approval_id":approval["approval_id"],"sha256":sha,"captured_at":captured})},files={"file":("different.jpg",blob,"image/jpeg")})
    assert rejected.status_code==404
    original=client.get("/api/v1/images/"+file_id+"/content",headers=auth)
    assert original.status_code==200 and hashlib.sha256(original.content).hexdigest()==setup["image"]["sha256"]
    data(client.patch("/api/v1/devices/"+device["device_id"],headers=foreign,json={"active":False}))
