
import base64,hashlib,hmac,json,math,re
import jcs
from fastapi import Request,Query,Response
from pydantic import ValidationError
from common.runtime import *
from .schemas import Observation,ReviewPatch
store=Store("inspection");app=app_for("inspection")
def payload(x):return {k:v for k,v in x.items() if k not in ("id","org_id","kind","version","created","expires")}
def canonical_case(id,org,s=None):
    seen=set()
    while True:
        if id in seen:fail(409,"MERGE_CYCLE")
        seen.add(id);c=store.get(id,org,"case",s)
        if not c.get("merged_into"):return c
        id=c["merged_into"]
def view(o,p,s=None):
    c=canonical_case(o["case_id"],p["org_id"],s)
    return {**o["payload"],"id":o["id"],"observation_id":o["id"],"case_id":c["id"],"case_version":c["version"],
            "review":{"status":c["status"],"priority":c["priority"],"reason":c.get("reason"),"updated_at":c.get("updated_at")},
            "expires_at":iso(o["expires"]),"image_content_path":"/api/v1/images/"+o["payload"]["image"]["file_id"]+"/content"}
@app.post("/api/v1/runs",status_code=201)
async def create_run(req:Request,response:Response):
    p=principal(req);role(p,"device");b=await req.json()
    required={"run_id","device_id","profile_id","started_at","manifest_hash","precheck"}
    if set(b)!=required:fail(422,"INVALID_RUN")
    if not isinstance(b["manifest_hash"],str) or not re.fullmatch("[a-f0-9]{64}",b["manifest_hash"]):fail(422,"INVALID_MANIFEST_HASH")
    for f in ("run_id","device_id","profile_id"):valid_uuid(b[f])
    if b["device_id"]!=p["sub"]:fail(403,"FORBIDDEN")
    if stamp(b["started_at"])>now()+2:fail(422,"INVALID_TIME")
    if stamp(b["started_at"])+90*86400<=now():fail(410,"DATA_EXPIRED")
    checks={"camera_preview_ok","mount_secure","lens_clear","profile_valid","gnss_device_connected","storage_ge_10gb","notice_present"}
    if not isinstance(b["precheck"],dict) or set(b["precheck"])!=checks or any(v is not True for v in b["precheck"].values()):fail(422,"PRECHECK_FAILED")
    await remote("identity","/internal/devices/"+b["device_id"],p)
    await remote("analysis","/api/v1/profiles/"+b["profile_id"],p)
    try:
        old=store.get(b["run_id"],p["org_id"],"run")
        if old["start_hash"]!=digest(b):fail(409,"IDEMPOTENCY_CONFLICT")
        response.status_code=200;return ok({"run_id":old["id"],"status":old["status"],"manifest_hash":old["manifest_hash"]})
    except HTTPException as e:
        if e.status_code!=404:raise
    store.create(b["run_id"],p["org_id"],"run",{**b,"start_hash":digest(b),"status":"recording","segments":[]},stamp(b["started_at"])+90*86400)
    response.headers["ETag"]='"'+b["manifest_hash"]+'"'
    return ok({"run_id":b["run_id"],"status":"recording","manifest_hash":b["manifest_hash"]})
@app.get("/api/v1/runs/{id}")
def get_run(id:str,req:Request):
    p=principal(req);x=store.get(id,p["org_id"],"run")
    if p["role"]=="device" and x["device_id"]!=p["sub"]:fail(403,"FORBIDDEN")
    return ok(x)
@app.patch("/api/v1/runs/{id}")
async def end_run(id:str,req:Request):
    p=principal(req);role(p,"device");b=await req.json();x=store.get(id,p["org_id"],"run")
    if x["device_id"]!=p["sub"]:fail(403,"FORBIDDEN")
    if not {"ended_at","status","manifest_hash"}<=set(b) or set(b)-{"ended_at","status","manifest_hash","summary"} or b["status"] not in ("completed","interrupted"):fail(422,"INVALID_RUN")
    if not isinstance(b["manifest_hash"],str) or not re.fullmatch("[a-f0-9]{64}",b["manifest_hash"]):fail(422,"INVALID_MANIFEST_HASH")
    if stamp(b["ended_at"])<stamp(x["started_at"]) or stamp(b["ended_at"])>now()+2:fail(422,"INVALID_TIME")
    if x.get("ended_at")==b["ended_at"] and x["manifest_hash"]==b["manifest_hash"]:return ok(x)
    expected=req.headers.get("if-match")
    if not expected:fail(428,"VERSION_REQUIRED")
    if expected.strip('"')!=x["manifest_hash"]:fail(409,"VERSION_CONFLICT")
    for o in store.list(p["org_id"],"observation"):
        if o["payload"]["run_id"]==id and stamp(o["payload"]["captured_at"])>stamp(b["ended_at"])+2:fail(409,"RUN_OBSERVATION_RANGE")
    data=payload(x);data.update(b)
    return ok(store.change(id,p["org_id"],data,x["version"]))
@app.put("/api/v1/runs/{id}/path")
async def set_path(id:str,req:Request):
    p=principal(req);role(p,"device");b=await req.json();x=store.get(id,p["org_id"],"run")
    if x["device_id"]!=p["sub"]:fail(403,"FORBIDDEN")
    if set(b)!={"segments","manifest_hash"} or not isinstance(b["segments"],list):fail(422,"INVALID_PATH")
    if b["manifest_hash"]!=x["manifest_hash"]:fail(409,"VERSION_CONFLICT")
    count=0
    for seg in b["segments"]:
        try:
            coords=seg["coordinates"]
            if len(coords)<2 or stamp(seg["from"])>=stamp(seg["to"]):raise ValueError("Invalid segment")
            for lon,lat in coords:
                if type(lon) not in (int,float) or type(lat) not in (int,float) or not math.isfinite(lon) or not math.isfinite(lat) or not -180<=lon<=180 or not -90<=lat<=90:raise ValueError("Invalid coordinate")
            count+=len(coords)
        except (KeyError,ValueError,TypeError,AssertionError):fail(422,"INVALID_PATH")
    if count>1000:fail(422,"TOO_MANY_COORDINATES")
    if x["status"]!="recording" and x.get("segments")!=b["segments"]:fail(409,"RUN_ALREADY_COMPLETE")
    data=payload(x);data["segments"]=b["segments"]
    return ok(store.change(id,p["org_id"],data,x["version"]))
