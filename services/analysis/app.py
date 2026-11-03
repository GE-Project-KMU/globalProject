from services.contracts import UUIDString
from services.contracts import ProfileCreate,BundleCreate

import math
from fastapi import Request
from common.runtime import *
store=Store("analysis");app=app_for("analysis")

from .registry import public_profile,public_bundle,quarantine_legacy,counts,assert_usable
quarantine_legacy(store)

def finite_tree(x):
    if isinstance(x,float) and not math.isfinite(x):fail(422,"FINITE_NUMBER_REQUIRED")
    if isinstance(x,dict):
        for v in x.values():finite_tree(v)
    if isinstance(x,list):
        for v in x:finite_tree(v)
def validate_hash(x):
    import re
    if not isinstance(x,str) or not re.fullmatch("[a-f0-9]{64}",x):fail(422,"INVALID_SHA256")
def numeric(x):
    return type(x) in (int,float) and math.isfinite(x)
def require(condition):
    if not condition:raise ValueError("Input constraint failed")
def scalar_string(x,maximum=150):
    return isinstance(x,str) and 1<=len(x)<=maximum
@app.post("/api/v1/profiles",status_code=201)
async def create_profile(req:Request,body:ProfileCreate):
    p=principal(req);role(p,"admin");b=body.model_dump(exclude_unset=True,by_alias=True);finite_tree(b)
    required={"name","version","camera_serial","lens_label","calibration","config_hash"}
    if not required<=set(b) or set(b)-required or type(b["version"]) is not int or b["version"]<1:fail(422,"INVALID_PROFILE")
    if not scalar_string(b["name"],100) or not scalar_string(b["camera_serial"]) or not scalar_string(b["lens_label"]):fail(422,"INVALID_PROFILE")
    validate_hash(b["config_hash"])
    c=b["calibration"];keys={"image_size","height_m","pitch_deg","camera_matrix","distortion","homography","roi_polygon_px","visible_distance_m","calibrated_at"}
    if not isinstance(c,dict) or not keys<=set(c):fail(422,"INVALID_CALIBRATION")
    import numpy as np
    try:
        require(isinstance(c["image_size"],list) and len(c["image_size"])==2)
        require(all(type(x) is int and 0<x<=16384 for x in c["image_size"]))
        require(isinstance(c["visible_distance_m"],list) and len(c["visible_distance_m"])==2)
        require(all(numeric(x) for x in c["visible_distance_m"]))
        w,h=c["image_size"];near,far=c["visible_distance_m"]
        require(numeric(c["height_m"]) and numeric(c["pitch_deg"]))
        require(w>0 and h>0 and c["height_m"]>0 and 0<=near<far)
        for key in ("camera_matrix","homography"):
            require(isinstance(c[key],list) and len(c[key])==3)
            require(all(isinstance(row,list) and len(row)==3 and all(numeric(x) for x in row) for row in c[key]))
            m=np.asarray(c[key],dtype=float);require(m.shape==(3,3) and abs(np.linalg.det(m))>1e-12)
        require(isinstance(c["distortion"],list) and len(c["distortion"]) in (4,5,8,12,14))
        require(all(numeric(x) for x in c["distortion"]))
        require(isinstance(c["roi_polygon_px"],list) and 3<=len(c["roi_polygon_px"])<=256)
        require(all(isinstance(pt,list) and len(pt)==2 and all(numeric(x) for x in pt) for pt in c["roi_polygon_px"]))
        require(all(0<=x<=w and 0<=y<=h for x,y in c["roi_polygon_px"]))
        stamp(c["calibrated_at"])
    except (AssertionError,ValueError,TypeError,KeyError):fail(422,"INVALID_CALIBRATION")
    if any(x["name"]==b["name"] and x["version"]==b["version"] for x in store.list(p["org_id"],"profile")):fail(409,"PROFILE_VERSION_EXISTS")
    id=uid();row=store.create(id,p["org_id"],"profile",{**b,"calibrated_by":p["sub"]});store.audit(p,"profile.create",id);return ok(public_profile(row))
@app.post("/api/v1/model-bundles",status_code=201)
async def create_bundle(req:Request,body:BundleCreate):
    p=principal(req);role(p,"admin");b=body.model_dump(exclude_unset=True,by_alias=True);finite_tree(b)
    required={"version","weights_hashes","preprocess","thresholds","config_hash","code_commit","environment_hash"}
    if not required<=set(b) or set(b)-required-{"code_commit","environment_hash"}:fail(422,"INVALID_BUNDLE")
    validate_hash(b["config_hash"])
    if not scalar_string(b["version"]):fail(422,"INVALID_VERSION")
    if not isinstance(b["weights_hashes"],dict) or set(b["weights_hashes"])!={"pothole","lane","redaction"}:fail(422,"INVALID_WEIGHTS")
    for h in b["weights_hashes"].values():validate_hash(h)
    if not isinstance(b["preprocess"],dict):fail(422,"INVALID_PREPROCESS")
    if "code_commit" in b and not scalar_string(b["code_commit"]):fail(422,"INVALID_CODE_COMMIT")
    if "environment_hash" in b:validate_hash(b["environment_hash"])
    t=b["thresholds"]
    try:
        require(isinstance(t,dict))
        require(all(numeric(t[k]) for k in ("pothole_confidence","pothole_nms_iou","single_frame_confidence","lane_monitor","lane_good","min_valid_ratio","min_segment_support_ratio")))
        require(0<=t["pothole_confidence"]<=1 and 0<=t["pothole_nms_iou"]<=1)
        require(0<=t["single_frame_confidence"]<=1)
        require(0<=t["lane_monitor"]<t["lane_good"]<=100)
        require(0<=t["min_valid_ratio"]<=1 and 0<=t["min_segment_support_ratio"]<=1)
    except (AssertionError,ValueError,KeyError,TypeError):fail(422,"INVALID_THRESHOLDS")
    if any(x["version"]==b["version"] for x in store.list(p["org_id"],"bundle")):fail(409,"BUNDLE_VERSION_EXISTS")
    id=uid();row=store.create(id,p["org_id"],"bundle",b)
    store.audit(p,"model_bundle.create",id)
    return ok(public_bundle(row))
@app.get("/api/v1/profiles")
def profiles(req:Request):
    p=principal(req);return ok({"items":[public_profile(row) for row in store.list(p["org_id"],"profile")],"quarantined_count":counts(store,p["org_id"],"profile")})
@app.get("/api/v1/profiles/{id}")
def get_profile(id:UUIDString,req:Request):
    p=principal(req);assert_usable(store,id,p["org_id"],"profile");return ok(public_profile(store.get(id,p["org_id"],"profile")))
@app.get("/api/v1/model-bundles")
def bundles(req:Request):
    p=principal(req);return ok({"items":[public_bundle(row) for row in store.list(p["org_id"],"bundle")],"quarantined_count":counts(store,p["org_id"],"bundle")})
@app.get("/api/v1/model-bundles/{id}")
def get_bundle(id:UUIDString,req:Request):
    p=principal(req);assert_usable(store,id,p["org_id"],"bundle");return ok(public_bundle(store.get(id,p["org_id"],"bundle")))
@app.get("/internal/model-version/{version}")
def by_version(version:str,req:Request):
    p=principal(req);x=next((x for x in store.list(p["org_id"],"bundle") if x["version"]==version),None)
    if not x:fail(404,"MODEL_VERSION_NOT_FOUND")
    return ok(x)
@app.get("/api/v1/analysis/capabilities")
def capabilities(req:Request):
    p=principal(req)
    models=DATA/"models"
    return ok({"raw_processing":"local_agent","server_raw_upload":False,
               "manifest_count":len(store.list(p["org_id"],"bundle")),
               "model_status":"verified_per_job","local_agent_command":"python -m agent --help",
               "execution_api":"/api/v1/analysis/jobs","prepared_configs_required":True,
               "job_statuses":["queued","running","completed","blocked","failed"],
               "note":"Registry metadata is not proof of trained model accuracy."})

from services.analysis.jobs import register
register(app,store)
