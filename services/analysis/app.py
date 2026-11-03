
import math
from fastapi import Request
from common.runtime import *
store=Store("analysis");app=app_for("analysis")
def finite_tree(x):
    if isinstance(x,float) and not math.isfinite(x):fail(422,"FINITE_NUMBER_REQUIRED")
    if isinstance(x,dict):
        for v in x.values():finite_tree(v)
    if isinstance(x,list):
        for v in x:finite_tree(v)
def validate_hash(x):
    import re
    if not isinstance(x,str) or not re.fullmatch("[a-f0-9]{64}",x):fail(422,"INVALID_SHA256")
@app.post("/api/v1/profiles",status_code=201)
async def create_profile(req:Request):
    p=principal(req);role(p,"admin");b=await req.json();finite_tree(b)
    required={"name","version","camera_serial","lens_label","calibration","config_hash"}
    if not required<=set(b) or set(b)-required or not isinstance(b["version"],int) or b["version"]<1:fail(422,"INVALID_PROFILE")
    validate_hash(b["config_hash"])
    c=b["calibration"];keys={"image_size","height_m","pitch_deg","camera_matrix","distortion","homography","roi_polygon_px","visible_distance_m","calibrated_at"}
    if not isinstance(c,dict) or not keys<=set(c):fail(422,"INVALID_CALIBRATION")
    import numpy as np
    try:
        w,h=c["image_size"];near,far=c["visible_distance_m"]
        assert w>0 and h>0 and c["height_m"]>0 and 0<=near<far
        for key in ("camera_matrix","homography"):
            m=np.asarray(c[key],dtype=float);assert m.shape==(3,3) and abs(np.linalg.det(m))>1e-12
        assert len(c["roi_polygon_px"])>=3
        assert all(0<=x<=w and 0<=y<=h for x,y in c["roi_polygon_px"])
        stamp(c["calibrated_at"])
    except (AssertionError,ValueError,TypeError,KeyError):fail(422,"INVALID_CALIBRATION")
    if any(x["name"]==b["name"] and x["version"]==b["version"] for x in store.list(p["org_id"],"profile")):fail(409,"PROFILE_VERSION_EXISTS")
    id=uid();store.create(id,p["org_id"],"profile",b);return ok({"profile_id":id,**b})
@app.post("/api/v1/model-bundles",status_code=201)
async def create_bundle(req:Request):
    p=principal(req);role(p,"admin");b=await req.json();finite_tree(b)
    required={"version","weights_hashes","preprocess","thresholds","config_hash"}
    if not required<=set(b) or set(b)-required-{"code_commit","environment_hash"}:fail(422,"INVALID_BUNDLE")
    validate_hash(b["config_hash"])
    if not isinstance(b["version"],str) or not b["version"]:fail(422,"INVALID_VERSION")
    if set(b["weights_hashes"])!={"pothole","lane","redaction"}:fail(422,"INVALID_WEIGHTS")
    for h in b["weights_hashes"].values():validate_hash(h)
    t=b["thresholds"]
    try:
        assert 0<=t["pothole_confidence"]<=1 and 0<=t["pothole_nms_iou"]<=1
        assert 0<=t["single_frame_confidence"]<=1
        assert 0<=t["lane_monitor"]<t["lane_good"]<=100
        assert 0<=t["min_valid_ratio"]<=1 and 0<=t["min_segment_support_ratio"]<=1
    except (AssertionError,KeyError,TypeError):fail(422,"INVALID_THRESHOLDS")
    if any(x["version"]==b["version"] for x in store.list(p["org_id"],"bundle")):fail(409,"BUNDLE_VERSION_EXISTS")
    id=uid();store.create(id,p["org_id"],"bundle",b)
    return ok({"model_bundle_id":id,**b,"deployment_status":"manifest_registered_weights_not_verified"})
@app.get("/api/v1/profiles")
def profiles(req:Request):return ok({"items":store.list(principal(req)["org_id"],"profile")})
@app.get("/api/v1/profiles/{id}")
def get_profile(id:str,req:Request):return ok(store.get(id,principal(req)["org_id"],"profile"))
@app.get("/api/v1/model-bundles")
def bundles(req:Request):return ok({"items":store.list(principal(req)["org_id"],"bundle")})
@app.get("/api/v1/model-bundles/{id}")
def get_bundle(id:str,req:Request):return ok(store.get(id,principal(req)["org_id"],"bundle"))
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
               "model_status":"target_weights_required","local_agent_command":"python -m agent --help",
               "note":"Registry metadata is not proof of trained model accuracy."})
