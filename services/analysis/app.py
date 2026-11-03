
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
