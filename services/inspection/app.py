
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
@app.post("/api/v1/observations",status_code=201)
async def ingest(b:Observation,req:Request,response:Response):
    p=principal(req);role(p,"device")
    if b.device_id!=p["sub"]:fail(403,"FORBIDDEN")
    body=b.model_dump();rh=hashlib.sha256(jcs.canonicalize(body)).hexdigest();expires=stamp(b.captured_at)+90*86400
    if expires<=now():fail(410,"DATA_EXPIRED")
    try:
        store.get(digest("observation:"+b.observation_id),p["org_id"],"tombstone")
        fail(410,"DATA_EXPIRED")
    except HTTPException as e:
        if e.status_code!=404:raise
    if stamp(b.captured_at)>now()+2:fail(422,"INVALID_TIME")
    try:
        old=store.get(b.observation_id,p["org_id"],"observation")
        if old["request_hash"]!=rh or old["payload"]["device_id"]!=p["sub"]:fail(409,"IDEMPOTENCY_CONFLICT")
        response.status_code=200;v=view(old,p);return ok({**v,"event_id":b.event_id,"deduplicated":True})
    except HTTPException as e:
        if e.status_code!=404:raise
    run=store.get(b.run_id,p["org_id"],"run")
    if run["device_id"]!=p["sub"] or run["profile_id"]!=b.profile_id:fail(422,"RUN_REFERENCE_MISMATCH")
    if stamp(b.captured_at)<stamp(run["started_at"])-2 or (run.get("ended_at") and stamp(b.captured_at)>stamp(run["ended_at"])+2):fail(422,"INVALID_RUN_TIME")
    image=(await remote("media","/internal/images/"+b.image.file_id,p)).json()["data"]
    if image["sha256"]!=b.image.sha256 or stamp(image["captured_at"])!=stamp(b.captured_at):fail(422,"IMAGE_REFERENCE_MISMATCH")
    bundle=(await remote("analysis","/internal/model-version/"+b.model_version,p)).json()["data"]
    if bundle["config_hash"]!=b.config_hash:fail(422,"CONFIG_HASH_MISMATCH")
    profile=(await remote("analysis","/api/v1/profiles/"+b.profile_id,p)).json()["data"]
    if b.pothole:
        x,y,w,h=b.pothole.bbox_xywh_px;iw,ih=profile["calibration"]["image_size"]
        if x+w>iw or y+h>ih:fail(422,"BOX_OUT_OF_IMAGE")
    with store.tx() as s:
        try:
            prior=store.get(b.observation_id,p["org_id"],"observation",s)
            if prior["request_hash"]!=rh:fail(409,"IDEMPOTENCY_CONFLICT")
            response.status_code=200
            return ok({**view(prior,p,s),"event_id":b.event_id,"deduplicated":True})
        except HTTPException as e:
            if e.status_code!=404:raise
        try:
            event=store.get(b.event_id,p["org_id"],"event",s)
            if event["run_id"]!=b.run_id or event["type"]!=b.type:fail(409,"EVENT_CONFLICT")
            case_id=event["case_id"]
        except HTTPException as e:
            if e.status_code!=404:raise
            case_id=uid()
            store.create(case_id,p["org_id"],"case",{"type":b.type,"status":"unreviewed","priority":"normal","reason":"","first_seen_at":b.captured_at,"last_seen_at":b.captured_at,"representative_observation_id":b.observation_id,"merged_into":None,"demo":b.demo,"updated_at":iso()},s=s)
            store.create(b.event_id,p["org_id"],"event",{"case_id":case_id,"run_id":b.run_id,"type":b.type},expires,s)
        o=store.create(b.observation_id,p["org_id"],"observation",{"payload":body,"case_id":case_id,"request_hash":rh},expires,s)
        store.audit(p,"observation.create",b.observation_id,s)
        v=view(o,p,s)
    return ok({**v,"event_id":b.event_id,"deduplicated":False})
def rank(o):
    return (0 if o["review"]["priority"]=="urgent" else 1,0 if o["type"]=="pothole" else 1 if o.get("lane",{}).get("grade")=="worn_candidate" else 2,o["captured_at"],o["id"])
@app.get("/api/v1/observations")
def listing(req:Request,type:str|None=None,review_status:str|None=None,quality:str|None=None,grade:str|None=None,run_id:str|None=None,bbox:str|None=None,from_:str|None=Query(None,alias="from"),to:str|None=None,limit:int=Query(50,ge=1,le=200),cursor:str|None=None):
    p=principal(req);role(p,"admin","reviewer","viewer")
    if type not in (None,"pothole","lane_segment") or quality not in (None,"valid","degraded","unavailable") or review_status not in (None,"unreviewed","confirmed","false_positive","recapture","action_planned","completed") or grade not in (None,"visible_good","monitor","worn_candidate","unassessable"):fail(422,"INVALID_FILTER")
    if grade and type=="pothole" or bbox and quality=="unavailable":fail(422,"INVALID_FILTER")
    bounds=None
    if bbox:
        try:
            bounds=[float(x) for x in bbox.split(",")];w,s,e,n=bounds
            if not(all(math.isfinite(x) for x in bounds) and -180<=w<=e<=180 and -90<=s<=n<=90):raise ValueError("Invalid bbox")
        except (ValueError,AssertionError):fail(422,"INVALID_BBOX")
    ft=stamp(from_) if from_ else None;tt=stamp(to) if to else None
    if ft is not None and tt is not None and ft>=tt:fail(422,"INVALID_TIME_RANGE")
    fh=digest([p["org_id"],type,review_status,quality,grade,run_id,bbox,from_,to])
    after=None;snapshot=now()
    if cursor:
        try:
            raw,sig=cursor.rsplit(".",1)
            if not hmac.compare_digest(hmac.new(service_key("inspection"),raw.encode(),"sha256").hexdigest(),sig):raise ValueError()
            c=json.loads(base64.urlsafe_b64decode(raw))
            if c["filter"]!=fh or c["expires"]<=now():raise ValueError("Invalid cursor")
            after=tuple(c["after"]);snapshot=c["snapshot"]
        except (ValueError,KeyError,AssertionError):fail(422,"INVALID_CURSOR")
    items=[]
    with store.tx() as s:
        for o in store.list(p["org_id"],"observation",s):
            if o["created"]>snapshot:continue
            v=view(o,p,s);loc=v["location"];capt=stamp(v["captured_at"])
            if type and v["type"]!=type or review_status and v["review"]["status"]!=review_status or quality and loc["quality"]!=quality or run_id and v["run_id"]!=run_id:continue
            if grade and (v.get("lane") or {}).get("grade")!=grade:continue
            if ft is not None and capt<ft or tt is not None and capt>=tt:continue
            if bounds:
                west,south,east,north=bounds
                if loc["lat"] is None or not(west<=loc["lon"]<=east and south<=loc["lat"]<=north):continue
            items.append(v)
    items.sort(key=rank);counts={t:sum(v["type"]==t for v in items) for t in ("pothole","lane_segment")}
    page=[v for v in items if after is None or rank(v)>after]
    result=page[:limit];next_cursor=None
    if len(page)>limit:
        raw=base64.urlsafe_b64encode(json.dumps({"filter":fh,"after":rank(result[-1]),"snapshot":snapshot,"expires":now()+300}).encode()).decode()
        next_cursor=raw+"."+hmac.new(service_key("inspection"),raw.encode(),"sha256").hexdigest()
    return ok({"items":result,"next_cursor":next_cursor,"counts_by_type":counts,"counts_unit":"observation"})
@app.get("/api/v1/observations/{id}")
def detail(id:str,req:Request):
    p=principal(req);role(p,"admin","reviewer","viewer");valid_uuid(id)
    try:
        store.get(digest("observation:"+id),p["org_id"],"tombstone")
        fail(410,"DATA_EXPIRED")
    except HTTPException as e:
        if e.status_code!=404:raise
    return ok(view(store.get(id,p["org_id"],"observation"),p))
@app.patch("/api/v1/observations/{id}/review")
async def review(id:str,req:Request):
    p=principal(req);role(p,"admin","reviewer");raw=await req.json()
    if "expected_version" not in raw:fail(428,"VERSION_REQUIRED")
    try:b=ReviewPatch.model_validate(raw)
    except ValidationError:fail(422,"VALIDATION_ERROR")
    transitions={"unreviewed":{"confirmed","false_positive","recapture"},"recapture":{"confirmed","false_positive"},"confirmed":{"action_planned"},"action_planned":{"completed"}}
    with store.tx() as s:
        o=store.get(id,p["org_id"],"observation",s);c=canonical_case(o["case_id"],p["org_id"],s)
        if c["id"]!=b.expected_case_id or c["version"]!=b.expected_version:fail(409,"VERSION_CONFLICT",current_version=c["version"])
        if b.status and b.status!=c["status"] and b.status not in transitions.get(c["status"],set()) and not b.override_transition:fail(409,"STATE_CONFLICT")
        data=payload(c);before={"status":c["status"],"priority":c["priority"]}
        if b.status:data["status"]=b.status
        if b.priority:data["priority"]=b.priority
        data.update(reason=b.reason,updated_at=iso())
        c=store.change(c["id"],p["org_id"],data,b.expected_version,s)
        store.create(uid(),p["org_id"],"history",{"case_id":c["id"],"actor_id":p["sub"],"at":iso(),"from_status":before["status"],"to_status":c["status"],"from_priority":before["priority"],"to_priority":c["priority"],"reason":b.reason,"case_version":c["version"]},o["expires"],s)
        store.audit(p,"case.review",c["id"],s)
    return ok({"case_id":c["id"],"version":c["version"],"review":{"status":c["status"],"priority":c["priority"],"updated_at":c["updated_at"]}})
@app.get("/api/v1/cases/{id}/history")
def history(id:str,req:Request):
    p=principal(req);role(p,"admin","reviewer","viewer");store.get(id,p["org_id"],"case")
    return ok({"entries":[x for x in store.list(p["org_id"],"history") if x["case_id"]==id],"next_cursor":None})
@app.get("/api/v1/summary")
def summary(req:Request):
    p=principal(req);role(p,"admin","reviewer","viewer")
    items=[view(o,p) for o in store.list(p["org_id"],"observation")]
    cases={v["case_id"]:v for v in items}.values()
    return ok({"total":len(items),"pending":sum(v["review"]["status"]=="unreviewed" for v in cases),"completed":sum(v["review"]["status"]=="completed" for v in cases),"potholes":sum(v["type"]=="pothole" for v in items),"worn_lanes":sum((v.get("lane") or {}).get("grade")=="worn_candidate" for v in items),"unassessable":sum((v.get("lane") or {}).get("grade")=="unassessable" for v in items),"counts_unit":"observation","demo":any(v.get("demo") for v in items)})

def distance(a,b):
    if a["lat"] is None or b["lat"] is None:return None
    dlat=math.radians(b["lat"]-a["lat"]);dlon=math.radians(b["lon"]-a["lon"])
    h=math.sin(dlat/2)**2+math.cos(math.radians(a["lat"]))*math.cos(math.radians(b["lat"]))*math.sin(dlon/2)**2
    return 6371000*2*math.asin(min(1,math.sqrt(h)))
def link_view(x,p):
    src=canonical_case(x["source_case_id"],p["org_id"]);target=canonical_case(x["target_case_id"],p["org_id"])
    return {**x,"source_version":src["version"],"target_version":target["version"]}
@app.get("/api/v1/duplicate-links")
def duplicates(req:Request,status:str|None=None):
    p=principal(req);role(p,"admin","reviewer")
    if status not in (None,"suggested","approved","rejected","revoked"):fail(422,"INVALID_FILTER")
    existing=store.list(p["org_id"],"link")
    # Geometry only proposes candidates. Humans inspect images/direction.
    observations=[view(o,p) for o in store.list(p["org_id"],"observation")]
    bycase={o["case_id"]:o for o in observations}
    candidates=list(bycase.values())[-200:]
    known={frozenset((l["source_case_id"],l["target_case_id"])) for l in existing}
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            if a["type"]!=b["type"] or a["review"]["status"]=="completed" or b["review"]["status"]=="completed":continue
            pair=frozenset((a["case_id"],b["case_id"]))
            if pair in known:continue
            dist=distance(a["location"],b["location"])
            if dist is not None and dist<=10:
                id=uid();x=store.create(id,p["org_id"],"link",{"source_case_id":a["case_id"],"target_case_id":b["case_id"],"source_observation_id":a["id"],"target_observation_id":b["id"],"status":"suggested","distance_m":dist,"evidence":{"basis":"nearby_vehicle_observations","direction_verified":False},"reason":""},min(stamp(a["expires_at"]),stamp(b["expires_at"])))
                existing.append(x);known.add(pair)
    return ok({"items":[link_view(x,p) for x in existing if status is None or x["status"]==status],"next_cursor":None})
@app.post("/api/v1/duplicate-links",status_code=201)
async def decide_link(req:Request,response:Response):
    p=principal(req);role(p,"admin","reviewer");b=await req.json()
    required={"source_observation_id","target_observation_id","decision","source_version","target_version","reason"}
    if set(b)!=required or b["decision"] not in ("suggested","approved","rejected") or not isinstance(b["reason"],str) or not 1<=len(b["reason"])<=500:fail(422,"VALIDATION_ERROR")
    if any(type(b[k]) is not int or b[k]<1 for k in ("source_version","target_version")):fail(422,"INVALID_VERSION")
    with store.tx() as s:
        a=store.get(valid_uuid(b["source_observation_id"]),p["org_id"],"observation",s);o=store.get(valid_uuid(b["target_observation_id"]),p["org_id"],"observation",s)
        src=canonical_case(a["case_id"],p["org_id"],s);target=canonical_case(o["case_id"],p["org_id"],s)
        if src["id"]==target["id"] or src["type"]!=target["type"]:fail(409,"INVALID_MERGE")
        if src["version"]!=b["source_version"] or target["version"]!=b["target_version"]:fail(409,"VERSION_CONFLICT")
        existing=next((x for x in store.list(p["org_id"],"link",s) if x["source_case_id"]==src["id"] and x["target_case_id"]==target["id"]),None)
        data={"source_case_id":src["id"],"target_case_id":target["id"],"source_observation_id":a["id"],"target_observation_id":o["id"],"status":b["decision"],"reason":b["reason"],"distance_m":distance(a["payload"]["location"],o["payload"]["location"]),"decided_at":iso(),"decided_by":p["sub"]}
        if existing:
            x=store.change(existing["id"],p["org_id"],data,existing["version"],s);response.status_code=200
        else:x=store.create(uid(),p["org_id"],"link",data,min(a["expires"],o["expires"]),s)
        if b["decision"]=="approved":
            if target["status"]=="completed":fail(409,"COMPLETED_CASE_RECURRENCE")
            sp=payload(src);sp["merged_into"]=target["id"]
            store.change(src["id"],p["org_id"],sp,src["version"],s)
            tp=payload(target);tp["updated_at"]=iso();store.change(target["id"],p["org_id"],tp,target["version"],s)
        store.audit(p,"duplicate."+b["decision"],x["id"],s)
    return ok(x)
@app.patch("/api/v1/duplicate-links/{id}")
async def revoke_link(id:str,req:Request):
    p=principal(req);role(p,"admin","reviewer");b=await req.json()
    if set(b)!={"decision","expected_version","reason"} or b["decision"]!="revoked" or not isinstance(b["reason"],str) or not 1<=len(b["reason"])<=500:fail(422,"VALIDATION_ERROR")
    with store.tx() as s:
        x=store.get(id,p["org_id"],"link",s)
        if x["version"]!=b["expected_version"]:fail(409,"VERSION_CONFLICT")
        source=store.get(x["source_case_id"],p["org_id"],"case",s);sp=payload(source);sp["merged_into"]=None
        store.change(source["id"],p["org_id"],sp,source["version"],s)
        lp=payload(x);lp.update(status="revoked",reason=b["reason"],decided_at=iso(),decided_by=p["sub"])
        x=store.change(id,p["org_id"],lp,x["version"],s);store.audit(p,"duplicate.revoke",id,s)
    return ok(x)
