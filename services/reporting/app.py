
import asyncio, csv, html, io, re
from contextlib import asynccontextmanager
from fastapi import Request,Query
from fastapi.responses import FileResponse
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from common.runtime import *
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from .contracts import Envelope as ReportingEnvelope,JobView,PrivacyView,PrivacyList,Receipt,AccessExport
from services.contracts import UUIDString
store=Store("reporting");app=app_for("reporting");FILES=DATA/"reporting"/"files";FILES.mkdir(exist_ok=True)
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
def clean(x):return {k:v for k,v in x.items() if k not in ("id","org_id","version","created","expires","kind")}
def invalidate_derived(org,ids,code):
    identifiers=set(ids)
    for record in store.list(org,"job"):
        if record.get("kind_name")=="report" and identifiers & {o["observation_id"] for o in record.get("snapshot",{}).get("items",[])}:
            (FILES/(record["id"]+"."+record["format"])).unlink(missing_ok=True)
            data=clean(record);data.update(status="failed",snapshot={"items":[]},error_code=code)
            store.change(record["id"],org,data,record["version"])
        elif record.get("access_snapshot") and identifiers & set(record["access_snapshot"]["observation_ids"]):
            data=clean(record);data.pop("access_snapshot",None);data["result_code"]=code
            store.change(record["id"],org,data,record["version"])
def jobview(j):
    return {"job_id":j["id"],"status":j["status"],"progress":j.get("progress",0),"error_code":j.get("error_code"),"status_url":"/api/v1/jobs/"+j["id"],"download_path":"/api/v1/reports/"+j["id"]+"/download" if j["status"]=="succeeded" and j["kind_name"]=="report" else None,"expires_at":iso(j["expires"]) if j["expires"] else None}
def csv_cell(v):
    s=str(v or "")
    return "'"+s if s.lstrip().startswith(("=","+","-","@")) else s
def render_csv(path,items):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["case_id","type","captured_at","status","priority","latitude","longitude","location_quality","retention_pct","grade","demo","observation_id","event_id","location_source","confidence","quality_usable","quality_reasons","unassessable_reason","valid_area_ratio","segment_support_ratio","model_version","profile_id","review_reason","reviewed_at","corrections","public_demo","data_origin"])
        for o in items:
            lane=o.get("lane") or {};quality=o.get("quality") or {}
            values=[o["case_id"],o["type"],o["captured_at"],o["review"]["status"],o["review"]["priority"],o["location"]["lat"],o["location"]["lon"],o["location"]["quality"],lane.get("retention_pct"),lane.get("grade"),o.get("demo",False),o["observation_id"],o["event_id"],o["location"].get("source"),(o.get("pothole") or {}).get("confidence"),quality.get("usable"),";".join(quality.get("reasons",[])),lane.get("unassessable_reason"),lane.get("valid_area_ratio"),lane.get("segment_support_ratio"),o.get("model_version"),o.get("profile_id"),o["review"].get("reason"),o["review"].get("updated_at"),json.dumps(o.get("corrections",[]),ensure_ascii=False),o.get("public_exported",False)]
            values.append(o.get('data_origin','field'))
            w.writerow([csv_cell(v) if isinstance(v,str) else v for v in values])
def public_export(items):
    import copy
    rows=copy.deepcopy(items)
    for o in rows:
        loc=o["location"]
        for key in ("lat","lon"):
            if loc[key] is not None:loc[key]=round(loc[key],3)
        o["captured_at"]=o["captured_at"][:10]
        for key in ("case_id","observation_id","event_id"):
            o[key]=digest(["public-report",o[key]])[:16]
        o["review"]={**o["review"],"reason":None,"updated_at":None}
        o["corrections"]=[]
        o["public_exported"]=True
    return rows
def render_pdf(path,items):
    style=ParagraphStyle("Korean",fontName="HYSMyeongJo-Medium",fontSize=9,leading=14,wordWrap="CJK")
    title=ParagraphStyle("TitleKR",parent=style,fontSize=18,leading=24)
    story=[Paragraph("Road Insight 도로점검 보고서",title),Spacer(1,12),
           Paragraph("비식별 관측 자료 · 차량 관측 위치 · 시각적 마모 후보 · 법정 재귀반사검사 대체 불가",style),
           Paragraph("생성 시각(UTC): "+iso()+" · 점검 건: "+str(len(items)),style),Spacer(1,12)]
    counts={kind:sum(o["type"]==kind for o in items) for kind in ("pothole","lane_segment")}
    if any(o.get("public_exported") for o in items):story.append(Paragraph("공개 시연용: 위치 소수점 3자리·촬영 날짜 단위로 축약하고 내부 식별자와 자유 서술을 최소화했습니다.",style))
    story.extend([Paragraph(f"대표 포트홀 점검 건: {counts['pothole']} · 대표 차선 구간 점검 건: {counts['lane_segment']}",style),Paragraph("위치는 차량의 관측 위치이며 손상 객체의 측량 좌표가 아닙니다. 차선 지표는 영상 잔존율로 법정 안전등급이 아닙니다.",style),Spacer(1,12)])
    for o in items:
        lane=o.get("lane") or {}
        story.append(Paragraph("자료 출처: "+{'http_test':'합성 HTTP 테스트 · 실제 탐지 결과 아님','synthetic_demo':'합성 시연 · 실제 탐지 결과 아님','field':'일반 관측 · 출처 미확인'}.get(o.get('data_origin','field'),'미확인'),style))
        values=[("점검 ID",o["case_id"]),("종류",o["type"]),("관측 시각",o["captured_at"]),("검토 상태",o["review"]["status"]),("우선순위",o["review"]["priority"]),("위치 품질",o["location"]["quality"]),("관측 위치",f"{o['location']['lat']}, {o['location']['lon']}"),("AI 신뢰도(위험 확률 아님)",(o.get("pothole") or {}).get("confidence")),("영상 잔존율",lane.get("retention_pct")),("영상 등급",lane.get("grade")),("판정 불가 사유",lane.get("unassessable_reason")),("품질 사유",", ".join((o.get("quality") or {}).get("reasons",[]))),("유효 면적/구간 지원율",f"{lane.get('valid_area_ratio')} / {lane.get('segment_support_ratio')}"),("모델/프로파일",f"{o.get('model_version')} / {o.get('profile_id')}"),("검토·조치 근거",o["review"].get("reason")),("정정 이력",json.dumps(o.get("corrections",[]),ensure_ascii=False)),("가상 시연",o.get("demo",False))]
        for k,v in values:story.append(Paragraph(html.escape(f"{k}: {v}"),style))
        story.append(Spacer(1,12))
    SimpleDocTemplate(str(path),title="Road Insight Inspection Report").build(story)
async def process_job(id,p):
    j=store.get(id,p["org_id"],"job")
    if j["status"]=="succeeded" or j["status"]=="failed":return
    data=clean(j);data.update(status="running",lease_until=now()+120,progress=10)
    try:j=store.change(id,p["org_id"],data,j["version"])
    except HTTPException:return
    try:
        data=clean(j)
        if j["kind_name"]=="report":
            items=j["snapshot"]["items"]
            valid=(await remote("inspection","/internal/validate-observations",p,method="POST",json={"ids":[x["observation_id"] for x in items]})).json()["data"]["valid"]
            if not valid:fail(410,"DATA_EXPIRED")
            path=FILES/(id+"."+j["format"])
            (render_csv if j["format"]=="csv" else render_pdf)(path,public_export(items) if j.get("public_demo") else items)
            data.update(status="succeeded",progress=100)
        else:
            refs=j["request"]["target_refs"]
            if j["request"]["request_type"]=="deletion":
                if not j.get("plan"):
                    plan=(await remote("inspection","/internal/privacy/plan",p,method="POST",json={"target_refs":refs})).json()["data"]
                    data["plan"]=plan;j=store.change(id,p["org_id"],data,j["version"]);data=clean(j)
                plan=data["plan"]
                await remote("media","/internal/privacy/delete",p,method="POST",json={"image_ids":plan["image_ids"]})
                await remote("inspection","/internal/privacy/delete",p,method="POST",json={"observation_ids":plan["observation_ids"],"run_ids":plan.get("run_ids",[])})
                for report in store.list(p["org_id"],"job"):
                    if report.get("kind_name")=="report":
                        intersects=set(plan["observation_ids"]) & {x["observation_id"] for x in report.get("snapshot",{}).get("items",[])}
                        if intersects:
                            (FILES/(report["id"]+"."+report["format"])).unlink(missing_ok=True)
                            rp=clean(report);rp.update(status="failed",snapshot={"items":[]} ,error_code="DATA_EXPIRED")
                            store.change(report["id"],p["org_id"],rp,report["version"])
                    elif report.get("kind_name")=="privacy" and report.get("access_snapshot"):
                        snapshot=report["access_snapshot"]
                        if set(plan["observation_ids"]) & set(snapshot["observation_ids"]):
                            rp=clean(report);rp.pop("access_snapshot",None);rp.update(result_code="ACCESS_DATA_DELETED")
                            store.change(report["id"],p["org_id"],rp,report["version"])
                checks={}
                for service,refs in (("media",{"image_ids":plan["image_ids"]}),("inspection",{"observation_ids":plan["observation_ids"],"run_ids":plan.get("run_ids",[])})):
                    checks[service]=(await remote(service,"/internal/privacy/verify",p,method="POST",json=refs)).json()["data"]
                if not all(x["verified"] for x in checks.values()):fail(503,"DELETION_NOT_VERIFIED")
                data.update(status="succeeded",progress=100,deletion_verification=checks,backup_policy="configuration_only_no_detail_backups")
            else:
                data.update(status="succeeded",progress=100,result_code="MANUAL_ADMIN_REVIEW_REQUIRED")
            if data.get("result_code")!="MANUAL_ADMIN_REVIEW_REQUIRED":
                data.update(resolved_at=iso(),result_code="DELETION_VERIFIED")
        store.change(id,p["org_id"],data,j["version"])
    except Exception as e:
        if not isinstance(e,HTTPException):log.exception("reporting job failed %s",id)
        if j.get("kind_name")=="report":(FILES/(id+"."+j["format"])).unlink(missing_ok=True)
        latest=store.get(id,p["org_id"],"job");data=clean(latest);data.update(status="failed",error_code=e.detail.get("code","PROCESSING_FAILED") if isinstance(e,HTTPException) and isinstance(e.detail,dict) else "PROCESSING_FAILED")
        store.change(id,p["org_id"],data,latest["version"])
async def worker():
    while True:
        try:
            for j in store.list(None,"job"):
                if j["status"]=="pending" or j["status"]=="running" and j.get("lease_until",0)<now():
                    await process_job(j["id"],j["principal"])
            with store.tx() as s:
                for row in s.scalars(select(Aggregate).where(Aggregate.kind=="job")):
                    snapshot=row.payload.get("access_snapshot")
                    if snapshot and stamp(snapshot["expires_at"])<=now():
                        data=dict(row.payload);data.pop("access_snapshot",None);data["result_code"]="ACCESS_DATA_EXPIRED";row.payload=data;row.version+=1
                for row in list(s.scalars(select(Aggregate).where(Aggregate.expires<=now()))):
                    if row.kind=="job":
                        for suffix in (".csv",".pdf"):(FILES/(row.id+suffix)).unlink(missing_ok=True)
                    s.delete(row)
        except asyncio.CancelledError:raise
        except Exception:log.exception("report worker cycle failed")
        await asyncio.sleep(.5)
@asynccontextmanager
async def lifespan(app):
    task=asyncio.create_task(worker())
    try:yield
    finally:
        task.cancel()
        try:await task
        except asyncio.CancelledError:pass
app.router.lifespan_context=lifespan
@app.get("/api/v1/reports",response_model=ReportingEnvelope[JobView],responses={202:{"model":ReportingEnvelope[JobView],"description":"Report queued"}})
async def report(req:Request,format:Literal["csv","pdf"]="csv",from_:str|None=Query(None,alias="from"),to:str|None=None,type:str|None=None,review_status:str|None=None,bbox:str|None=None,quality:str|None=None,grade:str|None=None,run_id:str|None=None,snapshot_at:str|None=None,public_demo:bool=False,data_scope:Literal['all','presentation','test','field','synthetic']='all'):
    p=principal(req);role(p,"admin","reviewer","viewer")
    if format not in ("csv","pdf"):fail(422,"INVALID_REPORT_FORMAT")
    params=dict(req.query_params);params.pop("format",None);params.pop("public_demo",None);snapshot_at=params.pop("snapshot_at",None)
    if set(params)-{"from","to","type","review_status","bbox","quality","grade","run_id","data_scope"}:fail(422,"INVALID_FILTER")
    if data_scope!='all':params['data_scope']=data_scope
    else:params.pop('data_scope',None)
    if snapshot_at is not None:stamp(snapshot_at)
    reference_time=now()
    start=stamp(params["from"]) if "from" in params else reference_time-90*86400
    end=stamp(params["to"]) if "to" in params else reference_time
    if start>=end or end-start>90*86400:fail(422,"INVALID_REPORT_RANGE")
    key=digest([p["sub"],params,format,snapshot_at,public_demo])
    for j in store.list(p["org_id"],"job"):
        if j.get("dedup_key")==key and j["status"]!="failed":
            return JSONResponse(ok(jobview(j)),status_code=200 if j["status"]=="succeeded" else 202)
    running=[j for j in store.list(p["org_id"],"job") if j.get("kind_name")=="report" and j["status"] in ("pending","running")]
    if len(running)>=20 or sum(j["owner"]==p["sub"] for j in running)>=5:fail(429,"REPORT_QUEUE_FULL")
    snapshot=(await remote("inspection","/internal/report-snapshot",p,params=params)).json()["data"]
    if len(snapshot["items"])>10000:fail(422,"REPORT_TOO_LARGE")
    expiry=min([now()+86400]+[stamp(x["expires_at"]) for x in snapshot["items"]])
    id=uid()
    j=store.create(id,p["org_id"],"job",{"owner":p["sub"],"kind_name":"report","principal":p,"format":format,"public_demo":public_demo,"dedup_key":key,"snapshot":snapshot,"status":"pending","progress":0},expiry)
    store.audit(p,"report.create",id)
    return JSONResponse(ok(jobview(j)),status_code=202)
@app.get("/api/v1/jobs/{id}",response_model=ReportingEnvelope[JobView])
def job(id:UUIDString,req:Request):
    p=principal(req);j=store.get(valid_uuid(id),p["org_id"],"job")
    if j["owner"]!=p["sub"] and p["role"]!="admin":fail(404,"NOT_FOUND")
    return ok(jobview(j))
@app.get("/api/v1/reports/{id}/download")
async def download(id:UUIDString,req:Request):
    p=principal(req);j=store.get(valid_uuid(id),p["org_id"],"job")
    if j["owner"]!=p["sub"] and p["role"]!="admin":fail(404,"NOT_FOUND")
    if j["status"]!="succeeded" or j["kind_name"]!="report":fail(409,"REPORT_NOT_READY")
    valid=(await remote("inspection","/internal/validate-observations",p,method="POST",json={"ids":[x["observation_id"] for x in j["snapshot"]["items"]]})).json()["data"]["valid"]
    if not valid:fail(410,"DATA_EXPIRED")
    path=FILES/(id+"."+j["format"])
    if not path.is_file():fail(404,"FILE_NOT_FOUND")
    store.audit(p,"report.download",id)
    return FileResponse(path,media_type="text/csv; charset=utf-8" if j["format"]=="csv" else "application/pdf",filename="road-insight."+j["format"],headers={"Cache-Control":"private, no-store"})
class TargetRef(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True)
    kind:Literal["observation","image","run","case"]
    id:str

class PrivacyCreate(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True)
    request_type:Literal["access","correction","deletion"]
    target_refs:list[TargetRef]=Field(min_length=1,max_length=100)
    reason:str=Field(min_length=1,max_length=500)

@app.post("/api/v1/privacy-requests",status_code=202,response_model=ReportingEnvelope[Receipt])
async def privacy(body:PrivacyCreate,req:Request):
    p=principal(req);role(p,"admin");b=body.model_dump()
    if set(b)!={"request_type","target_refs","reason"} or b["request_type"] not in ("access","correction","deletion") or not isinstance(b["reason"],str) or not 1<=len(b["reason"])<=500 or not isinstance(b["target_refs"],list) or not 1<=len(b["target_refs"])<=100:fail(422,"INVALID_PRIVACY_REQUEST")
    for ref in b["target_refs"]:
        if not isinstance(ref,dict) or set(ref)!={"kind","id"} or ref["kind"] not in ("observation","image","run","case"):fail(422,"INVALID_TARGET")
        valid_uuid(ref["id"])
    plan=(await remote("inspection","/internal/privacy/plan",p,method="POST",json={"target_refs":b["target_refs"]})).json()["data"]
    if b["request_type"]=="deletion":
        await remote("inspection","/internal/privacy/block",p,method="POST",json={"observation_ids":plan["observation_ids"],"run_ids":plan.get("run_ids",[])})
        await remote("media","/internal/privacy/block",p,method="POST",json={"image_ids":plan["image_ids"]})
    id=uid();store.create(id,p["org_id"],"job",{"owner":p["sub"],"kind_name":"privacy","principal":p,"request":b,"plan":plan,"status":"pending","progress":0,"created_at":iso(),"actions":[]},now()+90*86400)
    store.audit(p,"privacy.create",id)
    return ok({"request_id":id,"job_id":id,"status":"received"})
@app.get("/api/v1/privacy-requests/{id}",response_model=ReportingEnvelope[PrivacyView])
def privacy_status(id:UUIDString,req:Request):
    p=principal(req);role(p,"admin");j=store.get(valid_uuid(id),p["org_id"],"job")
    if j["kind_name"]!="privacy":fail(404,"NOT_FOUND")
    return ok(privacy_view(j))

def privacy_view(j):
    status={"pending":"received","running":"in_progress","succeeded":"completed","failed":"failed"}[j["status"]]
    if j.get("result_code")=="MANUAL_ADMIN_REVIEW_REQUIRED":status="in_progress"
    if j.get("rejected"):status="rejected"
    return {"request_id":j["id"],"job_id":j["id"],"version":j["version"],"request_type":j["request"]["request_type"],
            "status":status,"result_code":j.get("result_code"),"created_at":j["created_at"],
            "resolved_at":j.get("resolved_at"),"target_refs":j["request"]["target_refs"],"reason":j["request"]["reason"],
            "decision_reason":j.get("decision_reason"),"actions":j.get("actions",[]),
            "download_path":"/api/v1/privacy-requests/"+j["id"]+"/download" if j.get("result_code")=="ACCESS_APPROVED" else None}

@app.get("/api/v1/privacy-requests",response_model=ReportingEnvelope[PrivacyList])
def privacy_list(req:Request):
    p=principal(req);role(p,"admin")
    return ok({"items":[privacy_view(j) for j in sorted(store.list(p["org_id"],"job"),key=lambda x:x["created"],reverse=True) if j.get("kind_name")=="privacy"]})

class PrivacyDecision(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True)
    expected_version:int=Field(ge=1)
    action:Literal["approve_access","apply_correction","reject","retry"]
    reason:str=Field(min_length=1,max_length=500)
    corrections:list[dict]|None=Field(default=None,min_length=1,max_length=100)

@app.patch("/api/v1/privacy-requests/{id}",response_model=ReportingEnvelope[PrivacyView])
async def privacy_decide(id:UUIDString,b:PrivacyDecision,req:Request):
    p=principal(req);role(p,"admin");id=valid_uuid(id)
    j=store.get(id,p["org_id"],"job")
    if j.get("kind_name")!="privacy":fail(404,"NOT_FOUND")
    if j["version"]!=b.expected_version:fail(409,"VERSION_CONFLICT",current_version=j["version"])
    state=privacy_view(j)["status"]
    if state in ("completed","rejected") or j["status"] in ("pending","running"):fail(409,"PRIVACY_NOT_REVIEWABLE")
    data=clean(j);data.update(decision_reason=b.reason)
    if b.action=="retry":
        if j["status"]!="failed" or j["request"]["request_type"]!="deletion":fail(409,"PRIVACY_ACTION_MISMATCH")
        data.update(status="pending",error_code=None,result_code=None)
    elif b.action=="reject":
        if j["request"]["request_type"]=="deletion":fail(409,"DELETION_REQUIRES_RETRY")
        if b.corrections is not None:fail(422,"INVALID_CORRECTIONS")
        data.update(status="succeeded",rejected=True,result_code="ADMIN_REJECTED",resolved_at=iso())
    elif b.action=="approve_access":
        if j["request"]["request_type"]!="access" or b.corrections is not None:fail(409,"PRIVACY_ACTION_MISMATCH")
        snapshot=(await remote("inspection","/internal/privacy/access",p,method="POST",json={"target_refs":j["request"]["target_refs"]})).json()["data"]
        data.update(status="succeeded",result_code="ACCESS_APPROVED",resolved_at=iso(),access_snapshot=snapshot)
    else:
        if j["request"]["request_type"]!="correction" or b.corrections is None:fail(409,"PRIVACY_ACTION_MISMATCH")
        await remote("inspection","/internal/privacy/correct",p,method="POST",json={"request_id":id,"observation_ids":j["plan"]["observation_ids"],"corrections":b.corrections,"reason":b.reason})
        invalidate_derived(p["org_id"],j["plan"]["observation_ids"],"DATA_CORRECTED")
        data.update(status="succeeded",result_code="CORRECTION_APPLIED",resolved_at=iso())
    data["actions"]=[*data.get("actions",[]),{"at":iso(),"actor_id":p["sub"],"action":b.action,"reason":b.reason}]
    j=store.change(id,p["org_id"],data,j["version"]);store.audit(p,"privacy."+b.action,id)
    return ok(privacy_view(j))

@app.get("/api/v1/privacy-requests/{id}/download",response_model=ReportingEnvelope[AccessExport])
async def privacy_download(id:UUIDString,req:Request):
    p=principal(req);role(p,"admin");j=store.get(valid_uuid(id),p["org_id"],"job")
    if j.get("kind_name")!="privacy" or j.get("result_code")!="ACCESS_APPROVED":fail(404,"NOT_FOUND")
    snapshot=j["access_snapshot"]
    if stamp(snapshot["expires_at"])<=now():fail(410,"DATA_EXPIRED")
    valid=(await remote("inspection","/internal/validate-observations",p,method="POST",json={"ids":snapshot["observation_ids"]})).json()["data"]["valid"]
    if not valid:fail(410,"DATA_EXPIRED")
    store.audit(p,"privacy.access.download",id)
    return JSONResponse(ok({"observations":snapshot["observations"],"runs":snapshot["runs"]}),headers={"Cache-Control":"private, no-store","Content-Disposition":"attachment; filename=privacy-access.json"})
