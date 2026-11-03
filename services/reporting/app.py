
import asyncio, csv, html, io, re
from contextlib import asynccontextmanager
from fastapi import Request
from fastapi.responses import FileResponse
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from common.runtime import *
store=Store("reporting");app=app_for("reporting");FILES=DATA/"reporting"/"files";FILES.mkdir(exist_ok=True)
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
def clean(x):return {k:v for k,v in x.items() if k not in ("id","org_id","version","created","expires","kind")}
def jobview(j):
    return {"job_id":j["id"],"status":j["status"],"progress":j.get("progress",0),"error_code":j.get("error_code"),"status_url":"/api/v1/jobs/"+j["id"],"download_path":"/api/v1/reports/"+j["id"]+"/download" if j["status"]=="succeeded" and j["kind_name"]=="report" else None,"expires_at":iso(j["expires"]) if j["expires"] else None}
def csv_cell(v):
    s=str(v or "")
    return "'"+s if s.lstrip().startswith(("=","+","-","@")) else s
def render_csv(path,items):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["case_id","type","captured_at","status","priority","latitude","longitude","location_quality","retention_pct","grade","demo"])
        for o in items:
            w.writerow([o["case_id"],o["type"],o["captured_at"],o["review"]["status"],o["review"]["priority"],o["location"]["lat"],o["location"]["lon"],o["location"]["quality"],(o.get("lane") or {}).get("retention_pct"),(o.get("lane") or {}).get("grade"),o.get("demo",False)])
def render_pdf(path,items):
    style=ParagraphStyle("Korean",fontName="HYSMyeongJo-Medium",fontSize=9,leading=14,wordWrap="CJK")
    title=ParagraphStyle("TitleKR",parent=style,fontSize=18,leading=24)
    story=[Paragraph("Road Insight 도로점검 보고서",title),Spacer(1,12),
           Paragraph("비식별 관측 자료 · 차량 관측 위치 · 시각적 마모 후보 · 법정 재귀반사검사 대체 불가",style),
           Paragraph("생성 시각(UTC): "+iso()+" · 점검 건: "+str(len(items)),style),Spacer(1,12)]
    for o in items:
        values=[("점검 ID",o["case_id"]),("종류",o["type"]),("관측 시각",o["captured_at"]),("검토 상태",o["review"]["status"]),("위치 품질",o["location"]["quality"]),("잔존율",(o.get("lane") or {}).get("retention_pct")),("가상 시연",o.get("demo",False))]
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
            (render_csv if j["format"]=="csv" else render_pdf)(path,items)
            data.update(status="succeeded",progress=100)
        else:
            refs=j["request"]["target_refs"]
            if j["request"]["request_type"]=="deletion":
                if not j.get("plan"):
                    plan=(await remote("inspection","/internal/privacy/plan",p,method="POST",json={"target_refs":refs})).json()["data"]
                    data["plan"]=plan;j=store.change(id,p["org_id"],data,j["version"]);data=clean(j)
                plan=data["plan"]
                await remote("media","/internal/privacy/delete",p,method="POST",json={"image_ids":plan["image_ids"]})
                await remote("inspection","/internal/privacy/delete",p,method="POST",json={"observation_ids":plan["observation_ids"]})
                for report in store.list(p["org_id"],"job"):
                    if report.get("kind_name")=="report":
                        intersects=set(plan["observation_ids"]) & {x["observation_id"] for x in report.get("snapshot",{}).get("items",[])}
                        if intersects:
                            (FILES/(report["id"]+"."+report["format"])).unlink(missing_ok=True)
                            rp=clean(report);rp.update(status="failed",snapshot={"items":[]} ,error_code="DATA_EXPIRED")
                            store.change(report["id"],p["org_id"],rp,report["version"])
                data.update(status="succeeded",progress=100)
            else:
                data.update(status="succeeded",progress=100,result_code="MANUAL_ADMIN_REVIEW_REQUIRED")
        store.change(id,p["org_id"],data,j["version"])
    except Exception as e:
        log.exception("reporting job failed %s",id)
        latest=store.get(id,p["org_id"],"job");data=clean(latest);data.update(status="failed",error_code=e.detail.get("code","PROCESSING_FAILED") if isinstance(e,HTTPException) and isinstance(e.detail,dict) else "PROCESSING_FAILED")
        store.change(id,p["org_id"],data,latest["version"])
async def worker():
    while True:
        try:
            for j in store.list(None,"job"):
                if j["status"]=="pending" or j["status"]=="running" and j.get("lease_until",0)<now():
                    await process_job(j["id"],j["principal"])
            with store.tx() as s:
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
@app.get("/api/v1/reports")
async def report(req:Request,format:str="csv"):
    p=principal(req);role(p,"admin","reviewer","viewer")
    if format not in ("csv","pdf"):fail(422,"INVALID_REPORT_FORMAT")
    params=dict(req.query_params);params.pop("format",None);snapshot_at=params.pop("snapshot_at",None)
    if set(params)-{"from","to","type","review_status"}:fail(422,"INVALID_FILTER")
    reference_time=now()
    start=stamp(params["from"]) if "from" in params else reference_time-90*86400
    end=stamp(params["to"]) if "to" in params else reference_time
    if start>=end or end-start>90*86400:fail(422,"INVALID_REPORT_RANGE")
    key=digest([p["sub"],params,format,snapshot_at])
    for j in store.list(p["org_id"],"job"):
        if j.get("dedup_key")==key and j["status"]!="failed":
            return JSONResponse(ok(jobview(j)),status_code=200 if j["status"]=="succeeded" else 202)
    snapshot=(await remote("inspection","/internal/report-snapshot",p,params=params)).json()["data"]
    if len(snapshot["items"])>10000:fail(422,"REPORT_TOO_LARGE")
    expiry=min([now()+86400]+[stamp(x["expires_at"]) for x in snapshot["items"]])
    id=uid()
    j=store.create(id,p["org_id"],"job",{"owner":p["sub"],"kind_name":"report","principal":p,"format":format,"dedup_key":key,"snapshot":snapshot,"status":"pending","progress":0},expiry)
    return JSONResponse(ok(jobview(j)),status_code=202)
@app.get("/api/v1/jobs/{id}")
def job(id:str,req:Request):
    p=principal(req);j=store.get(valid_uuid(id),p["org_id"],"job")
    if j["owner"]!=p["sub"] and p["role"]!="admin":fail(404,"NOT_FOUND")
    return ok(jobview(j))
@app.get("/api/v1/reports/{id}/download")
async def download(id:str,req:Request):
    p=principal(req);j=store.get(valid_uuid(id),p["org_id"],"job")
    if j["owner"]!=p["sub"] and p["role"]!="admin":fail(404,"NOT_FOUND")
    if j["status"]!="succeeded" or j["kind_name"]!="report":fail(409,"REPORT_NOT_READY")
    valid=(await remote("inspection","/internal/validate-observations",p,method="POST",json={"ids":[x["observation_id"] for x in j["snapshot"]["items"]]})).json()["data"]["valid"]
    if not valid:fail(410,"DATA_EXPIRED")
    path=FILES/(id+"."+j["format"])
    if not path.is_file():fail(404,"FILE_NOT_FOUND")
    return FileResponse(path,media_type="text/csv; charset=utf-8" if j["format"]=="csv" else "application/pdf",filename="road-insight."+j["format"],headers={"Cache-Control":"private, no-store"})
@app.post("/api/v1/privacy-requests",status_code=202)
async def privacy(req:Request):
    p=principal(req);role(p,"admin");b=await req.json()
    if set(b)!={"request_type","target_refs","reason"} or b["request_type"] not in ("access","correction","deletion") or not isinstance(b["reason"],str) or not 1<=len(b["reason"])<=500 or not isinstance(b["target_refs"],list) or not 1<=len(b["target_refs"])<=100:fail(422,"INVALID_PRIVACY_REQUEST")
    for ref in b["target_refs"]:
        if not isinstance(ref,dict) or set(ref)!={"kind","id"} or ref["kind"] not in ("observation","image","run","case"):fail(422,"INVALID_TARGET")
        valid_uuid(ref["id"])
    plan=(await remote("inspection","/internal/privacy/plan",p,method="POST",json={"target_refs":b["target_refs"]})).json()["data"]
    id=uid();store.create(id,p["org_id"],"job",{"owner":p["sub"],"kind_name":"privacy","principal":p,"request":b,"plan":plan,"status":"pending","progress":0,"created_at":iso()},now()+90*86400)
    return ok({"request_id":id,"job_id":id,"status":"received"})
@app.get("/api/v1/privacy-requests/{id}")
def privacy_status(id:str,req:Request):
    p=principal(req);role(p,"admin");j=store.get(valid_uuid(id),p["org_id"],"job")
    if j["kind_name"]!="privacy":fail(404,"NOT_FOUND")
    status={"pending":"received","running":"in_progress","succeeded":"completed","failed":"failed"}[j["status"]]
    if j.get("result_code")=="MANUAL_ADMIN_REVIEW_REQUIRED":status="in_progress"
    return ok({"request_id":id,"job_id":id,"request_type":j["request"]["request_type"],"status":status,"result_code":j.get("result_code"),"created_at":j["created_at"],"resolved_at":iso() if status=="completed" else None})
