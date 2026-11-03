
import asyncio,json,os,time
from collections import defaultdict,deque
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import JSONResponse,Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError,ResponseValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import httpx
from common.runtime import URLS,remote,ok,fail,uid,now,sign,log,role,request_id_context
from services.contracts import Envelope
app=FastAPI(title="Road Insight API Gateway",version="0.1.0",docs_url=None,redoc_url=None,openapi_url=None)
app.add_middleware(CORSMiddleware,allow_origins=["http://127.0.0.1:5173","http://localhost:5173","http://127.0.0.1:8080"],allow_methods=["GET","POST","PUT","PATCH"],allow_headers=["Authorization","Content-Type","If-Match"],expose_headers=["ETag","X-Request-ID","Content-Disposition"])
login_attempts=defaultdict(deque)

@app.exception_handler(StarletteHTTPException)
async def http_error(req,exc):
    error=exc.detail if isinstance(exc.detail,dict) else {"code":{404:"NOT_FOUND",405:"METHOD_NOT_ALLOWED",422:"VALIDATION_ERROR"}.get(exc.status_code,str(exc.detail)),"message":str(exc.detail)}
    return JSONResponse({"error":error,"meta":{"request_id":request_id_context.get()}},status_code=exc.status_code,headers=exc.headers)

@app.exception_handler(RequestValidationError)
async def validation_error(req,exc):
    fields=[{"field_path":".".join(map(str,x["loc"])),"code":x["type"]} for x in exc.errors()]
    return JSONResponse({"error":{"code":"VALIDATION_ERROR","message":"Invalid request","details":fields},"meta":{"request_id":request_id_context.get()}},status_code=422)

@app.exception_handler(ResponseValidationError)
async def response_error(req,exc):
    fields=[{"field_path":".".join(map(str,x["loc"])),"code":x["type"]} for x in exc.errors()]
    log.error("gateway response schema failure fields=%s",fields)
    return JSONResponse({"error":{"code":"INTERNAL_ERROR","message":"Unexpected server response"},"meta":{"request_id":request_id_context.get()}},status_code=500)
def service(path):
    prefix=path.split("/",1)[0]
    if prefix in ("auth","me","users","devices"):return "identity"
    if prefix in ("images","image-approvals"):return "media"
    if prefix in ("profiles","model-bundles","analysis"):return "analysis"
    if prefix in ("reports","jobs","privacy-requests"):return "reporting"
    if prefix in ("runs","observations","cases","duplicate-links","summary"):return "inspection"
    fail(404,"NOT_FOUND")
@app.middleware("http")
async def errors(req,call_next):
    rid=uid()
    token=request_id_context.set(rid)
    try:r=await call_next(req)
    except HTTPException as exc:
        e=exc.detail if isinstance(exc.detail,dict) else {"code":str(exc.detail),"message":str(exc.detail)}
        r=JSONResponse({"error":e,"meta":{"request_id":rid}},status_code=exc.status_code,headers=exc.headers)
    except Exception:
        log.exception("gateway unhandled request %s",rid)
        r=JSONResponse({"error":{"code":"INTERNAL_ERROR","message":"Unexpected gateway error"},"meta":{"request_id":rid}},status_code=500)
    finally:request_id_context.reset(token)
    if "X-Request-ID" not in r.headers:r.headers["X-Request-ID"]=rid
    r.headers["X-Content-Type-Options"]="nosniff"
    return r
@app.get("/health")
def health():return {"status":"ok","service":"gateway","instance_id":os.getenv("ROAD_INSTANCE_ID",""),"pid":os.getpid()}
async def auth(req):
    value=req.headers.get("authorization","")
    if not value.startswith("Bearer ") or len(value)<10:fail(401,"UNAUTHENTICATED")
    r=await remote("identity","/internal/introspect",{"sub":"gateway","role":"system","org_id":"system"},method="POST",json={"token":value[7:]})
    return r.json()["data"]
@app.get("/api/v1/system/status",response_model=Envelope)
async def status(req:Request):
    p=await auth(req)
    async def check(name,url):
        try:
            async with httpx.AsyncClient(timeout=3,trust_env=False) as c:r=await c.get(url+"/health")
            return {"name":name,"status":"ok" if r.status_code==200 else "error"}
        except httpx.HTTPError:return {"name":name,"status":"unavailable"}
    return ok({"services":await asyncio.gather(*(check(n,u) for n,u in URLS.items()))})
@app.get("/api/v1/audit-logs",response_model=Envelope)
async def audit_logs(req:Request,limit:int=100):
    p=await auth(req);role(p,"admin")
    if not 1<=limit<=1000:fail(422,"INVALID_LIMIT")
    async def fetch(name):
        r=await remote(name,"/internal/audit",p,params={"limit":limit})
        return [{**x,"service":name} for x in r.json()["data"]["items"]]
    batches=await asyncio.gather(*(fetch(n) for n in URLS))
    rows=sorted([x for batch in batches for x in batch],key=lambda x:(x["at"],x["id"]),reverse=True)
    return ok({"items":rows[:limit],"next_cursor":None})
@app.get("/openapi.json")
async def schema(req:Request):
    p=await auth(req)
    if p["role"]!="admin":fail(403,"FORBIDDEN")
    merged={"openapi":"3.1.0","info":{"title":"Road Insight MSA API","version":"0.1.0"},"paths":{},"components":{"schemas":{},"securitySchemes":{"BearerAuth":{"type":"http","scheme":"bearer"}}},"security":[{"BearerAuth":[]}]}
    for name in URLS:
        r=await remote(name,"/openapi.json",p);s=r.json()
        merged["paths"].update({k:v for k,v in s.get("paths",{}).items() if k.startswith("/api/v1/")})
        merged["components"]["schemas"].update(s.get("components",{}).get("schemas",{}))
    own=app.openapi()
    merged["paths"].update({k:v for k,v in own.get("paths",{}).items() if k in ("/api/v1/audit-logs","/api/v1/system/status")})
    merged["paths"]["/api/v1/auth/login"]["post"]["security"]=[]
    return merged
@app.api_route("/api/v1/{path:path}",methods=["GET","POST","PATCH","PUT","DELETE"],include_in_schema=False)
async def proxy(path:str,req:Request):
    name=service(path)
    if path=="auth/login" and req.method=="POST":
        key=req.client.host if req.client else "unknown";q=login_attempts[key]
        while q and q[0]<now()-60:q.popleft()
        if len(q)>=30:raise HTTPException(429,{"code":"RATE_LIMITED","message":"Try again later"},headers={"Retry-After":"60"})
        q.append(now());p={"sub":"anonymous","role":"anonymous","org_id":"anonymous"}
    else:p=await auth(req)
    chunks=[];size=0
    max_size=5100000 if req.headers.get("content-type","").startswith("multipart/form-data") else 64000
    async for chunk in req.stream():
        size+=len(chunk)
        if size>max_size:fail(413,"PAYLOAD_TOO_LARGE")
        chunks.append(chunk)
    raw=b"".join(chunks)
    if req.method in ("POST","PATCH","PUT") and not raw and path!="auth/logout":fail(422,"BODY_REQUIRED")
    headers={"X-Road-Principal":sign(name,p)}
    for key in ("content-type","if-match"):
        if key in req.headers:headers[key]=req.headers[key]
    try:
        async with httpx.AsyncClient(timeout=45,trust_env=False) as c:
            r=await c.request(req.method,URLS[name]+"/api/v1/"+path,params=req.query_params,content=raw,headers=headers)
    except httpx.HTTPError:fail(503,"DEPENDENCY_UNAVAILABLE",service=name)
    allowed={k:v for k,v in r.headers.items() if k.lower() in ("content-type","content-disposition","cache-control","etag","retry-after","x-request-id")}
    return Response(content=r.content,status_code=r.status_code,headers=allowed)
