"""Technical infrastructure only. Domain services never open one another's stores."""
import asyncio, base64, hashlib, hmac, json, logging, math, os, secrets, time, uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse,Response
from fastapi.exceptions import RequestValidationError,ResponseValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.routing import APIRoute
from fastapi.datastructures import DefaultPlaceholder
from sqlalchemy import create_engine, String, Float, Integer, JSON, select, update, event, delete
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.exc import IntegrityError

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("ROAD_DATA_DIR", str(ROOT / "data"))).resolve()
if not DATA.is_relative_to(ROOT):
    raise RuntimeError("ROAD_DATA_DIR must be inside project")
DATA.mkdir(parents=True, exist_ok=True)
URLS = {n: os.getenv(f"ROAD_SERVICE_{n.upper()}_URL", f"http://127.0.0.1:{p}") for n,p in
        [("identity",8101),("media",8102),("inspection",8103),("analysis",8104),("reporting",8105)]}
log = logging.getLogger("roadwatch")
request_id_context=ContextVar("road_request_id",default=None)
def uid(): return str(uuid.uuid4())
def now(): return time.time()
def iso(t=None): return datetime.fromtimestamp(t if t is not None else now(), timezone.utc).isoformat().replace("+00:00","Z")
def stamp(s):
    if not isinstance(s,str): raise HTTPException(422, "date-time required")
    try:
        dt = datetime.fromisoformat(s.replace("Z","+00:00"))
        if dt.tzinfo is None: raise ValueError()
        return dt.timestamp()
    except (ValueError,TypeError): raise HTTPException(422,"timezone-aware date-time required")
def digest(v):
    if not isinstance(v,bytes): v=json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    return hashlib.sha256(v).hexdigest()
def fail(code,msg,**details): raise HTTPException(code,{"code":msg,"message":msg,"details":details})
def strict_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError("Duplicate JSON key")
        result[key]=value
    return result
def strict_float(raw):
    value=float(raw)
    if not math.isfinite(value):raise ValueError("Non-finite JSON number")
    return value
def strict_int(raw):
    value=int(raw)
    if abs(value)>9007199254740991:raise ValueError("Integer exceeds JSON-safe range")
    return value
def ok(data, **meta): return {"data":data,"meta":{"schema_version":"0.4","request_id":request_id_context.get(),**meta}}
def valid_uuid(v):
    try: return str(uuid.UUID(v))
    except (ValueError,TypeError,AttributeError): fail(422,"INVALID_UUID")
def service_key(name):
    key=os.getenv(f"ROAD_{name.upper()}_SIGNING_KEY") or os.getenv("ROAD_INTERNAL_SECRET","")
    if len(key)<24: raise RuntimeError("Configure ROAD_INTERNAL_SECRET (>=24 chars) or service signing keys")
    return key.encode()
def sign(name,p):
    claims={**p,"aud":name,"exp":now()+30}
    raw=base64.urlsafe_b64encode(json.dumps(claims,separators=(",",":")).encode()).decode()
    return raw+"."+hmac.new(service_key(name),raw.encode(),"sha256").hexdigest()
def principal(request:Request):
    p=getattr(request.state,"principal",None)
    if not p: fail(401,"UNAUTHENTICATED")
    return p
def role(p,*allowed):
    if p["role"] not in allowed: fail(403,"FORBIDDEN")
async def remote(name,path,p,method="GET",**kwargs):
    headers=kwargs.pop("headers",{})
    headers["X-Road-Principal"]=sign(name,p)
    try:
        async with httpx.AsyncClient(timeout=30,trust_env=False) as c:
            r=await c.request(method,URLS[name]+path,headers=headers,**kwargs)
    except httpx.HTTPError: fail(503,"DEPENDENCY_UNAVAILABLE",service=name)
    if r.is_error:
        try: detail=r.json().get("error",r.json())
        except ValueError: detail={"code":"DEPENDENCY_ERROR","message":"Dependency failed"}
        raise HTTPException(r.status_code,detail)
    return r

class Base(DeclarativeBase): pass
class Aggregate(Base):
    __tablename__="aggregates"
    id:Mapped[str]=mapped_column(String(128),primary_key=True)
    org:Mapped[str]=mapped_column(String(64),index=True)
    kind:Mapped[str]=mapped_column(String(40),index=True)
    payload:Mapped[dict]=mapped_column(JSON().with_variant(JSONB,"postgresql"))
    version:Mapped[int]=mapped_column(Integer,default=1)
    created:Mapped[float]=mapped_column(Float,default=now,index=True)
    expires:Mapped[float|None]=mapped_column(Float,nullable=True,index=True)
STORES={}
class Store:
    def __init__(self,name):
        self.name=name
        folder=DATA/name;folder.mkdir(exist_ok=True)
        url=os.getenv(f"{name.upper()}_DATABASE_URL",f"sqlite:///{(folder/'service.db').as_posix()}")
        if url.startswith("sqlite:///"):
            f=Path(url.removeprefix("sqlite:///")).resolve()
            if not f.is_relative_to(ROOT): raise RuntimeError("SQLite path outside project")
        self.engine=create_engine(url,connect_args={"check_same_thread":False,"timeout":30} if url.startswith("sqlite") else {},pool_pre_ping=True)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine,"connect")
            def secure_delete(dbapi_connection,record):
                dbapi_connection.execute("PRAGMA secure_delete=ON")
        from common.migrations import migrate
        migrate(self.engine,name,Base.metadata)
        STORES[name]=self
    @contextmanager
    def tx(self):
        with Session(self.engine) as s:
            try:
                with s.begin(): yield s
            except IntegrityError: fail(409,"RESOURCE_CONFLICT")
    def get(self,id,org=None,kind=None,s=None,expired=False):
        if s is None:
            with self.tx() as ss:return self.get(id,org,kind,ss,expired)
        r=s.get(Aggregate,id)
        if not r or (org is not None and r.org!=org) or (kind and r.kind!=kind): fail(404,"NOT_FOUND")
        if not expired and r.expires and r.expires<=now(): fail(410,"DATA_EXPIRED")
        return {"id":r.id,"org_id":r.org,"kind":r.kind,"version":r.version,"created":r.created,"expires":r.expires,**r.payload}
    def list(self,org,kind,s=None):
        if s is None:
            with self.tx() as ss:return self.list(org,kind,ss)
        q=select(Aggregate).where(Aggregate.kind==kind)
        if org is not None:q=q.where(Aggregate.org==org)
        return [{"id":r.id,"org_id":r.org,"version":r.version,"created":r.created,"expires":r.expires,**r.payload}
                for r in s.scalars(q) if r.expires is None or r.expires>now()]
    def create(self,id,org,kind,data,expires=None,s=None):
        if s is None:
            with self.tx() as ss:return self.create(id,org,kind,data,expires,ss)
        if s.get(Aggregate,id):fail(409,"RESOURCE_CONFLICT")
        s.add(Aggregate(id=id,org=org,kind=kind,payload=data,version=1,expires=expires))
        s.flush()
        return self.get(id,org,kind,s)
    def change(self,id,org,data,version,s=None):
        if s is None:
            with self.tx() as ss:return self.change(id,org,data,version,ss)
        r=s.get(Aggregate,id)
        if not r or r.org!=org:fail(404,"NOT_FOUND")
        result=s.execute(update(Aggregate).where(Aggregate.id==id,Aggregate.org==org,Aggregate.version==version).values(payload=data,version=version+1))
        if result.rowcount!=1:fail(409,"VERSION_CONFLICT",current_version=r.version)
        s.expire_all()
        return self.get(id,org,s=s)
    def delete(self,id,org,s=None):
        if s is None:
            with self.tx() as ss:return self.delete(id,org,ss)
        r=s.get(Aggregate,id)
        if r and r.org==org:s.delete(r)
    def set_expiry(self,id,org,expires,s):
        result=s.execute(update(Aggregate).where(Aggregate.id==id,Aggregate.org==org).values(expires=expires))
        if result.rowcount!=1:fail(404,"NOT_FOUND")
        s.expire_all()
    def audit(self,p,action,target,s=None):
        return self.create(uid(),p["org_id"],"audit",{"action":action,"actor_hash":digest(p["sub"]),"target_hash":digest(target),"at":iso()},now()+180*86400,s)

def app_for(name):
    app=FastAPI(title=f"RoadWatch {name}",version="0.1.0",docs_url=None,redoc_url=None)
    from services.contracts import Envelope,ErrorEnvelope
    class PublicContractRoute(APIRoute):
        def __init__(self,path,*args,**kwargs):
            if path.startswith("/api/v1/"):
                binary=path.startswith("/api/v1/images/") and path.endswith("/content") or path.startswith("/api/v1/reports/") and path.endswith("/download")
                if binary:
                    kwargs["response_class"]=Response
                    kwargs["response_model"]=None
                elif (kwargs.get("response_model") is None or isinstance(kwargs.get("response_model"),DefaultPlaceholder)) and kwargs.get("status_code")!=204:
                    from services.response_contracts import get_response_model
                    kwargs["response_model"]=get_response_model(path,kwargs.get("methods")) or Envelope
                responses=kwargs.setdefault("responses",{})
                if binary:
                    types=("image/jpeg","image/png") if path.startswith("/api/v1/images/") else ("text/csv","application/pdf")
                    responses[200]={"description":"Private evidence download","content":{t:{"schema":{"type":"string","format":"binary"}} for t in types}}
                for status in (401,403,404,409,410,413,422,428,503):responses.setdefault(status,{"model":ErrorEnvelope,"description":"Structured domain error"})
            super().__init__(path,*args,**kwargs)
    app.router.route_class=PublicContractRoute
    if name in ("identity","analysis"):
        async def expire():
            while True:
                try:
                    with STORES[name].tx() as session:session.execute(delete(Aggregate).where(Aggregate.expires<=now()))
                except Exception:log.exception("%s retention worker failed",name)
                await asyncio.sleep(60)
        @app.on_event("startup")
        async def start_expiry():app.state.expiry_task=asyncio.create_task(expire())
        @app.on_event("shutdown")
        async def stop_expiry():
            task=getattr(app.state,"expiry_task",None)
            if task:
                task.cancel()
                try:await task
                except asyncio.CancelledError:pass
    @app.exception_handler(StarletteHTTPException)
    async def he(req,exc):
        e=exc.detail if isinstance(exc.detail,dict) else {"code":{404:"NOT_FOUND",405:"METHOD_NOT_ALLOWED",422:"VALIDATION_ERROR"}.get(exc.status_code,str(exc.detail)),"message":str(exc.detail)}
        return JSONResponse({"error":e,"meta":{"request_id":getattr(req.state,"request_id","")}},status_code=exc.status_code,headers=exc.headers)
    @app.exception_handler(RequestValidationError)
    async def ve(req,exc):
        return JSONResponse({"error":{"code":"VALIDATION_ERROR","message":"Invalid request","details":[{"field_path":".".join(map(str,x["loc"])),"code":x["type"]} for x in exc.errors()]},"meta":{"request_id":getattr(req.state,"request_id","")}},status_code=422)
    @app.exception_handler(ResponseValidationError)
    async def response_invalid(req,exc):
        fields=[{"field_path":".".join(map(str,x["loc"])),"code":x["type"]} for x in exc.errors()]
        log.error("%s response schema failure fields=%s",name,fields)
        return JSONResponse({"error":{"code":"INTERNAL_ERROR","message":"Unexpected server response"},"meta":{"request_id":getattr(req.state,"request_id","")}},status_code=500)
    @app.middleware("http")
    async def guard(req,call_next):
        req.state.request_id=uid()
        context_token=request_id_context.set(req.state.request_id)
        def reject(status,code,message):
            request_id_context.reset(context_token)
            return JSONResponse({"error":{"code":code,"message":message},"meta":{"request_id":req.state.request_id}},status_code=status,headers={"X-Request-ID":req.state.request_id})
        try:
            public=req.url.path=="/health" or (name=="identity" and req.url.path=="/api/v1/auth/login")
            if not public:
                value=req.headers.get("X-Road-Principal","")
                raw,sig=value.rsplit(".",1)
                expected=hmac.new(service_key(name),raw.encode(),"sha256").hexdigest()
                if not hmac.compare_digest(sig,expected):raise ValueError()
                p=json.loads(base64.urlsafe_b64decode(raw))
                if p["aud"]!=name or p["exp"]<now():raise ValueError()
                req.state.principal=p
        except (ValueError,KeyError,TypeError,json.JSONDecodeError):
            return reject(401,"UNAUTHENTICATED","Internal authentication required")
        if req.method in ("POST","PUT","PATCH") and req.headers.get("content-type","").split(";")[0]=="application/json":
            raw=await req.body()
            if len(raw)>64000:
                return reject(413,"PAYLOAD_TOO_LARGE","JSON exceeds 64000 bytes")
            try:
                body=json.loads(raw,object_pairs_hook=strict_pairs,parse_float=strict_float,parse_int=strict_int,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                if not isinstance(body,dict):raise ValueError("Object required")
            except (ValueError,UnicodeDecodeError):
                return reject(422,"VALIDATION_ERROR","A valid JSON object is required")
        try:
            response=await call_next(req)
        except Exception:
            log.exception("%s unhandled request %s",name,req.state.request_id)
            response=JSONResponse({"error":{"code":"INTERNAL_ERROR","message":"Unexpected server error"},"meta":{"request_id":req.state.request_id}},status_code=500)
        finally:request_id_context.reset(context_token)
        response.headers["X-Request-ID"]=req.state.request_id
        response.headers["X-Content-Type-Options"]="nosniff"
        if "content" in req.url.path or "download" in req.url.path:response.headers["Cache-Control"]="private, no-store"
        return response
    @app.get("/health")
    def health():return {"status":"ok","service":name,"instance_id":os.getenv("ROAD_INSTANCE_ID",""),"pid":os.getpid()}
    @app.get("/internal/audit")
    def audit_export(req:Request,limit:int=1000):
        p=principal(req);role(p,"admin")
        if not 1<=limit<=5000:fail(422,"INVALID_LIMIT")
        rows=sorted(STORES[name].list(p["org_id"],"audit"),key=lambda x:(x["at"],x["id"]),reverse=True)[:limit]
        return ok({"items":[{k:x[k] for k in ("id","action","actor_hash","target_hash","at")} for x in rows]})
    return app
