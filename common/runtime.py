"""Technical infrastructure only. Domain services never open one another's stores."""
import base64, hashlib, hmac, json, logging, os, secrets, time, uuid
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy import create_engine, String, Float, Integer, JSON, select, update
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
def ok(data, **meta): return {"data":data,"meta":{"schema_version":"0.4",**meta}}
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
class Store:
    def __init__(self,name):
        self.name=name
        folder=DATA/name;folder.mkdir(exist_ok=True)
        url=os.getenv(f"{name.upper()}_DATABASE_URL",f"sqlite:///{(folder/'service.db').as_posix()}")
        if url.startswith("sqlite:///"):
            f=Path(url.removeprefix("sqlite:///")).resolve()
            if not f.is_relative_to(ROOT): raise RuntimeError("SQLite path outside project")
        self.engine=create_engine(url,connect_args={"check_same_thread":False,"timeout":30} if url.startswith("sqlite") else {},pool_pre_ping=True)
        Base.metadata.create_all(self.engine)
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
    def audit(self,p,action,target,s=None):
        return self.create(uid(),p["org_id"],"audit",{"action":action,"actor_hash":digest(p["sub"]),"target_hash":digest(target),"at":iso()},now()+180*86400,s)

def app_for(name):
    app=FastAPI(title=f"RoadWatch {name}",version="0.1.0",docs_url=None,redoc_url=None)
    @app.exception_handler(HTTPException)
    async def he(req,exc):
        e=exc.detail if isinstance(exc.detail,dict) else {"code":str(exc.detail),"message":str(exc.detail)}
        return JSONResponse({"error":e,"meta":{"request_id":getattr(req.state,"request_id","")}},status_code=exc.status_code,headers=exc.headers)
    @app.exception_handler(RequestValidationError)
    async def ve(req,exc):
        return JSONResponse({"error":{"code":"VALIDATION_ERROR","message":"Invalid request","details":[{"field_path":".".join(map(str,x["loc"])),"code":x["type"]} for x in exc.errors()]},"meta":{"request_id":getattr(req.state,"request_id","")}},status_code=422)
    @app.middleware("http")
    async def guard(req,call_next):
        req.state.request_id=uid()
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
            return JSONResponse({"error":{"code":"UNAUTHENTICATED","message":"Internal authentication required"}},status_code=401)
        try:
            response=await call_next(req)
        except Exception:
            log.exception("%s unhandled request %s",name,req.state.request_id)
            return JSONResponse({"error":{"code":"INTERNAL_ERROR","message":"Unexpected server error"},"meta":{"request_id":req.state.request_id}},status_code=500)
        response.headers["X-Request-ID"]=req.state.request_id
        response.headers["X-Content-Type-Options"]="nosniff"
        if "content" in req.url.path or "download" in req.url.path:response.headers["Cache-Control"]="private, no-store"
        return response
    @app.get("/health")
    def health():return {"status":"ok","service":name,"instance_id":os.getenv("ROAD_INSTANCE_ID",""),"pid":os.getpid()}
    return app
