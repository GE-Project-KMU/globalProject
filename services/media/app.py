
import io, json, re, os, asyncio
from contextlib import asynccontextmanager
from PIL import Image, UnidentifiedImageError
from fastapi import Request, UploadFile, File, Form, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic import ValidationError
from services.contracts import UUIDString
import functools
import os
import psutil
from common.runtime import *
store=Store("media");app=app_for("media")
FILES=DATA/"media"/"files";FILES.mkdir(exist_ok=True)
class Approval(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True)
    file_id:str;device_id:str
    sha256:str=Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes:int=Field(gt=0,le=5000000)
    mime:str
    captured_at:str;reviewed_at:str
    redaction_model_version:str=Field(min_length=1,max_length=150)
    @field_validator("file_id","device_id")
    @classmethod
    def ids(cls,v):return valid_uuid(v)
    @field_validator("mime")
    @classmethod
    def mime_check(cls,v):
        if v not in ("image/jpeg","image/png"):raise ValueError("JPEG/PNG required")
        return v
class UploadMetadata(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True)
    file_id:str;approval_id:str
    sha256:str=Field(pattern=r"^[a-f0-9]{64}$")
    captured_at:str
    @field_validator("file_id","approval_id")
    @classmethod
    def identifiers(cls,v):return valid_uuid(v)

def difference_hash(image):
    pixels=list(image.convert("L").resize((9,8)).getdata())
    value=0
    for y in range(8):
        for x in range(8):value=(value<<1)|int(pixels[y*9+x]>pixels[y*9+x+1])
    return f"{value:016x}"

def serialized_image(function):
    """Cross-process reservation prevents file/DB races for one immutable file ID."""
    @functools.wraps(function)
    async def locked(*args,**kwargs):
        req=kwargs["req"];role(principal(req),"device")
        try:identifier=UploadMetadata.model_validate(json.loads(kwargs["metadata"],object_pairs_hook=strict_pairs)).file_id
        except (ValueError,TypeError,ValidationError):fail(422,"INVALID_METADATA")
        locks=DATA/"media"/"locks";locks.mkdir(exist_ok=True)
        lock=locks/identifier;owner=json.dumps({"token":uid(),"pid":os.getpid(),"created":psutil.Process().create_time()});deadline=now()+15
        while True:
            try:
                fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
                with os.fdopen(fd,"w") as stream:stream.write(owner)
                break
            except FileExistsError:
                try:
                    if lock.stat().st_mtime<now()-120:
                        try:
                            previous=json.loads(lock.read_text());process=psutil.Process(previous["pid"])
                            alive=process.is_running() and abs(process.create_time()-previous["created"])<.1
                        except psutil.AccessDenied:alive=True
                        except (ValueError,KeyError,TypeError,psutil.NoSuchProcess):alive=False
                        if not alive:lock.unlink(missing_ok=True)
                except FileNotFoundError:pass
                if now()>=deadline:raise HTTPException(503,{"code":"UPLOAD_IN_PROGRESS","message":"Retry this immutable file ID"},headers={"Retry-After":"1"})
                await asyncio.sleep(.05)
        try:
            with store.tx() as s:
                row=s.get(Aggregate,identifier)
                if row is not None and (row.org!=principal(req)["org_id"] or row.kind!="image"):fail(404,"NOT_FOUND")
            return await function(*args,**kwargs)
        finally:
            try:
                if lock.read_text()==owner:lock.unlink(missing_ok=True)
            except FileNotFoundError:pass
    return locked
@app.post("/api/v1/image-approvals",status_code=201)
async def approve(b:Approval,req:Request):
    p=principal(req);role(p,"admin","reviewer")
    try:
        store.get(digest("image:"+b.file_id),p["org_id"],"tombstone")
        fail(410,"DATA_EXPIRED")
    except HTTPException as e:
        if e.status_code!=404:raise
    await remote("identity","/internal/devices/"+b.device_id,p)
    captured=stamp(b.captured_at);reviewed=stamp(b.reviewed_at)
    if captured>now()+2 or reviewed<captured or reviewed>now()+2:fail(422,"INVALID_TIME")
    if captured+90*86400<=now():fail(410,"DATA_EXPIRED")
    id=uid();store.create(id,p["org_id"],"approval",{**b.model_dump(),"reviewer_id":p["sub"],"approved_at":iso(),"approval_expires_at":now()+900,"used":False},captured+90*86400)
    store.audit(p,"image.approve",id)
    return ok({"approval_id":id,"expires_at":iso(now()+900)})
@app.post("/api/v1/images",status_code=201)
@serialized_image
async def upload(req:Request,response:Response,file:UploadFile=File(...),metadata:str=Form(...)):
    p=principal(req);role(p,"device")
    try:
        b=UploadMetadata.model_validate(json.loads(metadata,object_pairs_hook=strict_pairs)).model_dump()
        id=b["file_id"]
    except (ValueError,TypeError,ValidationError):fail(422,"INVALID_METADATA")
    try:
        store.get(digest("image:"+id),p["org_id"],"tombstone")
        fail(410,"DATA_EXPIRED")
    except HTTPException as e:
        if e.status_code!=404:raise
    content=await file.read(5000001)
    if len(content)>5000000:fail(413,"PAYLOAD_TOO_LARGE")
    try:
        im=Image.open(io.BytesIO(content))
        if im.width>4096 or im.height>4096:fail(413,"IMAGE_DIMENSIONS_EXCEEDED")
        width,height=im.size;fmt=im.format
        im.verify()
        with Image.open(io.BytesIO(content)) as decoded:
            if decoded.getexif() or getattr(decoded,"text",{}) or any(k.lower() in ("exif","xmp","xml:com.adobe.xmp","comment","description") for k in decoded.info):fail(422,"IMAGE_METADATA_NOT_STRIPPED")
            perceptual_hash=difference_hash(decoded)
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError):fail(415,"UNSUPPORTED_MEDIA")
    mime={"PNG":"image/png","JPEG":"image/jpeg"}.get(fmt)
    if not mime or file.content_type!=mime:fail(415,"UNSUPPORTED_MEDIA")
    sha=digest(content)
    if sha!=b["sha256"]:fail(422,"HASH_MISMATCH")
    try:
        old=store.get(id,p["org_id"],"image")
        if old["state"]!="ready":fail(410,"DATA_RESTRICTED")
        if old["sha256"]!=sha or old["device_id"]!=p["sub"] or old["captured_at"]!=b["captured_at"]:fail(409,"IDEMPOTENCY_CONFLICT")
        response.status_code=200
        return ok({"image_id":id,"state":"ready","sha256":sha,"expires_at":iso(old["expires"]),"deduplicated":True})
    except HTTPException as e:
        if e.status_code!=404:raise
    approval=store.get(b["approval_id"],p["org_id"],"approval")
    if approval["device_id"]!=p["sub"]:fail(403,"FORBIDDEN")
    if approval["approval_expires_at"]<now() or approval["used"]:fail(409,"APPROVAL_EXPIRED")
    if any(approval[k]!=b[k] for k in ("file_id","sha256","captured_at")) or approval["size_bytes"]!=len(content) or approval["mime"]!=mime:fail(422,"APPROVAL_MISMATCH")
    path=FILES/id;tmp=FILES/(id+"."+uid()+".tmp");tmp.write_bytes(content);tmp.replace(path)
    expires=stamp(b["captured_at"])+90*86400
    with store.tx() as s:
        a=store.get(b["approval_id"],p["org_id"],"approval",s)
        if a["used"]:fail(409,"APPROVAL_ALREADY_USED")
        payload={k:v for k,v in a.items() if k not in ("id","org_id","kind","version","created","expires")};payload["used"]=True
        store.change(a["id"],p["org_id"],payload,a["version"],s)
        store.create(id,p["org_id"],"image",{"sha256":sha,"perceptual_hash":perceptual_hash,"device_id":p["sub"],"captured_at":b["captured_at"],"mime":mime,"width":width,"height":height,"state":"ready","approval_id":a["id"]},expires,s)
    return ok({"image_id":id,"state":"ready","sha256":sha,"expires_at":iso(expires),"deduplicated":False})
@app.get("/internal/images/{id}")
def verify(id:UUIDString,req:Request):
    p=principal(req);valid_uuid(id)
    try:
        store.get(digest("image:"+id),p["org_id"],"tombstone")
        fail(410,"DATA_EXPIRED")
    except HTTPException as e:
        if e.status_code!=404:raise
    x=store.get(id,p["org_id"],"image")
    if p["role"]=="device" and x["device_id"]!=p["sub"]:fail(403,"FORBIDDEN")
    if x["state"]!="ready":fail(410,"DATA_EXPIRED")
    return ok(x)
@app.get("/api/v1/images/{id}/content")
def content(id:UUIDString,req:Request):
    p=principal(req);role(p,"admin","reviewer","viewer");x=verify(id,req)["data"]
    path=FILES/valid_uuid(id)
    if not path.is_file():fail(404,"FILE_NOT_FOUND")
    store.audit(p,"image.view",id)
    return FileResponse(path,media_type=x["mime"],headers={"Cache-Control":"private, no-store"})
@app.post("/internal/privacy/delete")
async def delete(req:Request):
    p=principal(req);role(p,"admin");b=await req.json();ids=b.get("image_ids",[])
    deleted=[]
    for id in ids:
        id=valid_uuid(id)
        try:store.get(id,p["org_id"],"image",expired=True)
        except HTTPException as e:
            if e.status_code==404:continue
            raise
        (FILES/id).unlink(missing_ok=True)
        with store.tx() as s:
            tid=digest("image:"+id)
            try:store.get(tid,p["org_id"],"tombstone",s)
            except HTTPException:store.create(tid,p["org_id"],"tombstone",{"reason":"privacy_deletion"},now()+180*86400,s)
            store.delete(id,p["org_id"],s)
            for approval in s.scalars(select(Aggregate).where(Aggregate.org==p["org_id"],Aggregate.kind=="approval")):
                if approval.payload.get("file_id")==id:s.delete(approval)
        deleted.append(id)
    return ok({"deleted":deleted})

@app.post("/internal/privacy/block")
async def block(req:Request):
    p=principal(req);role(p,"admin");b=await req.json()
    with store.tx() as s:
        for id in b.get("image_ids",[]):
            x=store.get(valid_uuid(id),p["org_id"],"image",s,expired=True)
            data={k:v for k,v in x.items() if k not in ("id","org_id","kind","created","version","expires")}
            data["state"]="blocked"
            store.change(id,p["org_id"],data,x["version"],s)
    return ok({"blocked":len(b.get("image_ids",[]))})

@app.post("/internal/privacy/verify")
async def verify_deletion(req:Request):
    p=principal(req);role(p,"admin");b=await req.json();ids={valid_uuid(x) for x in b.get("image_ids",[])}
    with store.tx() as s:
        images=list(s.scalars(select(Aggregate).where(Aggregate.org==p["org_id"],Aggregate.kind=="image",Aggregate.id.in_(ids)))) if ids else []
        approvals=[r for r in s.scalars(select(Aggregate).where(Aggregate.org==p["org_id"],Aggregate.kind=="approval")) if r.payload.get("file_id") in ids]
    remaining_files=sum((FILES/id).exists() for id in ids)
    return ok({"verified":not images and not approvals and not remaining_files,"remaining_images":len(images),"remaining_approvals":len(approvals),"remaining_files":remaining_files})
@app.post("/internal/retention")
def retention(req:Request):
    p=principal(req);role(p,"admin")
    with store.tx() as s:
        rows=list(s.scalars(select(Aggregate).where(Aggregate.org==p["org_id"],Aggregate.expires<=now())))
        for row in rows:
            if row.kind=="image":(FILES/row.id).unlink(missing_ok=True)
            s.delete(row)
    return ok({"deleted":len(rows)})

def purge_expired_media():
    with store.tx() as s:
        rows=list(s.scalars(select(Aggregate).where(Aggregate.expires<=now())))
        for row in rows:
            if row.kind=="image":
                (FILES/row.id).unlink(missing_ok=True)
                tid=digest("image:"+row.id)
                if s.get(Aggregate,tid) is None:store.create(tid,row.org,"tombstone",{"reason":"ttl_expiry"},now()+180*86400,s)
            s.delete(row)
    # Orphans are never externally readable, remove stale staged files.
    for f in FILES.iterdir():
        if f.is_file() and f.stat().st_mtime<now()-86400:
            with store.tx() as s:
                if s.get(Aggregate,f.name) is None:f.unlink(missing_ok=True)
async def retention_worker():
    while True:
        try:purge_expired_media()
        except asyncio.CancelledError:raise
        except Exception:log.exception("media retention cycle failed")
        await asyncio.sleep(30)
@asynccontextmanager
async def lifespan(app):
    task=asyncio.create_task(retention_worker())
    try:yield
    finally:
        task.cancel()
        try:await task
        except asyncio.CancelledError:pass
app.router.lifespan_context=lifespan
