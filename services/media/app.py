
import io, json, re, os
from PIL import Image, UnidentifiedImageError
from fastapi import Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator
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
@app.post("/api/v1/image-approvals",status_code=201)
async def approve(b:Approval,req:Request):
    p=principal(req);role(p,"admin","reviewer")
    await remote("identity","/internal/devices/"+b.device_id,p)
    captured=stamp(b.captured_at);reviewed=stamp(b.reviewed_at)
    if captured>now()+2 or reviewed<captured or reviewed>now()+2:fail(422,"INVALID_TIME")
    if captured+90*86400<=now():fail(410,"DATA_EXPIRED")
    id=uid();store.create(id,p["org_id"],"approval",{**b.model_dump(),"reviewer_id":p["sub"],"approved_at":iso(),"approval_expires_at":now()+900,"used":False},captured+90*86400)
    store.audit(p,"image.approve",id)
    return ok({"approval_id":id,"expires_at":iso(now()+900)})
@app.post("/api/v1/images",status_code=201)
async def upload(req:Request,file:UploadFile=File(...),metadata:str=Form(...)):
    p=principal(req);role(p,"device")
    try:
        b=json.loads(metadata)
        if set(b)!={"file_id","approval_id","sha256","captured_at"}:raise ValueError()
        id=valid_uuid(b["file_id"]);valid_uuid(b["approval_id"])
    except (ValueError,TypeError):fail(422,"INVALID_METADATA")
    content=await file.read(5000001)
    if len(content)>5000000:fail(413,"PAYLOAD_TOO_LARGE")
    try:
        im=Image.open(io.BytesIO(content))
        if im.width>4096 or im.height>4096:fail(413,"IMAGE_DIMENSIONS_EXCEEDED")
        width,height=im.size;fmt=im.format;im.verify()
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError):fail(415,"UNSUPPORTED_MEDIA")
    mime={"PNG":"image/png","JPEG":"image/jpeg"}.get(fmt)
    if not mime or file.content_type!=mime:fail(415,"UNSUPPORTED_MEDIA")
    sha=digest(content)
    if sha!=b["sha256"]:fail(422,"HASH_MISMATCH")
    try:
        old=store.get(id,p["org_id"],"image")
        if old["sha256"]!=sha or old["device_id"]!=p["sub"] or old["captured_at"]!=b["captured_at"]:fail(409,"IDEMPOTENCY_CONFLICT")
        return ok({"image_id":id,"state":"ready","sha256":sha,"expires_at":iso(old["expires"]),"deduplicated":True})
    except HTTPException as e:
        if e.status_code!=404:raise
    approval=store.get(b["approval_id"],p["org_id"],"approval")
    if approval["device_id"]!=p["sub"]:fail(403,"FORBIDDEN")
    if approval["approval_expires_at"]<now() or approval["used"]:fail(409,"APPROVAL_EXPIRED")
    if any(approval[k]!=b[k] for k in ("file_id","sha256","captured_at")) or approval["size_bytes"]!=len(content) or approval["mime"]!=mime:fail(422,"APPROVAL_MISMATCH")
    path=FILES/id;tmp=FILES/(id+".tmp");tmp.write_bytes(content);tmp.replace(path)
    expires=stamp(b["captured_at"])+90*86400
    with store.tx() as s:
        a=store.get(b["approval_id"],p["org_id"],"approval",s)
        if a["used"]:fail(409,"APPROVAL_ALREADY_USED")
        payload={k:v for k,v in a.items() if k not in ("id","org_id","kind","version","created","expires")};payload["used"]=True
        store.change(a["id"],p["org_id"],payload,a["version"],s)
        store.create(id,p["org_id"],"image",{"sha256":sha,"device_id":p["sub"],"captured_at":b["captured_at"],"mime":mime,"width":width,"height":height,"state":"ready","approval_id":a["id"]},expires,s)
    return ok({"image_id":id,"state":"ready","sha256":sha,"expires_at":iso(expires),"deduplicated":False})
