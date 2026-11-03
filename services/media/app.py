
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
