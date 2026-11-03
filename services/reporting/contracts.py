"""Public reporting response contracts, shared by OpenAPI and runtime validation."""
from typing import Generic, Literal, TypeVar
from pydantic import BaseModel, ConfigDict

T=TypeVar("T")
class Envelope(BaseModel,Generic[T]):
    data:T
    meta:dict

class JobView(BaseModel):
    job_id:str
    status:Literal["pending","running","succeeded","failed"]
    progress:int
    error_code:str|None=None
    status_url:str
    download_path:str|None=None
    expires_at:str|None=None

class PrivacyView(BaseModel):
    request_id:str
    job_id:str
    version:int
    request_type:Literal["access","correction","deletion"]
    status:Literal["received","in_progress","completed","rejected","failed"]
    result_code:str|None=None
    created_at:str
    resolved_at:str|None=None
    target_refs:list[dict]
    reason:str
    decision_reason:str|None=None
    actions:list[dict]
    download_path:str|None=None

class PrivacyList(BaseModel):
    items:list[PrivacyView]

class Receipt(BaseModel):
    request_id:str
    job_id:str
    status:Literal["received"]

class AccessExport(BaseModel):
    observations:list[dict]
    runs:list[dict]
