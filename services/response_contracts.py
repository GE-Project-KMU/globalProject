"""Typed public responses; additive fields remain compatible across v0.4 clients."""
from typing import Generic,Literal,TypeVar
from pydantic import BaseModel,ConfigDict
from services.contracts import Calibration,Weights,Thresholds,Precheck,PathSegment
from services.inspection.schemas import Location,Pothole,Lane,Quality,ImageRef

T=TypeVar("T")
class Public(BaseModel):
    model_config=ConfigDict(extra="allow")
class Envelope(Public,Generic[T]):
    data:T
    meta:dict
class Items(Public,Generic[T]):
    items:list[T]
    next_cursor:str|None=None
class User(Public):
    id:str
    login:str
    role:Literal["admin","reviewer","viewer"]
    active:bool|None=None
class Session(Public):
    access_token:str
    token_type:Literal["Bearer"]
    expires_in:int
    user:User
class Principal(Public):
    sub:str
    org_id:str
    role:str
    login:str|None=None
    device_id:str|None=None
class Device(Public):
    id:str
    name:str
    active:bool
    version:int|None=None
class DeviceCredential(Public):
    device_id:str
    device_token:str
    name:str|None=None
    active:bool|None=None
class Profile(Public):
    id:str|None=None
    profile_id:str|None=None
    name:str
    version:int
    camera_serial:str
    lens_label:str
    calibration:Calibration
    config_hash:str
    calibrated_by:str|None=None
    created_at:str
class Bundle(Public):
    id:str|None=None
    model_bundle_id:str|None=None
    version:str
    weights_hashes:Weights
    preprocess:dict
    thresholds:Thresholds
    config_hash:str
    code_commit:str|None=None
    environment_hash:str|None=None
    deployment_status:str|None=None
    created_at:str
class RunReceipt(Public):
    run_id:str
    status:Literal["recording","completed","interrupted"]
    manifest_hash:str
class Run(Public):
    id:str
    run_id:str
    device_id:str
    profile_id:str
    started_at:str
    ended_at:str|None=None
    status:Literal["recording","completed","interrupted"]
    manifest_hash:str
    precheck:Precheck
    segments:list[PathSegment]
    summary:dict|None=None
class Review(Public):
    status:Literal["unreviewed","confirmed","false_positive","recapture","action_planned","completed"]
    priority:Literal["urgent","normal","low"]
    reason:str|None=None
    updated_at:str|None=None
class Correction(Public):
    observation_id:str
    field:Literal["location","annotation"]
    value:dict|str
    reason:str
    corrected_at:str
    corrected_by:str
class ObservationView(Public):
    schema_version:Literal["0.4"]
    id:str
    observation_id:str
    event_id:str
    run_id:str
    device_id:str
    case_id:str
    case_version:int
    type:Literal["pothole","lane_segment"]
    captured_at:str
    frame_id:int
    location:Location
    profile_id:str
    model_version:str
    config_hash:str
    pothole:Pothole|None
    lane:Lane|None
    quality:Quality
    image:ImageRef
    review:Review
    corrections:list[Correction]
    expires_at:str
    image_content_path:str
    demo:bool=False
    data_origin:Literal['field','synthetic_demo','http_test']='field'
    deduplicated:bool|None=None
class ObservationList(Items[ObservationView]):
    counts_by_type:dict[str,int]
    counts_unit:Literal["observation"]
class ReviewResult(Public):
    case_id:str
    version:int
    review:Review
class HistoryEntry(Public):
    at:str
    actor_id:str
    case_id:str
    reason:str
    from_status:str|None=None
    to_status:str|None=None
    from_priority:str|None=None
    to_priority:str|None=None
    case_version:int|None=None
    action:str|None=None
class History(Public):
    entries:list[HistoryEntry]
    next_cursor:str|None=None
class Duplicate(Public):
    id:str
    source_case_id:str
    target_case_id:str
    source_observation_id:str
    target_observation_id:str
    status:Literal["suggested","approved","rejected","revoked"]
    version:int
    distance_m:float|None=None
    evidence:dict|None=None
    reason:str
    decided_at:str|None=None
    decided_by:str|None=None
class Trend(Public):
    date:str
    candidates:int
    confirmed:int
    completed:int
class Summary(Public):
    total:int
    pending:int
    completed:int
    potholes:int
    worn_lanes:int
    unassessable:int
    trend:list[Trend]
    trend_by_type:dict[str,list[Trend]]
    trend_counts_unit:Literal["case"]
    counts_unit:Literal["observation"]
    demo:bool
class ApprovalResult(Public):
    approval_id:str
    expires_at:str
class ImageResult(Public):
    image_id:str
    state:Literal["ready"]
    sha256:str
    expires_at:str
    deduplicated:bool
class AnalysisJob(Public):
    id:str
    run_id:str
    model_bundle_id:str
    status:Literal["queued","running","completed","blocked","failed"]
    created_at:str
    started_at:str|None=None
    finished_at:str|None=None
    observation_count:int|None=None
    reason:str|None=None
class Capabilities(Public):
    raw_processing:Literal["local_agent"]
    server_raw_upload:Literal[False]
    manifest_count:int
    model_status:str
    local_agent_command:str
    note:str

def get_response_model(path,methods):
    method=next(iter(methods or ["GET"]))
    model=None
    fixed={
        "/api/v1/auth/login":Session,"/api/v1/me":Principal,
        "/api/v1/summary":Summary,"/api/v1/image-approvals":ApprovalResult,
        "/api/v1/images":ImageResult,"/api/v1/analysis/capabilities":Capabilities}
    model=fixed.get(path)
    if path=="/api/v1/users":model=Items[User] if method=="GET" else User
    elif path=="/api/v1/users/{id}":model=User
    elif path=="/api/v1/devices":model=Items[Device] if method=="GET" else DeviceCredential
    elif path=="/api/v1/devices/{id}":model=Device
    elif path=="/api/v1/devices/{id}/credentials":model=DeviceCredential
    elif path=="/api/v1/profiles":model=Items[Profile] if method=="GET" else Profile
    elif path=="/api/v1/profiles/{id}":model=Profile
    elif path=="/api/v1/model-bundles":model=Items[Bundle] if method=="GET" else Bundle
    elif path=="/api/v1/model-bundles/{id}":model=Bundle
    elif path=="/api/v1/runs":model=Items[Run] if method=="GET" else RunReceipt
    elif path in ("/api/v1/runs/{id}","/api/v1/runs/{id}/path"):model=Run
    elif path=="/api/v1/observations":model=ObservationList if method=="GET" else ObservationView
    elif path=="/api/v1/observations/{id}":model=ObservationView
    elif path=="/api/v1/observations/{id}/review":model=ReviewResult
    elif path=="/api/v1/cases/{id}/history":model=History
    elif path=="/api/v1/duplicate-links":model=Items[Duplicate] if method=="GET" else Duplicate
    elif path=="/api/v1/duplicate-links/{id}":model=Duplicate
    elif path=="/api/v1/analysis/jobs":model=Items[AnalysisJob] if method=="GET" else AnalysisJob
    elif path=="/api/v1/analysis/jobs/{id}":model=AnalysisJob
    return Envelope[model] if model else None
