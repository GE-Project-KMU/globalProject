"""Strict public write contracts shared by domain OpenAPI and validation."""
from typing import Literal,Annotated
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator,AfterValidator

def _uuid(value:str)->str:
    return str(UUID(value))

UUIDString=Annotated[str,AfterValidator(_uuid),Field(json_schema_extra={"format":"uuid"})]
def _true(value:bool)->bool:
    if value is not True:raise ValueError("Must be true")
    return value
TrueBoolean=Annotated[bool,Field(strict=True,json_schema_extra={"const":True}),AfterValidator(_true)]

class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

class UserCreate(Contract):
    login: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=12, max_length=128)
    role: Literal["admin", "reviewer", "viewer"]

class UserPatch(Contract):
    role: Literal["admin", "reviewer", "viewer"] | None = None
    active: bool | None = None
    password: str | None = Field(default=None, min_length=12, max_length=128)

class DeviceCreate(Contract):
    name: str = Field(min_length=1, max_length=100)

class DevicePatch(Contract):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    active: bool | None = None

class Reason(Contract):
    reason: str = Field(min_length=1, max_length=500)

class Calibration(Contract):
    image_size: list[int] = Field(min_length=2, max_length=2)
    height_m: float = Field(gt=0)
    pitch_deg: float
    camera_matrix: list[list[float]]
    distortion: list[float]
    homography: list[list[float]]
    roi_polygon_px: list[list[float]]
    visible_distance_m: list[float] = Field(min_length=2, max_length=2)
    calibrated_at: str = Field(json_schema_extra={"format": "date-time"})

class ProfileCreate(Contract):
    name: str = Field(min_length=1, max_length=100)
    version: int = Field(ge=1)
    camera_serial: str = Field(min_length=1, max_length=150)
    lens_label: str = Field(min_length=1, max_length=150)
    calibration: Calibration
    config_hash: str = Field(pattern="^[a-f0-9]{64}$")

class Weights(Contract):
    pothole: str = Field(pattern="^[a-f0-9]{64}$")
    lane: str = Field(pattern="^[a-f0-9]{64}$")
    redaction: str = Field(pattern="^[a-f0-9]{64}$")

class Thresholds(Contract):
    pothole_confidence: float = Field(ge=0, le=1)
    pothole_nms_iou: float = Field(ge=0, le=1)
    single_frame_confidence: float = Field(ge=0, le=1)
    lane_monitor: float = Field(ge=0, le=100)
    lane_good: float = Field(ge=0, le=100)
    min_valid_ratio: float = Field(ge=0, le=1)
    min_segment_support_ratio: float = Field(ge=0, le=1)

class BundleCreate(Contract):
    version: str = Field(min_length=1, max_length=150)
    weights_hashes: Weights
    preprocess: dict
    thresholds: Thresholds
    config_hash: str = Field(pattern="^[a-f0-9]{64}$")
    code_commit: str = Field(min_length=1, max_length=150)
    environment_hash: str = Field(pattern="^[a-f0-9]{64}$")

class Precheck(Contract):
    camera_preview_ok: TrueBoolean
    mount_secure: TrueBoolean
    lens_clear: TrueBoolean
    profile_valid: TrueBoolean
    gnss_device_connected: TrueBoolean
    storage_ge_10gb: TrueBoolean
    notice_present: TrueBoolean

class RunCreate(Contract):
    run_id: UUIDString
    device_id: UUIDString
    profile_id: UUIDString
    started_at: str = Field(json_schema_extra={"format": "date-time"})
    manifest_hash: str = Field(pattern="^[a-f0-9]{64}$")
    precheck: Precheck

class RunEnd(Contract):
    ended_at: str = Field(json_schema_extra={"format": "date-time"})
    status: Literal["completed", "interrupted"]
    manifest_hash: str = Field(pattern="^[a-f0-9]{64}$")
    summary: dict | None = None

class VisibleScope(Contract):
    profile_id:UUIDString
    roi_id:str=Field(min_length=1,max_length=100)
    assessment_state:Literal["assessed","unassessable"]
    reason:str|None=Field(default=None,max_length=500)
    @model_validator(mode="after")
    def check(self):
        if self.assessment_state=="unassessable" and not self.reason:raise ValueError("Reason required")
        return self

class PathSegment(Contract):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, populate_by_name=True)
    from_: str = Field(alias="from", json_schema_extra={"format": "date-time"})
    to: str = Field(json_schema_extra={"format": "date-time"})
    coordinates: list[list[float]] = Field(min_length=2, max_length=1000)
    quality:Literal["valid","degraded","unavailable"]
    visible_scope:VisibleScope

class RunPath(Contract):
    segments: list[PathSegment] = Field(max_length=500)
    manifest_hash: str = Field(pattern="^[a-f0-9]{64}$")

class DuplicateCreate(Reason):
    source_observation_id: UUIDString
    target_observation_id: UUIDString
    decision: Literal["suggested", "approved", "rejected"]
    source_version: int = Field(ge=1)
    target_version: int = Field(ge=1)

class DuplicateRevoke(Reason):
    decision: Literal["revoked"]
    expected_version: int = Field(ge=1)

class Envelope(BaseModel):
    data: dict | list | None
    meta: dict

class ErrorEnvelope(BaseModel):
    error: dict
    meta: dict
