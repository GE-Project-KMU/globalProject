from services.contracts import UUIDString,TrueBoolean

import math
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator,field_validator
from common.runtime import valid_uuid,stamp,iso,fail
class Strict(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True,allow_inf_nan=False)
class Location(Strict):
    lat:float|None;lon:float|None
    source:Literal["vehicle_observation"]
    quality:Literal["valid","degraded","unavailable"]
    hdop:float|None=Field(default=None,ge=0)
    age_ms:int|None=Field(default=None,ge=0)
    accuracy_m:float|None=Field(default=None,ge=0)
    @model_validator(mode="after")
    def check(self):
        if (self.lat is None)!=(self.lon is None):raise ValueError("Coordinates must be paired")
        if self.lat is not None and not (-90<=self.lat<=90 and -180<=self.lon<=180):raise ValueError("Coordinate range")
        if self.quality=="unavailable" and self.lat is not None:raise ValueError("Unavailable needs null coordinates")
        if self.quality=="valid" and self.lat is None:raise ValueError("Valid needs coordinates")
        return self
class Pothole(Strict):
    bbox_xywh_px:list[float]=Field(min_length=4,max_length=4)
    confidence:float=Field(ge=0,le=1)
    track_observations:int=Field(ge=1)
    candidate_state:Literal["multi_frame","single_frame_review","held"]
    occluded:bool
    @model_validator(mode="after")
    def check(self):
        x,y,w,h=self.bbox_xywh_px
        if min(x,y)<0 or min(w,h)<=0:raise ValueError("Invalid box")
        if self.candidate_state=="multi_frame" and self.track_observations<3:raise ValueError("Track needs >=3 observations")
        if self.candidate_state=="single_frame_review" and (self.track_observations>2 or self.confidence<.8):raise ValueError("Invalid single frame")
        return self
class PaintMask(Strict):
    size:list[int]=Field(min_length=2,max_length=2)
    counts:list[int]=Field(min_length=1,max_length=8192)
    @model_validator(mode="after")
    def check(self):
        if any(x<1 or x>512 for x in self.size) or any(x<0 for x in self.counts) or sum(self.counts)!=self.size[0]*self.size[1]:raise ValueError("Invalid row-major binary RLE")
        return self
class Lane(Strict):
    segment_id:UUIDString;estimated_length_m:float|None=Field(gt=0)
    retention_pct:float|None=Field(ge=0,le=100)
    grade:Literal["visible_good","monitor","worn_candidate","unassessable"]
    color:Literal["white","yellow","unknown"];line_type:Literal["solid","dashed","unknown"]
    valid_area_ratio:float|None=Field(ge=0,le=1)
    segment_support_ratio:float|None=Field(ge=0,le=1)
    outline_supported:bool
    patch_ids:list[UUIDString]=Field(max_length=200)
    unassessable_reason:str|None=Field(max_length=200)
    paint_mask_rle:PaintMask|None=None
    outline_polygon_px:list[list[float]]|None=Field(default=None,min_length=3,max_length=64)
    @model_validator(mode="after")
    def check(self):
        valid_uuid(self.segment_id)
        if self.outline_polygon_px is not None and any(len(pt)!=2 or any(x<0 for x in pt) for pt in self.outline_polygon_px):raise ValueError("Invalid outline polygon")
        for x in self.patch_ids:valid_uuid(x)
        if len(self.patch_ids)!=len(set(self.patch_ids)):raise ValueError("Duplicate patches")
        if self.grade=="unassessable":
            if self.retention_pct is not None or not self.unassessable_reason:raise ValueError("Unassessable requires null R and reason")
        else:
            if not self.outline_supported or self.valid_area_ratio is None or self.valid_area_ratio<.7 or self.segment_support_ratio is None or self.segment_support_ratio<.8 or self.estimated_length_m is None or self.color=="unknown" or self.line_type=="unknown" or self.retention_pct is None or not self.patch_ids or self.unassessable_reason is not None:raise ValueError("Missing lane evidence")
            expected="visible_good" if self.retention_pct>=80 else "monitor" if self.retention_pct>=60 else "worn_candidate"
            if expected!=self.grade:raise ValueError("Grade boundary mismatch")
        return self
class Quality(Strict):
    usable:bool
    reasons:list[str]=Field(max_length=20)
    @model_validator(mode="after")
    def check(self):
        if not self.usable and not self.reasons:raise ValueError("Reason required")
        if any(not 1<=len(x)<=100 for x in self.reasons):raise ValueError("Invalid reason")
        return self
class ImageRef(Strict):
    file_id:UUIDString;sha256:str=Field(pattern=r"^[a-f0-9]{64}$")
    redacted:TrueBoolean;reviewed_for_sharing:TrueBoolean
class InitialReview(Strict):
    status:Literal["unreviewed"];priority:Literal["normal"]
class Observation(Strict):
    schema_version:Literal["0.4"]
    observation_id:UUIDString;event_id:UUIDString;run_id:UUIDString;device_id:UUIDString
    type:Literal["pothole","lane_segment"]
    captured_at:str
    frame_id:int=Field(ge=0,le=9007199254740991)
    location:Location
    profile_id:UUIDString;model_version:str=Field(min_length=1,max_length=150)
    config_hash:str=Field(pattern=r"^[a-f0-9]{64}$")
    pothole:Pothole|None;lane:Lane|None
    quality:Quality;image:ImageRef;review:InitialReview
    demo:bool=False
    @model_validator(mode="after")
    def check(self):
        for f in ("observation_id","event_id","run_id","device_id","profile_id"):valid_uuid(getattr(self,f))
        valid_uuid(self.image.file_id)
        self.captured_at=iso(stamp(self.captured_at))
        if (self.type=="pothole" and (self.pothole is None or self.lane is not None)) or (self.type=="lane_segment" and (self.lane is None or self.pothole is not None)):raise ValueError("Mutually exclusive details")
        return self
class ReviewPatch(Strict):
    expected_case_id:UUIDString
    expected_version:int=Field(ge=1)
    status:Literal["unreviewed","confirmed","false_positive","recapture","action_planned","completed"]|None=None
    priority:Literal["urgent","normal","low"]|None=None
    reason:str=Field(min_length=1,max_length=500)
    override_transition:bool=False
    @model_validator(mode="before")
    @classmethod
    def version_required(cls,value):
        if isinstance(value,dict) and "expected_version" not in value:fail(428,"VERSION_REQUIRED")
        return value
    @model_validator(mode="after")
    def check(self):
        valid_uuid(self.expected_case_id)
        if self.status is None and self.priority is None:raise ValueError("Change required")
        return self
