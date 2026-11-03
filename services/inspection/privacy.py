"""Restricted privacy actions; corrections are append-only and preserve evidence."""
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError
from typing import Literal
from common.runtime import *
from services.inspection.schemas import Location


class Correction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    observation_id: str
    field: Literal["location", "annotation"]
    value: dict | str

    @field_validator("observation_id")
    @classmethod
    def valid_id(cls, value):
        return valid_uuid(value)


def register_privacy(app, store, view):
    def selected(refs, p, s=None):
        obs = store.list(p["org_id"], "observation", s)
        result = {}
        runs = []
        for ref in refs:
            kind, identifier = ref["kind"], valid_uuid(ref["id"])
            if kind == "observation":
                result[identifier] = store.get(identifier, p["org_id"], "observation", s)
            elif kind in ("run", "case", "image"):
                if kind == "run":
                    run = store.get(identifier, p["org_id"], "run", s)
                    runs.append({k: v for k, v in run.items() if k not in ("org_id", "kind", "precheck", "path")})
                elif kind == "case":
                    seen=set()
                    while identifier not in seen:
                        seen.add(identifier);case=store.get(identifier,p["org_id"],"case",s)
                        if not case.get("merged_into"):break
                        identifier=case["merged_into"]
                for observation in obs:
                    key = observation["payload"]["image"]["file_id"] if kind == "image" else observation["payload"]["run_id"] if kind == "run" else observation["case_id"]
                    if kind=="case":
                        seen=set()
                        while key not in seen:
                            seen.add(key);case=store.get(key,p["org_id"],"case",s)
                            if not case.get("merged_into"):break
                            key=case["merged_into"]
                    if key == identifier:
                        result[observation["id"]] = observation
            else:
                fail(422, "INVALID_TARGET")
        return list(result.values()), runs

    @app.post("/internal/privacy/access")
    async def access(req: Request):
        p = principal(req); role(p, "admin")
        body = await req.json()
        observations, runs = selected(body["target_refs"], p)
        if len(observations) > 10000:
            fail(422, "EXPORT_TOO_LARGE")
        records = [view(o, p) for o in observations]
        return ok({"observations": records, "runs": runs,
                   "observation_ids": [o["id"] for o in observations],
                   "expires_at": iso(min([now() + 86400] + [o["expires"] for o in observations if o["expires"]] + [r["expires"] for r in runs if r.get("expires")]))})

    @app.post("/internal/privacy/correct")
    async def correct(req: Request):
        p = principal(req); role(p, "admin")
        b = await req.json(); request_id = valid_uuid(b["request_id"])
        allowed = {valid_uuid(x) for x in b["observation_ids"]}
        if not isinstance(b.get("corrections"), list) or not 1 <= len(b["corrections"]) <= 100:
            fail(422, "INVALID_CORRECTIONS")
        normalized = []
        for item in b["corrections"]:
            try: c = Correction.model_validate(item)
            except ValidationError: fail(422, "INVALID_CORRECTIONS")
            if c.observation_id not in allowed:
                fail(422, "CORRECTION_TARGET_MISMATCH")
            if c.field == "annotation":
                if not isinstance(c.value, str) or not 1 <= len(c.value.strip()) <= 500:
                    fail(422, "INVALID_CORRECTION_VALUE")
            else:
                if not isinstance(c.value, dict): fail(422, "INVALID_CORRECTION_VALUE")
                try: c.value = Location.model_validate({**c.value, "source": "vehicle_observation"}).model_dump()
                except ValidationError: fail(422, "INVALID_CORRECTION_VALUE")
            normalized.append(c.model_dump())
        with store.tx() as s:
            for index, c in enumerate(normalized):
                observation = store.get(c["observation_id"], p["org_id"], "observation", s)
                if observation.get("privacy_blocked"):fail(410,"DATA_RESTRICTED")
                identifier = digest(["correction", request_id, index])
                try:
                    previous = store.get(identifier, p["org_id"], "correction", s)
                    if any(previous[k] != c[k] for k in c): fail(409, "CORRECTION_CONFLICT")
                    continue
                except HTTPException as e:
                    if e.status_code != 404: raise
                store.create(identifier, p["org_id"], "correction", {**c, "request_id": request_id,
                             "reason": b["reason"], "corrected_at": iso(), "corrected_by": p["sub"]}, observation["expires"], s)
            store.audit(p, "privacy.correct", request_id, s)
        return ok({"applied": len(normalized), "immutable_original_preserved": True})

    @app.post("/internal/privacy/verify")
    async def verify(req: Request):
        p = principal(req); role(p, "admin"); b = await req.json()
        with store.tx() as s:
            ids = set(b.get("observation_ids", [])) | set(b.get("run_ids", []))
            present = [r.id for r in s.scalars(select(Aggregate).where(Aggregate.org == p["org_id"], Aggregate.id.in_(ids)))] if ids else []
        return ok({"verified": not present, "remaining": len(present)})

    @app.post("/internal/retention")
    def retention(req: Request):
        p = principal(req); role(p, "admin")
        return ok(purge(store, p["org_id"]))


def purge(store, org=None):
    with store.tx() as s:
        query = select(Aggregate)
        if org is not None: query = query.where(Aggregate.org == org)
        all_rows=list(s.scalars(query));moment=now()
        active=[r for r in all_rows if r.kind=="observation" and (r.expires is None or r.expires>moment)]
        active_ids={r.id for r in active};cases={r.id:r for r in all_rows if r.kind=="case"}
        keepcases=set();keepevents=set();deadlines={}
        for observation in active:
            event=observation.payload.get("payload",{}).get("event_id")
            if event:
                keepevents.add(event);deadlines[event]=max(deadlines.get(event,0),observation.expires or moment+90*86400)
            case_id=observation.payload.get("case_id");visited=set()
            while case_id and case_id not in visited:
                visited.add(case_id);keepcases.add(case_id)
                deadlines[case_id]=max(deadlines.get(case_id,0),observation.expires or moment+90*86400)
                case=cases.get(case_id)
                case_id=case.payload.get("merged_into") if case else None
        rows=[]
        for row in all_rows:
            expired=row.expires is not None and row.expires<=moment
            orphan=(row.kind=="case" and row.id not in keepcases or row.kind=="event" and row.id not in keepevents
                    or row.kind=="history" and row.payload.get("case_id") not in keepcases
                    or row.kind=="correction" and row.payload.get("observation_id") not in active_ids
                    or row.kind=="link" and (row.payload.get("source_case_id") not in keepcases or row.payload.get("target_case_id") not in keepcases))
            if (row.kind=="case" and row.id in keepcases or row.kind=="event" and row.id in keepevents) and (expired or row.expires is None):
                # Retain only necessary administrative context for live evidence;
                # old free-text details never acquire a new retention period.
                value=dict(row.payload);value.pop("reason",None);row.payload=value
                row.expires=deadlines[row.id];row.version+=1
                expired=False
            if expired or orphan:rows.append(row)
        for row in rows:
            if row.kind in ("observation", "run"):
                identifier = digest(row.kind + ":" + row.id)
                if s.get(Aggregate, identifier) is None:
                    store.create(identifier, row.org, "tombstone", {"reason": "ttl_expiry"}, now() + 180 * 86400, s)
            s.delete(row)
    return {"deleted": len(rows)}
