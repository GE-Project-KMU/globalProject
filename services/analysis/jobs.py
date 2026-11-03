"""Real local-agent workflow bridge. Prepared configs are operator-owned files."""
import asyncio, json, os, subprocess, sys
from pathlib import Path
from fastapi import Request
from pydantic import Field
from services.contracts import Contract,UUIDString
from common.runtime import *

class AnalysisJobCreate(Contract):
    run_id:UUIDString
    model_bundle_id:UUIDString
    prepared_config_id:UUIDString

def public(row):
    return {k:row.get(k) for k in ("id","run_id","model_bundle_id","status","created_at","started_at","finished_at","observation_count","reason")}

def register(app,store):
    tasks=set()
    async def execute(identifier,p,config):
        row=store.get(identifier,p["org_id"],"analysis_job")
        data={k:v for k,v in row.items() if k not in ("id","org_id","kind","created","expires","version")}
        data.update(status="running",started_at=iso())
        store.change(identifier,p["org_id"],data,row["version"])
        try:
            env=os.environ.copy()
            cache=DATA/".cache"; (cache/"tmp").mkdir(parents=True,exist_ok=True)
            env["PYTHONPYCACHEPREFIX"]=str(cache/"pycache")
            env["TEMP"]=env["TMP"]=str(cache/"tmp")
            completed=await asyncio.to_thread(subprocess.run,[sys.executable,"-m","agent","workflow",str(config)],cwd=ROOT,env=env,capture_output=True,text=True,encoding="utf-8",timeout=3600,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            try:result=json.loads(completed.stdout.strip().splitlines()[-1])
            except (ValueError,IndexError):result={}
            status="completed" if completed.returncode==0 and result.get("status")=="completed" else "blocked" if completed.returncode==2 else "failed"
            data.update(status=status,finished_at=iso(),observation_count=result.get("observation_count",0) if status=="completed" else None,reason=None if status=="completed" else "MODEL_NOT_READY" if status=="blocked" else "WORKFLOW_FAILED")
        except subprocess.TimeoutExpired:data.update(status="failed",finished_at=iso(),reason="WORKFLOW_TIMEOUT")
        except Exception:
            log.exception("analysis workflow execution failed")
            data.update(status="failed",finished_at=iso(),reason="WORKFLOW_FAILED")
        row=store.get(identifier,p["org_id"],"analysis_job")
        store.change(identifier,p["org_id"],data,row["version"])
        store.audit(p,"analysis."+data["status"],identifier)

    @app.post("/api/v1/analysis/jobs",status_code=202)
    async def create(body:AnalysisJobCreate,req:Request):
        p=principal(req);role(p,"admin","reviewer")
        for value in (body.run_id,body.model_bundle_id,body.prepared_config_id):valid_uuid(value)
        run=(await remote("inspection","/api/v1/runs/"+body.run_id,p)).json()["data"]
        bundle=store.get(body.model_bundle_id,p["org_id"],"bundle")
        config=(DATA/"agent-jobs"/"prepared"/(body.prepared_config_id+".json")).resolve()
        if not config.is_relative_to(DATA/"agent-jobs"/"prepared") or not config.is_file():fail(404,"PREPARED_CONFIG_NOT_FOUND")
        if config.stat().st_size>64000:fail(422,"INVALID_PREPARED_CONFIG")
        try:
            b=json.loads(config.read_text(encoding="utf-8"))
            if b["run_id"]!=body.run_id or b["model_bundle_id"]!=body.model_bundle_id or b["device_id"]!=run["device_id"] or b["profile_id"]!=run["profile_id"]:raise ValueError()
            output=Path(b["output"])
            output=(ROOT/output).resolve() if not output.is_absolute() else output.resolve()
            if not output.is_relative_to(DATA/"agent-jobs"/"results"):raise ValueError()
            local_bundle=Path(b["bundle_path"])
            local_bundle=(ROOT/local_bundle).resolve() if not local_bundle.is_absolute() else local_bundle.resolve()
            if not local_bundle.is_relative_to(ROOT):raise ValueError()
            registered=json.loads(local_bundle.read_text(encoding="utf-8"))
            if any(registered[k]!=bundle[k] for k in ("version","config_hash","weights_hashes")):raise ValueError()
            profile=store.get(run["profile_id"],p["org_id"],"profile")
            local_profile=Path(b["profile_path"])
            local_profile=(ROOT/local_profile).resolve() if not local_profile.is_absolute() else local_profile.resolve()
            if not local_profile.is_relative_to(ROOT):raise ValueError()
            profile_config=json.loads(local_profile.read_text(encoding="utf-8"))
            if profile_config.get("calibration")!=profile["calibration"]:raise ValueError()
        except (ValueError,KeyError,TypeError,OSError):fail(422,"PREPARED_CONFIG_MISMATCH")
        for old in store.list(p["org_id"],"analysis_job"):
            if old["run_id"]==body.run_id and old["status"] in ("queued","running"):fail(409,"ANALYSIS_ALREADY_RUNNING")
        identifier=uid()
        row=store.create(identifier,p["org_id"],"analysis_job",{"run_id":body.run_id,"model_bundle_id":body.model_bundle_id,"status":"queued","created_at":iso()},run["expires"])
        store.audit(p,"analysis.queue",identifier)
        task=asyncio.create_task(execute(identifier,p,config));tasks.add(task);task.add_done_callback(tasks.discard)
        return ok(public(row))

    @app.get("/api/v1/analysis/jobs")
    def listing(req:Request):
        p=principal(req);role(p,"admin","reviewer")
        return ok({"items":[public(r) for r in store.list(p["org_id"],"analysis_job")]})

    @app.get("/api/v1/analysis/jobs/{id}")
    def detail(id:UUIDString,req:Request):
        p=principal(req);role(p,"admin","reviewer")
        return ok(public(store.get(valid_uuid(id),p["org_id"],"analysis_job")))

    # Interrupted work is never reported as successfully completed after a restart.
    for row in store.list(None,"analysis_job"):
        if row["status"] in ("queued","running"):
            data={k:v for k,v in row.items() if k not in ("id","org_id","kind","created","expires","version")}
            data.update(status="failed",reason="SERVICE_RESTARTED",finished_at=iso())
            store.change(row["id"],row["org_id"],data,row["version"])
