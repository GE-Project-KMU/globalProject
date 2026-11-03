from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import shutil
from .paths import workspace_path, atomic_json


def purge_expired_runs(directory="data/agent/runs", *, now=None, dry_run=True):
    directory = workspace_path(directory)
    now = now or datetime.now(timezone.utc)
    results = []
    if not directory.exists():
        return results
    for manifest_path in directory.glob("*/manifest.json"):
        folder = workspace_path(manifest_path.parent, exists=True)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expires = datetime.fromisoformat(manifest["expires_at"].replace("Z", "+00:00"))
            if expires.tzinfo is None:
                raise ValueError("timezone_required")
            if expires > now:
                continue
            # Resolve and verify absolute target before recursive deletion.
            if not folder.is_relative_to(directory.resolve()) or folder == directory.resolve():
                raise ValueError("unsafe_delete_target")
            entry = {"run_id": manifest["run_id"], "expired": True, "deleted": not dry_run}
            if not dry_run:
                shutil.rmtree(folder)
            results.append(entry)
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            results.append({"manifest": manifest_path.name, "deleted": False, "error": str(exc)})
    return results


def purge_local_material(*, work="data/agent/work", results="data/agent-jobs/results",training="data/training", now=None, dry_run=True):
    """Delete only known expired artifacts; preserve unrelated files in job folders."""
    now=now or datetime.now(timezone.utc)
    deleted=[]
    work,results=workspace_path(work),workspace_path(results)
    if work.exists():
        for marker in work.glob("*/retention.json"):
            folder=workspace_path(marker.parent,exists=True)
            expires=datetime.fromisoformat(json.loads(marker.read_text())["expires_at"])
            if expires.tzinfo is None:raise ValueError("timezone_required")
            if expires<=now:
                if folder==work or not folder.is_relative_to(work):raise ValueError("unsafe_work_delete")
                deleted.append({"kind":"temporary_raw","path":str(folder),"deleted":not dry_run})
                if not dry_run:shutil.rmtree(folder)
    if results.exists():
        for package in results.rglob("review.json"):
            package=workspace_path(package,exists=True)
            value=json.loads(package.read_text(encoding="utf-8"))
            expires=datetime.fromisoformat(value["run"]["started_at"])+timedelta(days=90)
            if expires.tzinfo is None:raise ValueError("timezone_required")
            if expires>now:continue
            # Known generated stills only; never follow an image pointer outside its job.
            folder=package.parent
            paths=[workspace_path(p,exists=True) for p in folder.glob("*.redacted.png")]
            if any(p.parent!=folder for p in paths):raise ValueError("unsafe_derived_delete")
            deleted.append({"kind":"derived","path":str(folder),"deleted":not dry_run})
            if not dry_run:
                for path in paths+[package]:path.unlink()
    from .governance import revoke_known
    deleted.extend(revoke_known(now=now,dry_run=dry_run,training_root=training,results_root=results)["affected"])
    return deleted
