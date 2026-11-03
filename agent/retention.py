from datetime import datetime, timezone
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
