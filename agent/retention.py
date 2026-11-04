from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import shutil
import os
from .paths import workspace_path, atomic_json
from .paths import sha256
from .material_policy import instant,safe_file


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
    safe_file(Path(work)/'.scope.json',work,exists=False,suffixes=('.json',))
    safe_file(Path(results)/'.scope.json',results,exists=False,suffixes=('.json',))
    work,results=workspace_path(work),workspace_path(results)
    if workspace_path('.') in (work,results):raise ValueError('project_root_cleanup_forbidden')
    if work.exists():
        for marker in work.glob("*/retention.json"):
            safe_file(marker,marker.parent,suffixes=('.json',))
            folder=workspace_path(marker.parent,exists=True)
            expires=datetime.fromisoformat(json.loads(marker.read_text())["expires_at"])
            if expires.tzinfo is None:raise ValueError("timezone_required")
            if expires<=now:
                if folder==work or not folder.is_relative_to(work):raise ValueError("unsafe_work_delete")
                deleted.append({"kind":"temporary_raw","path":str(folder),"deleted":not dry_run})
                if not dry_run:shutil.rmtree(folder)
    if results.exists():
        packages=[]
        for directory,dirs,files in os.walk(results,followlinks=False):
            safe=[]
            for name in dirs:
                child=Path(directory)/name
                try:safe_file(child/'.scope.json',child,exists=False,suffixes=('.json',));safe.append(name)
                except ValueError:deleted.append({'kind':'derived_refused','deleted':False,'reason':'material_links_forbidden'})
            dirs[:]=safe
            packages.extend(Path(directory)/name for name in files if name in ('review.json','analysis.json'))
        packages.sort()
        for package in packages:
            try:
                package=safe_file(package,package.parent,suffixes=('.json',))
                marker_hash=sha256(package)
                value=json.loads(package.read_text(encoding='utf-8'))
                policy=value.get('local_material',{})
                state=policy.get('state')
                if state not in ('unreviewed','approved'):raise ValueError('unbound_legacy_material_requires_manual_review')
                expires=instant(policy['expires_at']);cap=instant(policy['source_expires_at']) if policy.get('source_expires_at') else None
                if cap and expires>cap:raise ValueError('material_expiry_exceeds_original_source')
                if expires>now:continue
                refs=[]
                if 'observations' in value: # Diagnostic analyze_video, no guessed capture UTC.
                    if state!='unreviewed':raise ValueError('diagnostic_material_cannot_be_approved')
                    maximum=instant(value['started_at'])+timedelta(days=7)
                    if not cap and not policy.get('source_captured_at'):raise ValueError('explicit_original_source_context_required')
                    if policy.get('source_captured_at'):maximum=min(maximum,instant(policy['source_captured_at'])+timedelta(days=7))
                    if expires>maximum:raise ValueError('unreviewed_material_exceeds_seven_days')
                    refs=[(row['image']['path'],row['image']['sha256']) for row in value['observations']]
                else:
                    for item in value.get('items',[]):
                        obs=item['observation'];maximum=instant(obs['captured_at'])+timedelta(days=90 if state=='approved' else 7)
                        if expires>maximum:raise ValueError('material_expiry_exceeds_capture_policy')
                        if state=='approved' and obs['image']['reviewed_for_sharing'] is not True:raise ValueError('approved_copy_requires_explicit_review')
                        refs.append((item['image_path'],obs['image']['sha256']))
                paths={}
                for relative,expected in refs:
                    path=safe_file(relative,package.parent,exists=False)
                    if not path.name.endswith('.redacted.png'):raise ValueError('unowned_derived_filename')
                    if path.exists():
                        if sha256(path)!=expected:raise ValueError('material_image_hash_changed')
                        paths[path]=expected
                # Validate the entire group before deleting any owned file.
                if sha256(package)!=marker_hash or any(sha256(path)!=expected for path,expected in paths.items()):raise ValueError('material_changed_during_cleanup')
                deleted.append({'kind':'derived','path':str(package.parent),'deleted':not dry_run,'state':state,'known_images':len(paths)})
                if not dry_run:
                    for path,expected in paths.items():
                        if sha256(path)!=expected:raise ValueError('material_changed_during_cleanup')
                        path.unlink()
                    if sha256(package)!=marker_hash:raise ValueError('material_changed_during_cleanup')
                    package.unlink()
            except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as error:
                deleted.append({'kind':'derived_refused','deleted':False,'reason':str(error) if isinstance(error,ValueError) else type(error).__name__})
    from .governance import revoke_known
    deleted.extend(revoke_known(now=now,dry_run=dry_run,training_root=training,results_root=results)["affected"])
    return deleted
