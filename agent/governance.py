"""Known-file training/result retention and source revocation, with minimal audit."""
from datetime import datetime,timezone
import hashlib
import json
import uuid
from .paths import ROOT,workspace_path,atomic_json,sha256

REGISTRY=ROOT/"data/training/registry"
REVOCATIONS=ROOT/"data/training/revocations.json"


def identity_hash(kind,value):return hashlib.sha256((kind+":"+str(value)).encode()).hexdigest()


def identities(row):
    return [identity_hash(kind,row[kind]) for kind in ("sample_id","source_id","observation_id") if row.get(kind)]


def blocked_identities():
    if not REVOCATIONS.exists():return set()
    return set(json.loads(REVOCATIONS.read_text(encoding="utf-8"))["identity_hashes"])


def effective_expiry(row):
    values=[datetime.fromisoformat(row[k].replace("Z","+00:00")) for k in ("expires_at","source_expires_at","collection_expires_at","mask_expires_at","license_expires_at") if row.get(k)]
    if not values or any(value.tzinfo is None for value in values):raise ValueError("training_source_expiry_required")
    return min(values).isoformat()


def begin_training(output,release_path,rows,task):
    output=workspace_path(output)
    if output==ROOT:raise ValueError("project_root_cannot_be_training_output")
    output.mkdir(parents=True,exist_ok=True)
    blocked=blocked_identities()
    if any(set(identities(row))&blocked for row in rows):raise ValueError("source_data_revoked")
    if any(datetime.fromisoformat(effective_expiry(row))<=datetime.now(timezone.utc) for row in rows):raise ValueError("source_data_expired")
    path=output/"provenance.json"
    if path.exists():
        previous=json.loads(path.read_text(encoding="utf-8"))
        if previous.get("data_revoked"):raise ValueError("revoked_training_directory_requires_new_release_and_directory")
        # Never overwrite ownership metadata while old copies remain.
        if previous.get("assets"):raise ValueError("training_output_already_has_assets_use_new_directory")
    provenance={"format":"training-provenance-v1","release_path":workspace_path(release_path,exists=True).relative_to(ROOT).as_posix(),
        "release_sha256":sha256(workspace_path(release_path,exists=True)),"task":task,"test_split_exported":False,"production_approved":False,
        "created_at":datetime.now(timezone.utc).isoformat(),"data_revoked":False,"needs_revalidation":False,"assets":[],"models":[]}
    atomic_json(path,provenance)
    REGISTRY.mkdir(parents=True,exist_ok=True)
    atomic_json(REGISTRY/(hashlib.sha256(str(path).encode()).hexdigest()+".json"),{"provenance":path.relative_to(ROOT).as_posix()})
    return path,provenance


def register_asset(provenance,row,files,*,label_refs=None):
    source=workspace_path(row["path"],exists=True)
    return {"sample_id":row["sample_id"],"source_id":row["source_id"],"observation_id":row.get("observation_id"),
        "source_path":source.relative_to(ROOT).as_posix(),"source_sha256":row["sha256"],"expires_at":effective_expiry(row),
        "source_files":[{"path":source.relative_to(ROOT).as_posix(),"sha256":row["sha256"]}]+([
            {"path":workspace_path(row["mask_path"]).relative_to(ROOT).as_posix(),"sha256":row["mask_sha256"]}] if row.get("mask_path") else []),
        "identity_hashes":identities(row),"files":[{"path":workspace_path(p,exists=True).relative_to(ROOT).as_posix(),"sha256":sha256(workspace_path(p,exists=True))} for p in files],
        "label_refs":label_refs or []}


def link_model(provenance_path,manifest_path):
    value=json.loads(workspace_path(provenance_path,exists=True).read_text(encoding="utf-8"))
    if value.get("data_revoked") or value.get("needs_revalidation"):
        model=json.loads(workspace_path(manifest_path,exists=True).read_text(encoding="utf-8"))
        model.update(data_revoked=True,needs_revalidation=True,production_approved=False)
        atomic_json(manifest_path,model)
    value["models"].append(workspace_path(manifest_path,exists=True).relative_to(ROOT).as_posix())
    atomic_json(provenance_path,value)


def provenance_paths(training_root="data/training"):
    found=set()
    root=workspace_path(training_root)
    if root.exists():found.update(root.rglob("provenance.json"))
    if REGISTRY.exists():
        for pointer in REGISTRY.glob("*.json"):
            path=workspace_path(json.loads(pointer.read_text())["provenance"])
            if path.exists():found.add(path)
    return sorted(found)


def revoke_known(*,request=None,now=None,dry_run=True,training_root="data/training",results_root="data/agent-jobs/results"):
    now=now or datetime.now(timezone.utc)
    targets=set();request_hash=None
    if request is not None:
        if not isinstance(request,dict) or set(request)-{"sample_ids","source_ids","observation_ids","reason","request_id"}:raise ValueError("revocation_request_fields_invalid")
        for plural,kind in (("sample_ids","sample_id"),("source_ids","source_id"),("observation_ids","observation_id")):
            values=request.get(plural,[])
            if not isinstance(values,list) or len(values)>10000 or any(not isinstance(v,str) or not v.strip() for v in values):raise ValueError("revocation_identifiers_invalid")
            targets.update(identity_hash(kind,value) for value in values)
        if not targets or not isinstance(request.get("reason"),str) or not request["reason"].strip():raise ValueError("revocation_ids_and_reason_required")
        request_hash=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if not dry_run:
            # Block reuse immediately, even if a damaged artifact later needs repair.
            existing=blocked_identities();existing.update(targets)
            atomic_json(REVOCATIONS,{"identity_hashes":sorted(existing)})
    outcomes=[]
    for path in provenance_paths(training_root):
        path=workspace_path(path,exists=True)
        value=json.loads(path.read_text(encoding="utf-8"))
        if value.get("format")!="training-provenance-v1":continue
        folder=workspace_path(path.parent,exists=True)
        affected=[]
        for index,asset in enumerate(value["assets"]):
            if asset.get("revoked_at"):continue
            expires=datetime.fromisoformat(asset["expires_at"])
            if expires.tzinfo is None:raise ValueError("timezone_required")
            if not (set(asset["identity_hashes"])&targets or expires<=now):continue
            # Restrict deletion to exact, owned files under this registered artifact directory.
            files=[workspace_path(item["path"]) for item in asset["files"]]
            labels=[workspace_path(item["path"]) for item in asset["label_refs"]]
            if any(not p.is_relative_to(folder) or p==folder for p in files+labels):raise ValueError("unowned_training_delete_refused")
            if not dry_run:
                for file in files:file.unlink(missing_ok=True)
                # Original registered assets are touched only inside the managed training root.
                managed=workspace_path(training_root)
                for source in asset.get("source_files",[]):
                    original=workspace_path(source["path"])
                    if original.is_relative_to(managed) and original!=managed and original.suffix.lower() in (".png",".jpg",".jpeg") and original.exists() and sha256(original)==source["sha256"]:
                        original.unlink(missing_ok=True)
                for ref in asset["label_refs"]:
                    label=workspace_path(ref["path"])
                    if label.exists():
                        coco=json.loads(label.read_text(encoding="utf-8"));identifier=ref["image_id"]
                        coco["images"]=[row for row in coco["images"] if row["id"]!=identifier]
                        coco["annotations"]=[row for row in coco["annotations"] if row["image_id"]!=identifier]
                        atomic_json(label,coco)
                value["assets"][index]={"identity_hashes":asset["identity_hashes"],"revoked_at":now.isoformat(),
                    "reason":"explicit_request" if targets&set(asset["identity_hashes"]) else "source_expired"}
            affected.append(asset)
        if affected:
            outcomes.append({"kind":"training","provenance":path.relative_to(ROOT).as_posix(),"assets":len(affected),"applied":not dry_run})
            if not dry_run:
                for asset in value["assets"]:
                    for ref in asset.get("label_refs",[]):
                        label=workspace_path(ref["path"])
                        if label.exists():ref["sha256"]=sha256(label)
                value.update(data_revoked=True,needs_revalidation=True,production_approved=False)
                for model_path in value["models"]:
                    model_path=workspace_path(model_path,exists=True)
                    model=json.loads(model_path.read_text(encoding="utf-8"));model.update(data_revoked=True,needs_revalidation=True,production_approved=False)
                    atomic_json(model_path,model)
                atomic_json(path,value)
                targets.update(key for asset in affected for key in asset["identity_hashes"])
    root=workspace_path(results_root)
    if root.exists() and targets:
        for package_path in root.rglob("review.json"):
            value=json.loads(package_path.read_text(encoding="utf-8"));selected=[]
            for item in value.get("items",[]):
                obs=item["observation"]
                if set(identities(obs|item.get("lineage",{})))&targets:selected.append(item)
            if not selected:continue
            shared={item["image_path"] for item in selected}
            selected=[item for item in value["items"] if item["image_path"] in shared]
            folder=workspace_path(package_path.parent)
            for item in selected:
                file=workspace_path(item["image_path"])
                if not file.is_relative_to(folder):raise ValueError("unowned_review_delete_refused")
                if not dry_run:file.unlink(missing_ok=True)
            if not dry_run:
                removed={item["observation"]["observation_id"] for item in selected}
                value["items"]=[item for item in value["items"] if item["observation"]["observation_id"] not in removed]
                # Local diagnostic evidence can embed image/position references; remove it on revocation.
                value["lane_segments"]=[];value["frames"]=[];value["status"]="data_revoked"
                atomic_json(package_path,value)
            outcomes.append({"kind":"review","items":len(selected),"applied":not dry_run})
    if not dry_run and targets:
        existing=blocked_identities();existing.update(targets)
        atomic_json(REVOCATIONS,{"identity_hashes":sorted(existing)})
        audit=REGISTRY.parent/"revocation-audit.jsonl"
        with audit.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps({"at":now.isoformat(),"request_sha256":request_hash,"identity_count":len(targets),
                "affected_groups":len(outcomes)},sort_keys=True)+"\n")
    return {"dry_run":dry_run,"affected":outcomes,"identity_count":len(targets)}
