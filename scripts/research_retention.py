"""Manual, hash-bound research-file retention. Dry-run is the default."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone,timedelta
import argparse,hashlib,json,os,re,uuid
import csv
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/"data/research-retention"
KNOWN_ROOTS={"data/research/redaction","data/research/rdd2020","data/research/rdd2022","data/research/rdd2022-official-v5","data/research/mwpd-v6","data/demo-reference","data/public/rtk","data/public/a2d2","data/public/a2d2-improvement-v2","data/public/a2d2-refinement-v3","data/cpu-inference-benchmark","data/gpu-inference-probe","data/gpu-model-sets/final-v6"}
KNOWN_ROOTS.update({'data/research/rdd2022-blind-v7','data/public/a2d2-cycle2-v5',
                    'data/gpu-model-sets/io-overlap-v1','data/gpu-model-sets/cycle2-final-v7'})
WEIGHTS={".pt",".pth",".onnx",".safetensors",".ckpt",".pkl",".pickle"}
KINDS={"source_archive":{ ".zip",".tar",".gz",".xz",".7z",".download",".part",".assembled"},"source_image":{ ".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"},"derived_image":{ ".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"},"source_annotation":{ ".txt",".xml",".mat",".json",".jsonl",".csv"},"per_image_prediction":{ ".json",".jsonl",".csv",".npy",".npz",".jpg",".jpeg",".png"},"identifiable_manifest":{ ".json",".jsonl",".csv",".log"}}
KINDS['source_range']={'.bin'}  # Never a generic archive/model-binary extension.
RANGE_ROOT='data/research/rdd2022-blind-v7/range-cache'
RANGE_ARCHIVE_BYTES=13_264_172_619
RANGE_MAX_BYTES=4*1024*1024
EXPIRIES={"expires_at","source_expires_at","collection_expires_at","mask_expires_at","license_expires_at","retention_expires_at"}
PRESERVED={"provenance.json","source.json","source-qc.json","source-metadata.json","acquisition.json","license-evidence.json","license.json","license.txt","license.md","license","notice","readme.md","readme.txt","environment-report.json","install-receipt.json"}
PRIVATE_FIELDS={"source_member","path","image_path","mask_path","source_path","filename","file_name","sample_id","source_id","image_id","image_sha256","bbox_xywh_px","paint_mask_rle"}


class PreservedResearchFile(ValueError):
    """Explicit preservation policy, distinct from an unreadable file."""

def clock(value=None):
    if value is None:return datetime.now(timezone.utc)
    if isinstance(value,datetime):date=value
    else:date=datetime.fromisoformat(value.replace("Z","+00:00"))
    if date.tzinfo is None:raise ValueError("timezone_required")
    return date.astimezone(timezone.utc)

def iso(value):return clock(value).isoformat().replace("+00:00","Z")

def hash_text(value):return hashlib.sha256(value.encode("utf-8")).hexdigest()

def valid_hash(value):
    if not isinstance(value,str) or not re.fullmatch("[a-f0-9]{64}",value):raise ValueError("pinned_sha256_required")
    return value

def no_links(path):
    cursor=path
    while cursor!=ROOT:
        if cursor.is_symlink() or getattr(cursor,"is_junction",lambda:False)():raise ValueError("links_not_allowed")
        cursor=cursor.parent

def project_path(value,*,file=False):
    if not isinstance(value,str) or "\\" in value or ":" in value:raise ValueError("relative_posix_path_required")
    parts=PurePosixPath(value).parts
    if not parts or value.startswith("/") or ".." in parts or "." in parts:raise ValueError("path_outside_scope")
    lexical=ROOT.joinpath(*parts);no_links(lexical);path=lexical.resolve()
    if not path.is_relative_to(ROOT) or path.relative_to(ROOT).as_posix()!=value:raise ValueError("path_outside_scope")
    if file and not path.is_file():raise ValueError("file_missing")
    return path

def state_path(value):
    path=project_path(value)
    if not path.is_relative_to(STATE.resolve()) or path.suffix!=".json":raise ValueError("inventory_must_stay_in_ignored_state_directory")
    return path

def read(path):
    if path.stat().st_size>100*1024*1024:raise ValueError("metadata_too_large")
    return json.loads(path.read_text(encoding="utf-8"))

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2)
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)

@contextmanager
def inventory_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True);lock=path.with_suffix(".lock");token=uuid.uuid4().hex
    try:descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise ValueError("inventory_busy_no_concurrent_writers")
    try:
        with os.fdopen(descriptor,"w",encoding="ascii") as stream:stream.write(token)
        yield
    finally:
        if lock.exists() and lock.read_text(encoding="ascii")==token:lock.unlink()

def file_snapshot(path):
    # mtime is used only to refuse changes while hashing, NEVER as a deadline.
    if not path.is_file():raise ValueError("regular_file_required")
    with path.open("rb") as stream:
        before=os.fstat(stream.fileno());digest=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
        after=os.fstat(stream.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    if identity(before)!=identity(after) or identity(after)!=identity(path.stat()):raise ValueError("file_changed_while_hashing")
    return digest.hexdigest(),identity(after)

def roots(values):
    if not isinstance(values,list) or not values:raise ValueError("explicit_known_roots_required")
    result=[]
    for value in values:
        parts=PurePosixPath(value).parts
        experiment=len(parts)==3 and parts[:2]==("data","training") and re.fullmatch("[a-z0-9][a-z0-9_-]{0,95}",parts[2]) and parts[2] not in ("registry","environment","models")
        if value not in KNOWN_ROOTS and not experiment:raise ValueError("unknown_research_root")
        result.append(project_path(value))
    return result

def eligible(path,kind):
    if kind not in KINDS or path.suffix.lower() not in KINDS[kind] or any(suffix.lower() in WEIGHTS for suffix in path.suffixes):raise ValueError("unsupported_research_file_kind")
    if kind=='source_range':
        relative=path.relative_to(ROOT).as_posix()
        if path.parent!=ROOT/RANGE_ROOT or not re.fullmatch(r'(?:0|[1-9][0-9]*)-(?:0|[1-9][0-9]*)\.bin',path.name):raise ValueError('exact_official_range_cache_required')
        project_path(relative,file=True)
        start,end=map(int,path.stem.split('-'));expected=end-start+1
        if start<0 or end<start or end>=RANGE_ARCHIVE_BYTES or expected>RANGE_MAX_BYTES or path.stat().st_size!=expected:raise ValueError('official_range_bounds_or_size_mismatch')
        proof=project_path(path.with_suffix('.sha256').relative_to(ROOT).as_posix(),file=True)
        if proof.stat().st_size!=64:raise ValueError('bounded_range_hash_sidecar_required')
        if valid_hash(proof.read_text(encoding='ascii'))!=file_snapshot(path)[0]:raise ValueError('range_hash_sidecar_mismatch')
        return
    name=path.name.lower()
    legal_document=re.search(r"(?:^|[-_.])(?:license|licence|readme)(?:\.(?:txt|md|json|rst|html?))?$",name)
    if name in PRESERVED or name.endswith(".source.json") or legal_document:raise PreservedResearchFile("minimal_source_or_provenance_is_preserved")
    def has_private_reference(item):
        if isinstance(item,dict):return any(k in PRIVATE_FIELDS or k in {"images","predictions","detections"} and isinstance(v,(list,dict)) or has_private_reference(v) for k,v in item.items())
        if isinstance(item,list):return any(has_private_reference(v) for v in item)
        return False
    if path.suffix.lower()==".json":
        value=read(path)
        if isinstance(value,dict):
            if value.get("format")=="onnx" or "weights" in value or value.get("format") in ("training-provenance-v1","model-release-summary-v1"):raise PreservedResearchFile("model_or_minimal_provenance_is_preserved")
        # Aggregate reports without individual identifiers/pixels are retained.
        if not has_private_reference(value):raise PreservedResearchFile("aggregate_or_unclassified_json_is_preserved")
    elif path.suffix.lower()==".jsonl":
        if path.stat().st_size>100*1024*1024:raise ValueError("metadata_too_large")
        with path.open(encoding="utf-8") as stream:
            if not any(has_private_reference(json.loads(line)) for line in stream if line.strip()):raise PreservedResearchFile("aggregate_or_unclassified_jsonl_is_preserved")
    elif path.suffix.lower()==".csv":
        with path.open(encoding="utf-8-sig",newline="") as stream:
            if not set(next(csv.reader(stream),[]))&PRIVATE_FIELDS:raise PreservedResearchFile("aggregate_or_unclassified_csv_is_preserved")
    elif path.suffix.lower()==".log":
        if path.stat().st_size>100*1024*1024:raise ValueError("metadata_too_large")
        text=path.read_text(encoding="utf-8",errors="replace")
        # Verifier stdout may identify a probe by image_sha256 without a media
        # filename. Accept JSON/JSONL and Python repr keys in mixed stdout too.
        private_key=r"[\"'](?:"+"|".join(re.escape(key) for key in sorted(PRIVATE_FIELDS))+r")[\"']\s*:"
        if not re.search(private_key,text) and not re.search(r"[^\s\"<>]+\.(?:jpg|jpeg|png|bmp|webp|xml)(?:[\s\"'<>:]|$)",text,re.I):raise PreservedResearchFile("aggregate_or_unclassified_log_is_preserved")

def evidence_expiries(value):
    found=[]
    if isinstance(value,dict):
        # Exact, existing synthetic-prototype context only. These two names
        # are not general expiry aliases and never create a fresh deadline.
        if value.get('format')=='redaction-rescue-synthetic-artifact-context-v1':
            found.extend(clock(value[key]) for key in ('synthetic_tensor_temporary_expiry','legacy_source_earliest_expiry'))
        for key,item in value.items():
            if key in EXPIRIES and item is not None:found.append(clock(item))
            elif isinstance(item,(dict,list)):found.extend(evidence_expiries(item))
    elif isinstance(value,list):
        for item in value:found.extend(evidence_expiries(item))
    return found

def _register(plan_path,inventory_path="data/research-retention/inventory.json",*,now=None):
    plan_file=project_path(plan_path,file=True)
    if not any(plan_file.is_relative_to(ROOT/base) for base in ("data",".cache")):raise ValueError("private_plan_must_stay_in_ignored_directory")
    plan=read(plan_file);allowed=roots(plan.get("roots"));entries=plan.get("entries")
    if plan.get("schema_version")!=1 or not isinstance(entries,list) or not entries or len(entries)>100000:raise ValueError("invalid_inventory_plan")
    destination=state_path(inventory_path)
    if destination==plan_file:raise ValueError("plan_and_inventory_must_differ")
    value=read(destination) if destination.exists() else {"schema_version":1,"entries":{},"events":[]}
    if value.get("schema_version")!=1:raise ValueError("unsupported_inventory_version")
    changes=[];seen=set()
    for item in entries:
        path=project_path(item["path"],file=True)
        if not any(path.is_relative_to(base) and path!=base for base in allowed):raise ValueError("file_not_under_explicit_root")
        eligible(path,item["kind"]);identifier=hash_text(item["path"])
        if identifier in seen:raise ValueError("duplicate_plan_file")
        seen.add(identifier);actual,_=file_snapshot(path)
        if actual!=valid_hash(item["sha256"]):raise ValueError("file_hash_mismatch")
        proofs=item.get("evidence")
        if not isinstance(proofs,list) or not proofs or len(proofs)>10:raise ValueError("pinned_source_evidence_required")
        deadlines=[];evidence_hashes=[]
        for proof in proofs:
            source=project_path(proof["path"],file=True)
            if not source.is_relative_to(ROOT/"data") or source.suffix!=".json":raise ValueError("source_evidence_must_be_project_data_json")
            evidence_hash,_=file_snapshot(source)
            if evidence_hash!=valid_hash(proof["sha256"]):raise ValueError("source_evidence_hash_mismatch")
            deadlines.extend(evidence_expiries(read(source)));evidence_hashes.append(evidence_hash)
        if not deadlines:raise ValueError("fixed_source_expiry_required_no_mtime_fallback")
        deadline=min(deadlines)
        if item.get("expires_at") is not None and clock(item["expires_at"])!=deadline:raise ValueError("deadline_must_match_earliest_pinned_source_expiry")
        previous=value["entries"].get(identifier)
        if previous:
            if previous["sha256"]!=actual or previous.get("status") in ("deleted","missing"):raise ValueError("registered_file_changed_or_already_retired")
            if deadline>clock(previous["expires_at"]):raise ValueError("source_deadline_extension_refused")
        base=next(base for base in allowed if path.is_relative_to(base))
        record={"path":item["path"],"root":base.relative_to(ROOT).as_posix(),"path_hash":identifier,"sha256":actual,"kind":item["kind"],"expires_at":iso(deadline),"source_evidence_hashes":sorted(set(evidence_hashes)),"status":"registered"}
        changes.append((identifier,record))
    for identifier,record in changes:value["entries"][identifier]=record
    plan_relative=plan_file.relative_to(ROOT).as_posix();plan_id=hash_text(plan_relative);plan_sha=file_snapshot(plan_file)[0]
    controls=value.setdefault("control_plans",{});previous=controls.get(plan_id)
    if previous and (previous["sha256"]!=plan_sha or previous.get("status") in ("deleted","missing")):raise ValueError("registered_private_plan_changed_or_retired")
    plan_deadline=min(clock(record["expires_at"]) for _,record in changes)
    if previous:plan_deadline=min(plan_deadline,clock(previous["expires_at"]))
    controls[plan_id]={"path":plan_relative,"path_hash":plan_id,"sha256":plan_sha,"expires_at":iso(plan_deadline),"status":"registered"}
    moment=clock(now);value["events"].append({"at":iso(moment),"action":"register","plan_sha256":file_snapshot(plan_file)[0],"files":len(changes),"audit_expires_at":iso(moment+timedelta(days=180))})
    write(destination,value)
    return {"registered_files":len(changes),"inventory_files":len(value["entries"]),"source_deadlines_extended":False,"source_files_modified":False}

def register(plan_path,inventory_path="data/research-retention/inventory.json",*,now=None):
    with inventory_lock(state_path(inventory_path)):return _register(plan_path,inventory_path,now=now)

def _purge(inventory_path="data/research-retention/inventory.json",*,apply=False,now=None):
    destination=state_path(inventory_path);value=read(destination);moment=clock(now)
    if value.get("schema_version")!=1:raise ValueError("unsupported_inventory_version")
    counts={"not_due":0,"would_delete":0,"deleted":0,"missing":0,"modified_refused":0,"scope_refused":0,"retired":0}
    events=[]
    for identifier,item in value["entries"].items():
        if item.get("status") in ("deleted","missing"):counts["retired"]+=1;continue
        if clock(item["expires_at"])>moment:counts["not_due"]+=1;continue
        try:
            if identifier!=hash_text(item["path"]):raise ValueError("inventory_identity_mismatch")
            path=project_path(item["path"])
            # Revalidate the static allowed namespace, even if inventory edited.
            base=roots([item["root"]])[0]
            if not path.is_relative_to(base) or path==base:raise ValueError("file_not_under_registered_root")
            if any(suffix.lower() in WEIGHTS for suffix in path.suffixes):raise ValueError("model_weights_preserved")
            if not path.exists():state="missing"
            else:
                eligible(path,item["kind"]);actual,identity=file_snapshot(path)
                if actual!=valid_hash(item["sha256"]):state="modified_refused"
                elif apply:
                    current=path.stat()
                    if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)!=identity:state="modified_refused"
                    else:path.unlink();state="deleted"
                else:state="would_delete"
        except (ValueError,OSError,KeyError,TypeError):state="scope_refused"
        counts[state]+=1
        if apply:
            item["status"]=state if state in ("deleted","missing") else "registered"
            if state in ("deleted","missing"):item.pop("path",None);item.pop("root",None)
            events.append({"at":iso(moment),"action":state,"path_hash":identifier,"sha256":item.get("sha256"),"audit_expires_at":iso(moment+timedelta(days=180))})
    control_counts={"not_due":0,"would_delete":0,"deleted":0,"missing":0,"modified_refused":0,"scope_refused":0,"retired":0}
    for identifier,item in value.get("control_plans",{}).items():
        if item.get("status") in ("deleted","missing"):control_counts["retired"]+=1;continue
        if clock(item["expires_at"])>moment:control_counts["not_due"]+=1;continue
        try:
            if hash_text(item["path"])!=identifier:raise ValueError("inventory_identity_mismatch")
            path=project_path(item["path"])
            if path==destination or not any(path.is_relative_to(ROOT/base) for base in ("data",".cache")) or path.suffix!=".json":raise ValueError("private_plan_scope_invalid")
            if not path.exists():state="missing"
            else:
                actual,identity=file_snapshot(path)
                if actual!=valid_hash(item["sha256"]):state="modified_refused"
                elif apply:
                    s=path.stat()
                    if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=identity:state="modified_refused"
                    else:path.unlink();state="deleted"
                else:state="would_delete"
        except (ValueError,OSError,KeyError,TypeError):state="scope_refused"
        control_counts[state]+=1
        if apply:
            if state in ("deleted","missing"):item["status"]=state;item.pop("path",None)
            events.append({"at":iso(moment),"action":"private_plan_"+state,"path_hash":identifier,"sha256":item.get("sha256"),"audit_expires_at":iso(moment+timedelta(days=180))})
    if apply:
        value["events"]=[event for event in value.get("events",[]) if clock(event["audit_expires_at"])>moment]+events
        write(destination,value)
    return {"dry_run":not apply,"automatic_schedule_configured":False,"counts":counts,"private_plan_counts":control_counts,"model_weights_deleted":0}

def purge(inventory_path="data/research-retention/inventory.json",*,apply=False,now=None):
    if not apply:return _purge(inventory_path,now=now)
    with inventory_lock(state_path(inventory_path)):return _purge(inventory_path,apply=True,now=now)

def coverage(plan_path,inventory_path="data/research-retention/inventory.json",*,details=None):
    """Read-only scan of explicit plan roots; never discovers deletion targets."""
    plan=read(project_path(plan_path,file=True));allowed=roots(plan.get("roots"))
    planned={hash_text(item["path"]) for item in plan.get("entries",[])}
    inventory=state_path(inventory_path);state=read(inventory) if inventory.exists() else {}
    registered={**state.get("entries",{}),**state.get("control_plans",{})};active={key for key,item in registered.items() if item.get("status")=="registered"}
    counts={"files":0,"candidate_files":0,"uncovered_candidates":0,"planned_candidates":0,"registered_candidates":0,"preserved_files":0,"unclassified_files":0,"links_refused":0};bykind={};missing=[];seen=set()
    for base in allowed:
        for directory,subdirs,files in os.walk(base,followlinks=False):
            safe=[]
            for name in subdirs:
                path=Path(directory)/name
                if path.is_symlink() or getattr(path,"is_junction",lambda:False)():counts["links_refused"]+=1
                else:safe.append(name)
            subdirs[:]=safe
            for name in files:
                path=Path(directory)/name;relative=path.relative_to(ROOT).as_posix();identifier=hash_text(relative)
                if identifier in seen:continue
                seen.add(identifier);counts["files"]+=1
                try:project_path(relative,file=True)
                except ValueError:counts["links_refused"]+=1;continue
                suffix=path.suffix.lower()
                if any(s.lower() in WEIGHTS for s in path.suffixes):counts["preserved_files"]+=1;continue
                if suffix in KINDS["source_archive"]:kind="source_archive"
                elif suffix in KINDS["source_image"]:kind="derived_image" if relative.startswith("data/training/") else "source_image"
                elif suffix in {".npy",".npz"}:kind="per_image_prediction"
                elif suffix in KINDS["identifiable_manifest"]:kind="identifiable_manifest"
                elif suffix in KINDS["source_annotation"]:kind="source_annotation"
                else:counts["unclassified_files"]+=1;continue
                try:eligible(path,kind)
                except PreservedResearchFile:counts["preserved_files"]+=1;continue
                counts["candidate_files"]+=1;bykind[kind]=bykind.get(kind,0)+1
                if identifier in planned:counts["planned_candidates"]+=1
                if identifier in active:counts["registered_candidates"]+=1
                if identifier not in planned and identifier not in active:counts["uncovered_candidates"]+=1;missing.append({"path_hash":identifier,"kind":kind})
    if details:
        output=state_path(details);write(output,{"format":"retention-coverage-hashes-v1","uncovered_files":missing})
    return {"read_only":True,"automatic_deletion_discovery":False,"content_hashes_rechecked":False,"counts":counts,"candidate_counts_by_kind":bykind}

def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest="action",required=True)
    r=commands.add_parser("register");r.add_argument("plan");r.add_argument("--inventory",default="data/research-retention/inventory.json")
    p=commands.add_parser("purge");p.add_argument("--inventory",default="data/research-retention/inventory.json");p.add_argument("--apply",action="store_true")
    c=commands.add_parser("coverage");c.add_argument("plan");c.add_argument("--inventory",default="data/research-retention/inventory.json");c.add_argument("--details")
    args=parser.parse_args()
    try:
        result=register(args.plan,args.inventory) if args.action=="register" else coverage(args.plan,args.inventory,details=args.details) if args.action=="coverage" else purge(args.inventory,apply=args.apply)
        print(json.dumps(result,indent=2));return 0
    except Exception as error:
        # Never emit filenames, source URLs or exception payloads into reports.
        reason=str(error);reason=reason if re.fullmatch("[a-z0-9_]{1,120}",reason) else "invalid_retention_request_or_state"
        print(json.dumps({"status":"retention_operation_refused","error_type":type(error).__name__,"reason":reason}));return 2

if __name__=="__main__":raise SystemExit(main())
