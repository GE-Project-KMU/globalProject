"""Build/install research model archives with pinned hashes and no raw datasets."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,stat,sys,tempfile,uuid,urllib.parse,urllib.request,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_trained_models import ROOT,scoped,digest,read,validate_metadata

FILES={"model.json","weights.onnx","provenance.json","release-summary.json","verification-summary.json"}
LIMIT=512*1024*1024

def encoded(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8")

def validate_sha(value):
    if not isinstance(value,str) or not re.fullmatch("[a-f0-9]{64}",value):raise ValueError("pinned_sha256_required")
    return value

def sanitize_provenance(info):
    old=info["provenance"];release=read(scoped(info["manifest"]["source_release_path"]))
    byid={row["sample_id"]:row for row in release} if isinstance(release,list) else {}
    assets=[]
    for asset in old["assets"]:
        split=byid.get(asset.get("sample_id"),{}).get("split",asset.get("training_split"))
        if split not in ("train","validation"):raise ValueError("test_or_unknown_asset_in_training")
        hashes=sorted({validate_sha(asset["source_sha256"]),*[validate_sha(x["sha256"]) for x in asset.get("source_files",[])]})
        item={"training_split":split,"identity_hashes":[validate_sha(x) for x in asset["identity_hashes"]],"source_sha256":asset["source_sha256"],"source_hashes":hashes,"expires_at":asset["expires_at"]}
        if asset.get("revoked_at"):item["revoked_at"]=asset["revoked_at"]
        assets.append(item)
    return {"format":"training-provenance-v1","sanitized_for_distribution":True,"release_sha256":info["source_hash"],"task":info["manifest"]["task"],"test_split_exported":False,"production_approved":False,"data_revoked":False,"needs_revalidation":False,"assets":assets}

def verification_summary(path,info):
    value=read(scoped(path));manifest_hash=digest(info["manifest_path"])
    if value.get("manifest_sha256")!=manifest_hash or value.get("weight_sha256")!=info["manifest"]["sha256"]:raise ValueError("verification_artifact_hash_mismatch")
    for key in ("metadata_valid","weight_hash_verified","finite_output","preprocess_contract_verified"):
        if value.get(key) is not True:raise ValueError("full_runtime_verification_required")
    if value.get("app_vs_ort",{}).get("passed") is not True:raise ValueError("onnxruntime_parity_required")
    if value.get("numerical_parity_passed") is not True:raise ValueError("all_requested_numerical_comparisons_must_pass")
    if value.get("postprocess_parity_passed",True) is not True:raise ValueError("detection_postprocess_parity_required")
    result={key:value[key] for key in ("manifest_sha256","weight_sha256","metadata_valid","weight_hash_verified","finite_output","preprocess_contract_verified")}
    for key in ("app_vs_ort","app_vs_torch"):
        if key in value:result[key]={k:v for k,v in value[key].items() if k in ("passed","max_abs_error","mean_abs_error","rtol","atol") and isinstance(v,(bool,int,float))}
    for key in ("app_backend","app_backend_version","runtime_python","opencv_preprocessing_version"):
        if key in value and isinstance(value[key],str) and re.fullmatch("[a-zA-Z0-9_+.-]{1,80}",value[key]):result[key]=value[key]
    result["numerical_parity_passed"]=True
    result.update(research_only=True,production_approved=False,accuracy_evaluation_performed=False)
    return result

def build(manifest,output,verification):
    info=validate_metadata(manifest);m=info["manifest"]
    if m.get("artifact_redistribution_allowed") is not True:raise ValueError("artifact_redistribution_not_explicitly_allowed")
    path=scoped(output,exists=False)
    if path.exists() or path.suffix.lower()!=".zip":raise ValueError("new_zip_output_required")
    weights=info["weight_path"]
    if not weights.is_file() or weights.stat().st_size>LIMIT or digest(weights)!=m["sha256"]:raise ValueError("weight_hash_mismatch_or_size")
    verified=verification_summary(verification,info)
    summary={"format":"model-release-summary-v1","original_release_sha256":info["source_hash"],"split_counts":info["split_counts"],"research_only":True,"licenses":m["source_licenses"],"independent_route_verified":m["independent_route_verified"],"training_test_split_used":False,"production_approved":False}
    summary_bytes=encoded(summary)
    allowed={"sha256","format","task","classes","target_trained","dataset_release","input_size","mean","std","output_layout","strides","confidence","nms_iou","architecture","research_only","source_licenses","independent_route_verified","project_acceptance_eligible","production_approved","test_split_used","data_revoked","needs_revalidation","validation_paint_iou","previous_manifest_sha256","quality_status","deployment_blocked_reasons","paint_threshold","threshold_calibration","artifact_redistribution_allowed"}
    model={key:value for key,value in m.items() if key in allowed}
    model.update(weights="weights.onnx",training_provenance="provenance.json",source_release_path="release-summary.json",source_release_format="summary-v1",release_summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),origin_manifest_sha256=digest(info["manifest_path"]),production_approved=False)
    payload={"model.json":encoded(model),"provenance.json":encoded(sanitize_provenance(info)),"release-summary.json":summary_bytes,"verification-summary.json":encoded(verified)}
    hashes={name:hashlib.sha256(value).hexdigest() for name,value in payload.items()};hashes["weights.onnx"]=m["sha256"]
    bundle={"schema_version":1,"task":m["task"],"research_only":True,"artifact_redistribution_allowed":True,"origin_manifest_sha256":digest(info["manifest_path"]),"files":hashes}
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with zipfile.ZipFile(path,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name,value in payload.items():archive.writestr(name,value)
            archive.write(weights,"weights.onnx");archive.writestr("bundle.json",encoded(bundle))
        if digest(weights)!=m["sha256"]:raise ValueError("weights_changed_during_packaging")
    except Exception:
        path.unlink(missing_ok=True);raise
    return {"archive":path.relative_to(ROOT).as_posix(),"sha256":digest(path),"bytes":path.stat().st_size,"task":m["task"],"research_only":True,"production_approved":False}

def inspect_archive(path,sha):
    path=scoped(path);validate_sha(sha)
    if path.stat().st_size>LIMIT or digest(path)!=sha:raise ValueError("archive_hash_mismatch_or_size")
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=FILES|{"bundle.json"}:raise ValueError("unexpected_or_duplicate_archive_files")
        if sum(x.file_size for x in archive.infolist())>LIMIT:raise ValueError("archive_uncompressed_size_exceeded")
        for item in archive.infolist():
            if item.is_dir() or stat.S_ISLNK(item.external_attr>>16):raise ValueError("archive_links_not_allowed")
        bundle=json.loads(archive.read("bundle.json"))
        if bundle.get("schema_version")!=1 or bundle.get("research_only") is not True or bundle.get("artifact_redistribution_allowed") is not True or set(bundle.get("files",{}))!=FILES:raise ValueError("invalid_bundle_metadata")
        payload={name:archive.read(name) for name in FILES}
        for name,value in payload.items():
            if hashlib.sha256(value).hexdigest()!=validate_sha(bundle["files"][name]):raise ValueError("bundle_file_hash_mismatch")
    model=json.loads(payload["model.json"])
    if validate_sha(bundle.get("origin_manifest_sha256"))!=model.get("origin_manifest_sha256"):raise ValueError("original_manifest_hash_mismatch")
    if model.get("weights")!="weights.onnx" or model.get("training_provenance")!="provenance.json" or model.get("source_release_path")!="release-summary.json" or model.get("sha256")!=bundle["files"]["weights.onnx"]:raise ValueError("bundle_model_reference_mismatch")
    if model.get("research_only") is not True or model.get("production_approved") is not False or model.get("artifact_redistribution_allowed") is not True:raise ValueError("invalid_bundle_research_flags")
    return bundle,payload

def install(path,sha,identifier):
    if not re.fullmatch("[a-z0-9][a-z0-9_-]{0,63}",identifier):raise ValueError("invalid_artifact_id")
    bundle,payload=inspect_archive(path,sha)
    parent=ROOT/"data/models/research";parent.mkdir(parents=True,exist_ok=True);target=(parent/identifier).resolve()
    if not target.is_relative_to(ROOT) or target.exists():raise ValueError("new_project_model_destination_required")
    with tempfile.TemporaryDirectory(prefix=".install-",dir=parent) as temporary:
        stage=Path(temporary)
        for name,value in payload.items():(stage/name).write_bytes(value)
        model=json.loads(payload["model.json"])
        for key,name in (("weights","weights.onnx"),("training_provenance","provenance.json"),("source_release_path","release-summary.json")):model[key]=(stage/name).relative_to(ROOT).as_posix()
        (stage/"model.json").write_bytes(encoded(model));validate_metadata(stage/"model.json")
        for key,name in (("weights","weights.onnx"),("training_provenance","provenance.json"),("source_release_path","release-summary.json")):model[key]=(target/name).relative_to(ROOT).as_posix()
        model["installed_from_archive_sha256"]=sha
        (stage/"model.json").write_bytes(encoded(model))
        receipt={"archive_sha256":sha,"origin_manifest_sha256":bundle["origin_manifest_sha256"],"installed_manifest_sha256":digest(stage/"model.json"),"research_only":True,"production_approved":False}
        (stage/"install-receipt.json").write_bytes(encoded(receipt));stage.rename(target)
    return {"manifest":(target/"model.json").relative_to(ROOT).as_posix(),**receipt}

def download(catalog,identifier,output):
    data=read(scoped(catalog))
    if data.get("schema_version")!=1:raise ValueError("unsupported_publisher_catalog_version")
    entries=[item for item in data.get("artifacts",[]) if item.get("id")==identifier]
    if len(entries)!=1:raise ValueError("unique_publisher_catalog_entry_required")
    entry=entries[0]
    if entry.get("research_only") is not True or entry.get("artifact_redistribution_allowed") is not True:raise ValueError("publisher_redistribution_not_allowed")
    sha=validate_sha(entry.get("sha256"));url=urllib.parse.urlparse(entry["url"])
    if url.scheme!="https" or not url.hostname or url.username or url.password or url.query:raise ValueError("public_https_catalog_url_required")
    path=scoped(output,exists=False)
    if path.exists():raise ValueError("new_download_destination_required")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".part")
    try:
        total=0
        with urllib.request.urlopen(entry["url"],timeout=60) as response,temporary.open("xb") as stream:
            if urllib.parse.urlparse(response.url).scheme!="https":raise ValueError("insecure_download_redirect")
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                total+=len(chunk)
                if total>LIMIT:raise ValueError("download_size_exceeded")
                stream.write(chunk)
        inspect_archive(temporary,sha);os.link(temporary,path)
    finally:temporary.unlink(missing_ok=True)
    return {"archive":path.relative_to(ROOT).as_posix(),"sha256":sha,"research_only":True}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    b=sub.add_parser("build");b.add_argument("manifest");b.add_argument("output");b.add_argument("--verification",required=True)
    i=sub.add_parser("install");i.add_argument("archive");i.add_argument("--sha256",required=True);i.add_argument("--id",required=True)
    d=sub.add_parser("download");d.add_argument("catalog");d.add_argument("--id",required=True);d.add_argument("--output",required=True)
    args=p.parse_args()
    try:
        result=build(args.manifest,args.output,args.verification) if args.action=="build" else install(args.archive,args.sha256,args.id) if args.action=="install" else download(args.catalog,args.id,args.output)
        print(json.dumps(result,indent=2));return 0
    except Exception as error:print(json.dumps({"status":"artifact_operation_blocked","reason":str(error)[:200]}));return 2

if __name__=="__main__":raise SystemExit(main())
