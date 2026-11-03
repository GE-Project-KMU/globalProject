"""Frozen same-data detector comparison in the real application CPU runtime."""
import argparse,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from agent.models import OnnxModel
from agent.dataset import validate_research_release
from agent.evaluate import detection_metrics
from scripts.compare_lane_models import preserve_freeze


def known_classes(row,classes):
    known=row.get("known_label_coverage",[row["evaluation_class"]] if row.get("evaluation_class") else classes)
    if not known or set(known)-set(classes):raise ValueError("invalid_known_class_scope")
    return set(known)


def score(truth,predictions,classes,confidence):
    result=detection_metrics(truth,[p for p in predictions if p["confidence"]>=confidence],classes)
    for c in result["classes"].values():c["precision"]=c["tp"]/(c["tp"]+c["fp"]) if c["tp"]+c["fp"] else 0.
    result["confidence"]=confidence
    return result


def _compare(plan_path,output):
    plan_path=workspace_path(plan_path,exists=True);plan=json.loads(plan_path.read_text(encoding="utf-8"))
    if output.exists():raise ValueError("final_comparison_already_exists")
    release=workspace_path(plan["benchmark_manifest"],exists=True)
    if sha256(release)!=plan["benchmark_sha256"]:raise ValueError("benchmark_changed")
    corpus=json.loads(release.read_text(encoding="utf-8"));rows=[r for r in corpus if r["split"]=="test"]
    if not rows or not validate_research_release(rows)["valid"]:raise ValueError("invalid_frozen_test")
    hashes={r["sha256"] for r in rows};groups={r["leakage_group_id"] for r in rows}
    models={};metadata={};model_paths={};classes=None
    for name,reference in plan["models"].items():
        path=workspace_path(reference,exists=True);model=OnnxModel(path);artifact=dict(model.manifest)
        if not artifact.get("research_only") or artifact.get("production_approved") is not False or artifact["task"] not in ("pothole","redaction"):
            raise ValueError("research_detector_required")
        if classes is None:classes=artifact["classes"]
        elif artifact["classes"]!=classes:raise ValueError("incomparable_classes")
        source=workspace_path(artifact["source_release_path"],exists=True)
        if "manifest-sha256:"+sha256(source)!=artifact["dataset_release"]:raise ValueError("source_changed")
        development=[r for r in json.loads(source.read_text(encoding="utf-8")) if r["split"] in ("train","validation")]
        if hashes&{r["sha256"] for r in development} or groups&{r["leakage_group_id"] for r in development}:raise ValueError("holdout_development_leakage")
        if name=="baseline" and sha256(path)!=plan["baseline_manifest_sha256"]:raise ValueError("baseline_changed")
        if name=="candidate" and artifact["dataset_release"]!="manifest-sha256:"+plan["candidate_release_sha256"]:raise ValueError("candidate_release_changed")
        verification=json.loads((path.parent/"verification-final.json").read_text(encoding="utf-8"))
        if verification.get("manifest_sha256")!=sha256(path) or verification.get("numerical_parity_passed") is not True or verification.get("postprocess_parity_passed") is not True:
            raise ValueError("frozen_numerical_and_postprocess_verification_required")
        confidence=artifact.get("confidence",.35)
        if confidence<.01:raise ValueError("operating_confidence_below_fixed_benchmark_floor")
        metadata[name]={"manifest_sha256":sha256(path),"weights_sha256":artifact["sha256"],"operating_confidence":confidence,
                        "architecture":artifact.get("architecture"),"input_size":artifact["input_size"],"backend":model.backend,"backend_version":model.backend_version}
        model.manifest=dict(artifact,confidence=.01);models[name]=model;model_paths[name]=path
    if set(models)!={"baseline","candidate"}:raise ValueError("one_frozen_candidate_required")
    preserve_freeze(output.with_name(output.stem+"-freeze.json"),{"plan_sha256":sha256(plan_path),"benchmark_sha256":sha256(release),"models":metadata,"model_or_threshold_selection_allowed":False})
    truth=[];predictions={name:[] for name in models};timings={name:[] for name in models}
    for row in sorted(rows,key=lambda r:r["sample_id"]):
        if row.get("ignore_bboxes_xywh_px"):raise ValueError("explicit_ignore_region_protocol_required")
        known=known_classes(row,classes)
        truth.extend({"sample_id":row["sample_id"],"class":o["class"],"bbox_xywh_px":o["bbox_xywh_px"]} for o in row["objects"] if o["class"] in known)
        image=cv2.imread(str(workspace_path(row["path"],exists=True)))
        for name,model in models.items():
            start=time.perf_counter();found=model.predict(image);timings[name].append(time.perf_counter()-start)
            predictions[name].extend(dict(p,sample_id=row["sample_id"]) for p in found if p["class"] in known)
    for name,model in models.items():
        if sha256(model_paths[name])!=metadata[name]["manifest_sha256"] or sha256(workspace_path(model.manifest["weights"]))!=metadata[name]["weights_sha256"]:raise ValueError("model_changed_during_comparison")
    import numpy as np
    results={name:{**metadata[name],"ap_metrics":score(truth,predictions[name],classes,.01),
                "operating_metrics":score(truth,predictions[name],classes,metadata[name]["operating_confidence"]),
                "fixed_035_reference":score(truth,predictions[name],classes,.35),
                "latency_seconds":{"median":float(np.median(timings[name])),"p95":float(np.percentile(timings[name],95))}} for name in models}
    report={"scope":plan["role"],"evaluated_at":datetime.now(timezone.utc).isoformat(),"benchmark_sha256":sha256(release),"plan_sha256":sha256(plan_path),
            "samples":len(rows),"results":results,"map50_change":results["candidate"]["ap_metrics"]["map50"]-results["baseline"]["ap_metrics"]["map50"],
            "known_class_conditional":any(r.get("known_label_coverage") or r.get("evaluation_class") for r in rows),
            "test_used_for_selection":False,"independent_physical_identity_or_route_verified":False,"project_acceptance_passed":False,
            "limits":["Publisher-known classes only; unknown other classes excluded from negatives", "Single-model CPU timing under shared host load, not field/full-pipeline FPS", "Small source proxy groups do not prove physical independence"]}
    atomic_json(output,report)
    atomic_json(output.with_name(output.stem+"-per-image.json"),{"truth":truth,"predictions":predictions})
    return report


def compare(plan_path,output):
    output=workspace_path(output);output.parent.mkdir(parents=True,exist_ok=True);lock=output.with_name(output.name+".lock")
    try:handle=lock.open("x",encoding="utf-8")
    except FileExistsError:raise ValueError("comparison_busy_or_stale_lock_review_required")
    try:
        with handle:handle.write(str(os.getpid()))
        return _compare(plan_path,output)
    finally:lock.unlink()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("plan");p.add_argument("output");a=p.parse_args()
    print(json.dumps(compare(a.plan,a.output),ensure_ascii=False,indent=2))
