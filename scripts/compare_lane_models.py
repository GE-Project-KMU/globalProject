"""Final-only same-holdout comparison of frozen ONNX paint models."""
import argparse,json,sys,time,os
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from agent.models import OnnxModel
from agent.dataset import validate_research_release
from scripts.evaluate_lane_model import confusion,metrics


def preserve_freeze(path,value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=value:raise ValueError("frozen_comparison_changed_no_reselection")
    else:atomic_json(path,value)


def verified_lane_artifact(path,*,diagnostic=False):
    from scripts.verify_trained_models import lane_postprocess_passed
    model=json.loads(path.read_text(encoding="utf-8"))
    proof=json.loads((path.parent/"verification-final.json").read_text(encoding="utf-8"))
    if (proof.get("manifest_sha256")!=sha256(path)
        or not model.get("sha256") or proof.get("weight_sha256")!=model["sha256"]
        or proof.get("torch_checkpoint_compared") is not True or proof.get("synthetic_input") is not False
        or not lane_postprocess_passed(proof)):
        raise ValueError("positive_validation_numerical_and_postprocess_verification_required")
    strict=proof.get("numerical_parity_passed") is True
    if strict and proof.get("artifact_contract_verified") is not True:raise ValueError("positive_validation_numerical_and_postprocess_verification_required")
    if not strict:
        if not diagnostic:raise ValueError("positive_validation_numerical_and_postprocess_verification_required")
        masks=proof.get("mask_agreement",{})
        if (proof.get("numerical_parity_passed") is not False or proof.get("artifact_contract_verified") is not False
            or proof.get("app_vs_ort",{}).get("passed") is not True
            or model.get("checkpoint_raw_tensor_parity_passed") is not False
            or model.get("diagnostic_accuracy_only") is not True
            or "checkpoint_raw_tensor_parity_unresolved" not in model.get("deployment_blocked_reasons",[])
            or model.get("research_only") is not True or model.get("production_approved") is not False
            or model.get("project_acceptance_eligible") is not False
            or not model.get("trained_checkpoint_sha256") or proof.get("checkpoint_sha256")!=model["trained_checkpoint_sha256"]
            or any(masks.get(name,{}).get("pixel_agreement")!=1.0 or masks.get(name,{}).get("paint_pixels",0)<1 for name in ("app","ort","torch"))):
            raise ValueError("explicit_runtime_consistent_research_diagnostic_required")
    return {"checkpoint_raw_tensor_parity_passed":strict,"diagnostic_accuracy_only":not strict,
            "verification_manifest_sha256":proof["manifest_sha256"],"validation_probe_scope":"one real image; GT positivity and predeclaration audited separately"}


def _compare(plan_path,output):
    plan_path=workspace_path(plan_path,exists=True);plan=json.loads(plan_path.read_text(encoding="utf-8"))
    output=workspace_path(output)
    if output.exists():raise ValueError("comparison_already_exists_do_not_reselect")
    bench=workspace_path(plan["benchmark_manifest"],exists=True)
    if sha256(bench)!=plan["benchmark_sha256"]:raise ValueError("benchmark_changed")
    rows=json.loads(bench.read_text(encoding="utf-8"))
    if not rows or any(r["split"]!="test" for r in rows):raise ValueError("isolated_test_only_required")
    if not validate_research_release(rows)["valid"]:raise ValueError("invalid_research_holdout")
    models={};frozen={};paths={};seen={r["sha256"] for r in rows};groups={r["leakage_group_id"] for r in rows}
    for name,path in plan["models"].items():
        path=workspace_path(path,exists=True);model=OnnxModel(path);m=model.manifest
        if m["task"]!="lane" or not m.get("research_only") or m.get("production_approved") is not False:raise ValueError("research_paint_model_required")
        if not m.get("threshold_calibration") or m["threshold_calibration"]["selected_on"]!="validation":raise ValueError("validation_threshold_freeze_required")
        source=workspace_path(m["source_release_path"],exists=True)
        if "manifest-sha256:"+sha256(source)!=m["dataset_release"]:raise ValueError("training_release_changed")
        development=[r for r in json.loads(source.read_text(encoding="utf-8")) if r["split"] in ("train","validation")]
        if seen&{r["sha256"] for r in development} or groups&{r["leakage_group_id"] for r in development}:raise ValueError("holdout_training_or_validation_leakage")
        if name=="baseline" and sha256(path)!=plan["baseline_manifest_sha256"]:raise ValueError("baseline_changed")
        if name=="candidate" and m["dataset_release"]!="manifest-sha256:"+plan["candidate_release_sha256"]:raise ValueError("candidate_source_changed")
        diagnostic=plan.get("verification_mode")=="diagnostic_accuracy_only_with_unresolved_checkpoint_raw_parity"
        verification=verified_lane_artifact(path,diagnostic=diagnostic and name=="candidate")
        models[name]=model;paths[name]=path;frozen[name]={"manifest_sha256":sha256(path),"weights_sha256":m["sha256"],"paint_threshold":m["paint_threshold"],
            "input_size":m["input_size"],"backend":model.backend,"backend_version":model.backend_version,"verification":verification}
    if set(models)!={"baseline","candidate"}:raise ValueError("exactly_one_baseline_and_candidate_required")
    # This receipt is written before any holdout prediction. It records no
    # scores and is not a threshold/model-selection mechanism.
    preserve_freeze(output.with_name(output.stem+"-freeze.json"),{"plan_sha256":sha256(plan_path),"benchmark_sha256":sha256(bench),"models":frozen,"test_selection_allowed":False})
    totals={name:Counter() for name in models};recordings={name:defaultdict(Counter) for name in models};times={name:[] for name in models};samples=[]
    for row in sorted(rows,key=lambda r:r["source_id"]):
        image=cv2.imread(str(workspace_path(row["path"],exists=True)))
        truth=cv2.imread(str(workspace_path(row["mask_path"],exists=True)),cv2.IMREAD_GRAYSCALE)
        if sha256(workspace_path(row["mask_path"]))!=row["mask_sha256"]:raise ValueError("test_mask_changed")
        one={"sample_id":row["sample_id"],"recording":row.get("recording",row["leakage_group_id"])}
        for name,model in models.items():
            start=time.perf_counter();predicted=model.predict(image);times[name].append(time.perf_counter()-start)
            c=confusion(truth,predicted);totals[name].update(c);recordings[name][one["recording"]].update(c)
            one[name]={"confusion":c,**metrics(c)}
        samples.append(one)
    if any(sha256(paths[name])!=frozen[name]["manifest_sha256"] or sha256(workspace_path(model.manifest["weights"]))!=frozen[name]["weights_sha256"] for name,model in models.items()):raise ValueError("model_changed_during_comparison")
    results={name:{**frozen[name],"confusion":dict(c),**metrics(c),"recordings":{r:{"confusion":dict(v),**metrics(v)} for r,v in recordings[name].items()},
        "latency_seconds":{"median":float(np.median(times[name])),"p95":float(np.percentile(times[name],95))}} for name,c in totals.items()}
    report={"scope":plan.get("role","same_predeclared_new_recording_holdout_actual_app"),"evaluated_at":datetime.now(timezone.utc).isoformat(),
        "plan_sha256":sha256(plan_path),"benchmark_sha256":sha256(bench),"samples":len(rows),"results":results,
        "paint_iou_change":results["candidate"]["paint_iou"]-results["baseline"]["paint_iou"],
        "test_used_for_model_or_threshold_selection":False,"physical_route_independence_verified":False,"project_acceptance_passed":False,
        "checkpoint_raw_tensor_parity_all_models_passed":all(value["verification"]["checkpoint_raw_tensor_parity_passed"] for value in frozen.values()),
        "diagnostic_accuracy_only":any(value["verification"]["diagnostic_accuracy_only"] for value in frozen.values()),
        "retention_mae_pp":None,"limits":["Two recording groups do not prove independent physical roads", "Publisher paint proxy only; no outline/gap/valid wear-ratio GT", "Single-model CPU timing under shared host load, not full-pipeline FPS"]}
    atomic_json(output,report);atomic_json(output.with_name(output.stem+"-per-image.json"),samples)
    return report


def compare(plan_path,output):
    output=workspace_path(output);output.parent.mkdir(parents=True,exist_ok=True)
    lock=output.with_name(output.name+".lock")
    try:handle=lock.open("x",encoding="utf-8")
    except FileExistsError:raise ValueError("comparison_busy_or_stale_lock_review_required")
    try:
        with handle:handle.write(str(os.getpid()))
        return _compare(plan_path,output)
    finally:lock.unlink()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("plan");p.add_argument("output");a=p.parse_args()
    print(json.dumps(compare(a.plan,a.output),ensure_ascii=False,indent=2))
