"""Publish aggregate-only frozen experiment evidence; never performs inference."""
import json,sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256

def read(path):return json.loads(path.read_text(encoding="utf-8"))
def metrics_equal(a,b):
    return a["map50"]==b["map50"] and all(a["classes"][name][key]==b["classes"][name][key]
        for name in a["classes"] for key in ("gt_count","prediction_count","tp","fp","fn","ap50"))

def aggregate_only(value):
    forbidden={"sample_id","source_id","image","image_path","image_sha256","path","source_path","source_member","original_member","filename","file_name","image_id","pixel_coordinates","tensor_indices","bbox","bbox_xywh_px","truth","predictions","weights","checkpoint"}
    if isinstance(value,dict):
        if forbidden&set(value):raise ValueError("private_fields_in_public_report:"+",".join(sorted(forbidden&set(value))))
        for item in value.values():aggregate_only(item)
    elif isinstance(value,list):
        for item in value:aggregate_only(item)
    elif isinstance(value,str):
        normal=value.replace("\\","/").lower()
        if "data/training/" in normal or re.match(r"^[a-z]:/",normal) or normal.startswith("/") or normal.endswith((".jpg",".jpeg",".png",".mp4",".npy",".npz",".onnx",".pt",".pth")):
            raise ValueError("private_path_in_public_report")

def publication_checks():
    checks={}
    for name,value in {"sample_identity":{"sample_id":"synthetic"},"nested_box":{"nested":{"bbox_xywh_px":[0,0,1,1]}},
                       "weight_reference":{"weights":"synthetic.onnx"},"absolute_windows":{"value":"C:/Users/example/image.jpg"},
                       "relative_backslash":{"value":"data\\training\\example\\frame.jpg"},"bare_photo":{"value":"synthetic.jpg"},
                       "per_image_probe_hash":{"image_sha256":"0"*64},"per_image_numeric_coordinates":{"pixel_coordinates":[0,1]}}.items():
        try:aggregate_only(value)
        except ValueError:checks[name+"_rejected"]=True
        else:checks[name+"_rejected"]=False
    aggregate_only({"weights_sha256":"0"*64,"AP50":.5,"max_abs_bbox_error_px":.0001})
    checks["aggregate_hashes_metrics_and_error_summaries_allowed"]=True
    if not all(checks.values()):raise ValueError("publication_privacy_contract_failed")
    return {"checks":checks,"passed":True,"accuracy_evidence":False}

def publish():
    folder=ROOT/"data/training/pothole-nano-rdd2020-v4"
    manifest=read(folder/"model.json");verification=read(folder/"verification-final.json")
    if verification.get("manifest_sha256")!=sha256(folder/"model.json") or not verification.get("artifact_contract_verified"):
        raise ValueError("verified_frozen_manifest_required")
    if sha256(ROOT/manifest["weights"])!=manifest["sha256"]:raise ValueError("weights_changed")
    native=read(ROOT/"docs/reports/POTHOLE-IMPROVEMENT-RESULTS.json")
    if native["protocol"].get("checkpoint")!="new validation AP50 maximum":raise ValueError("unexpected_checkpoint_selection_description")
    native["protocol"]["checkpoint_selection"]=native["protocol"].pop("checkpoint")
    fresh=read(folder/"app-fresh-comparison.json");regression=read(folder/"app-regression-comparison.json")
    native_fresh=read(folder/"heldout-predictions.json");native_baseline=read(folder/"baseline-new-heldout-predictions.json")
    native_reg=read(folder/"old-czech-regression-predictions.json")
    old_native=read(ROOT/"data/training/pothole-nano-rdd2020-v1/heldout-predictions.json")
    agreement={"fresh_candidate":metrics_equal(fresh["results"]["candidate"]["ap_metrics"],native_fresh["metrics"]),
        "fresh_baseline":metrics_equal(fresh["results"]["baseline"]["ap_metrics"],native_baseline["metrics"]),
        "regression_candidate":metrics_equal(regression["results"]["candidate"]["ap_metrics"],native_reg["metrics"]),
        "regression_baseline":metrics_equal(regression["results"]["baseline"]["ap_metrics"],old_native["metrics"])}
    if not all(agreement.values()):raise ValueError("native_app_aggregate_disagreement_requires_review")
    summary=read(folder/"summary.json");source=read(folder/"executed-source-audit.json")
    guards=read(ROOT/"docs/reports/POTHOLE-IMPROVEMENT-GUARD.json")
    original=read(folder/"initial-software-contract.executed.json")
    future=read(ROOT/"docs/reports/POTHOLE-IMPROVEMENT-CONTRACT-FUTURE.json")
    verification_keys=["status","manifest_sha256","checkpoint_sha256","numerical_parity_passed","postprocess_parity_passed",
        "artifact_contract_verified","preprocess_contract_verified","finite_output","output_shape","adapter_result_count",
        "runtime_python","opencv_preprocessing_version","app_backend","app_backend_version","torch_checkpoint_compared",
        "synthetic_input","app_vs_ort","app_vs_torch","ort_vs_torch","detection_postprocess_parity"]
    report={"research_only":True,"production_approved":False,"project_acceptance_eligible":False,
        "conclusion":"Improves the new Japan filename-block cohort; regresses on the previously examined Czech cohort. Not a universal or field accuracy improvement claim.",
        "model":{"architecture":manifest["architecture"],"input_size":manifest["input_size"],"classes":manifest["classes"],
            "manifest_sha256":sha256(folder/"model.json"),"weights_sha256":manifest["sha256"],"confidence":manifest["confidence"],
            "nms_iou":manifest["nms_iou"],"source_licenses":manifest["source_licenses"],"human_privacy_reviewed":False},
        "training":{"completed_epochs":summary["completed_epochs"],"best_epoch":summary["best_epoch"],
            "validation_AP50":summary["validation_map50"],"stop_reason":summary["stop_reason"],
            "noaug_L1_phase_reached":summary["noaug_phase_reached"],"test_used_for_selection":False},
        "source_audit":{"executed_trainer_sha256":source["trainer_sha256"],"executed_helper_sha256":source["helper_sha256"],
            "current_future_CLI_sha256":guards["current_future_CLI_sha256"],"postrun_change":guards["postrun_change"],
            "existing_weights_or_threshold_changed":False},
        "software_checks":{"original_executed_contracts":{"count":len(original["checks"]),"passed":original["passed"],
                "sha256":sha256(folder/"initial-software-contract.executed.json")},
            "future_CLI_contracts":{"count":len(future["checks"]),"passed":future["passed"]},
            "postrun_input_guards":{"count":len(guards["checks"]),"passed":guards["passed"]},
            "accuracy_evidence":False,"direct_original_audit":guards["direct_original_audit"],
            "clean_clone_simulation_audit":guards["clean_clone_simulation_audit"]},
        "positive_validation_numerical_verification":{k:verification[k] for k in verification_keys if k in verification},
        "native_app_AP_and_counts_exactly_equal":agreement,"native_comparison":native,
        "actual_app_fresh_comparison":fresh,"actual_app_regression_comparison":regression,
        "final_evaluation_did_not_reselect_weights_threshold_or_NMS":True,
        "publication_contract_checks":publication_checks(),
        "privacy":"Aggregate-only publication; original photos, per-image identities, boxes and weight binaries remain local and ignored.",
        "limits":["86 fresh images, 43 positive images, 62 GT potholes, 21 filename proxy blocks; actual physical routes unverified",
            "Balanced sample prevalence is not real driving prevalence; precision is not operational false alarms per km",
            "Czech regression degradation is retained and prevents any cross-country superiority claim",
            "CC-BY-NC-3.0 research data; Korean/ZED2i field validation and commercial/privacy clearance not established",
            "Mosaic-off/L1 phase was configured but NOT reached because validation plateau stopped at epoch50 before its epoch51 start"]}
    for value in (report,fresh,regression):aggregate_only(value)
    atomic_json(ROOT/"docs/reports/POTHOLE-IMPROVEMENT-FINAL.json",report)
    atomic_json(ROOT/"docs/reports/POTHOLE-APP-FRESH-COMPARISON-RESULTS.json",fresh)
    atomic_json(ROOT/"docs/reports/POTHOLE-APP-REGRESSION-COMPARISON-RESULTS.json",regression)
    print(json.dumps({"status":"aggregate_reports_published","native_app_exact_agreement":agreement,
        "fresh_AP50_change":fresh["map50_change"],"regression_AP50_change":regression["map50_change"],
        "manifest_sha256":sha256(folder/"model.json")}),flush=True)

if __name__=="__main__":publish()
