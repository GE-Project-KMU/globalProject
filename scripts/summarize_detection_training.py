"""Publish aggregate local experiment evidence, never raw images/predictions."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json


def read(path):return json.loads(workspace_path(path,exists=True).read_text(encoding="utf-8"))


def collect(folder,command):
    folder=workspace_path(folder,exists=True)
    manifest=read(folder/"model.json");summary=read(folder/"summary.json");config=read(folder/"training-config.json")
    calibration=read(folder/"confidence-calibration.json")
    audit=read(folder/"manifest-metadata-audit.json")
    if not manifest.get("research_only") or manifest.get("production_approved") or manifest.get("project_acceptance_eligible"):
        raise ValueError("unexpected_acceptance_flags")
    return {"architecture":manifest["architecture"],"task":manifest["task"],"input_size":manifest["input_size"],
        "classes":manifest["classes"],"model_path":manifest["weights"],"model_sha256":manifest["sha256"],
        "model_manifest_sha256":sha256(folder/"model.json"),"metadata_amendment_audit":audit,
        "test_split_used":manifest["test_split_used"],"test_evaluated":manifest["test_evaluated"],"test_split_used_scope":manifest["test_split_used_scope"],
        "source_release_sha256":config["release_sha256"],"source_licenses":manifest["source_licenses"],
        "command":command,"epochs_requested":config["epochs"],"best_epoch":summary["best_epoch"],"seed":config["seed"],
        "images":{"train":summary["training_images"],"validation":summary["validation_images"],"test":summary["test_images"]},
        "validation_map50":summary["validation_map50"],"heldout_map50":summary["heldout_metrics"]["map50"],
        "heldout_metrics":summary["heldout_metrics"],"frozen_operating_confidence":calibration["selected_confidence"],
        "confidence_selection_data":calibration["selection_data"],"calibration_frozen_at":calibration["frozen_at"],
        "operating_metrics":summary["operating_metrics"],"elapsed_seconds_shared_training":summary["elapsed_seconds"],
        "shared_gpu_training":True,"isolated_speed_claim":False,"environment":summary["environment"],
        "known_class_evaluation":config.get("known_class_evaluation",False),"partial_annotation_limit":config.get("partial_annotation_limit"),
        "research_only":True,"production_approved":False,"project_acceptance_eligible":False,"independent_route_verified":False,
        "human_privacy_reviewed":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pothole",default="data/training/pothole-nano-rdd2020-v1")
    parser.add_argument("--redaction",default="data/training/redaction-nano-v3")
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--output",default="docs/reports/DETECTION-TRAINING-RESULTS.json")
    args=parser.parse_args()
    commands={
        "pothole":"python scripts/setup_training.py run scripts/train_yolox_research.py data/training/research-pothole-rdd2020/release.json data/training/pothole-nano-rdd2020-v1 --task pothole --epochs 30 --batch-size 8 --learning-rate 0.0005 --input-size 640",
        "redaction":"python scripts/setup_training.py run scripts/train_yolox_research.py data/training/research-redaction-fullframe/release.json data/training/redaction-nano-v3 --task redaction --epochs 30 --batch-size 8 --learning-rate 0.0005 --input-size 640 --max-labels 512 --known-class-evaluation"}
    results={"pothole":collect(args.pothole,commands["pothole"]),"redaction":collect(args.redaction,commands["redaction"])}
    baseline=workspace_path("data/training/redaction-nano-v2/summary.json")
    if baseline.exists():
        crop=read(baseline);challenge=read(baseline.parent/"fullscene-heldout-predictions.json")
        results["rejected_crop_only_baseline"]={"crop_heldout_map50":crop["heldout_metrics"]["map50"],
            "original_frame_challenge_map50":challenge["metrics"]["map50"],"reason":"Crop localization performance did not transfer to full frames; excluded as usable runtime model."}
    report={"schema_version":"1.0","created_at":datetime.now(timezone.utc).isoformat(),"source_repo_commit":args.source_commit,
        "executed_trainer_script_sha256":read(workspace_path(args.pothole)/"manifest-metadata-audit.json")["executed_training_script_sha256"],
        "current_trainer_script_sha256":sha256(ROOT/"scripts/train_yolox_research.py"),
        "post_run_trainer_change":"Add automatic executed code/argv and explicit training-selection test usage metadata; no algorithm/weights/threshold changes to completed experiments.",
        "source_preparation_script_sha256":sha256(ROOT/"scripts/research_detection_data.py"),"runs":results,
        "metric_protocol":"One-to-one class/image IoU0.5, confidence-floor.01 AP101. Operating TP/FP/FN and precision/recall use frozen validation-selected confidence, not test tuning.",
        "source_qc_reports":["docs/reports/POTHOLE-RDD2020-SOURCE-QC.json","docs/reports/REDACTION-FULLFRAME-V3-QC.json"],
        "math_contract_report":"docs/reports/DETECTION-TRAINING-CONTRACT.json","raw_images_or_per_image_predictions_published":False,
        "project_acceptance":"Not granted. Public research data, no route audit or human privacy approval; license/privacy/field evaluation remain separate."}
    atomic_json(args.output,report)
    print(json.dumps({"report":args.output,"pothole_map50":results["pothole"]["heldout_map50"],"redaction_map50":results["redaction"]["heldout_map50"]}))


if __name__=="__main__":main()
