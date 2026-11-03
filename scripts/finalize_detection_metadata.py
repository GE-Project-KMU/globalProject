"""Dedicated post-hoc metadata amendment CLI for the completed research runs.

Never trains, evaluates, or changes model weights, confidence or NMS. Historical
executed experiment source remains the immutable d3c040b snapshot.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json


def require(condition,reason):
    if not condition:raise ValueError(reason)


def finalize(folder):
    folder=workspace_path(folder,exists=True);path=folder/"model.json"
    manifest=json.loads(path.read_text());config=json.loads((folder/"training-config.json").read_text())
    provenance=json.loads(workspace_path(manifest["training_provenance"],exists=True).read_text())
    release_path=workspace_path(manifest["source_release_path"],exists=True);release=json.loads(release_path.read_text())
    require(manifest.get("research_only") is True and manifest.get("production_approved") is False and
            manifest.get("project_acceptance_eligible") is False,"research_only_nonaccepted_model_required")
    require(config.get("test_used_for_selection") is False and provenance.get("test_split_exported") is False,"test_training_or_selection_not_excluded")
    release_hash=sha256(release_path)
    require(release_hash==config.get("release_sha256")==provenance.get("release_sha256") and
            manifest.get("dataset_release")=="manifest-sha256:"+release_hash,"release_hash_mismatch")
    splits={row["sample_id"]:row["split"] for row in release}
    require(all(splits.get(asset.get("sample_id")) in ("train","validation") for asset in provenance["assets"]),"non_training_validation_asset_present")
    require(len(provenance["assets"])==sum(row["split"] in ("train","validation") for row in release),"training_validation_asset_count_mismatch")
    require((folder/"heldout-predictions.json").is_file() and (folder/"summary.json").is_file(),"completed_heldout_evidence_required")
    weight_hash=sha256(manifest["weights"]);require(weight_hash==manifest.get("sha256"),"model_weight_hash_mismatch")
    additions={"test_split_used":False,"test_evaluated":True,"test_split_used_scope":"training_and_selection"}
    audit_path=folder/"manifest-metadata-audit.json"
    if all(manifest.get(key)==value for key,value in additions.items()) and audit_path.exists():return json.loads(audit_path.read_text())
    before=sha256(path);original=dict(manifest)
    atomic_json(folder/"model.original-training-manifest.json",original)
    manifest.update(additions);atomic_json(path,manifest)
    contract=json.loads((ROOT/"docs/reports/DETECTION-TRAINING-CONTRACT.json").read_text())
    report={"at":datetime.now(timezone.utc).isoformat(),"operation":"metadata_only_test_usage_scope_clarification",
        "manifest_before_sha256":before,"manifest_after_sha256":sha256(path),"changed_fields":additions,
        "weights_sha256_unchanged":weight_hash,"confidence_unchanged":manifest["confidence"],"nms_iou_unchanged":manifest["nms_iou"],
        "verified_train_validation_assets":len(provenance["assets"]),"test_assets_used_for_training_or_selection":0,
        "release_sha256":config["release_sha256"],"source_repo_commit":"d3c040b",
        "executed_training_script_sha256":contract["training_script_sha256"],
        "evidence":"Original fixed script, test_used_for_selection:false config, test_split_exported:false provenance, train/validation-only asset membership, completed post-freeze heldout artifact."}
    atomic_json(audit_path,report);return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("folders",nargs="+");args=parser.parse_args()
    for folder in args.folders:
        report=finalize(folder);print(json.dumps({"folder":folder,"manifest_sha256":report["manifest_after_sha256"],"weights_unchanged":True}))
