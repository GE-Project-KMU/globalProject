"""Frozen research holdout using the actual application CPU adapter, not native training inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys, time
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.dataset import validate_research_release
from agent.models import OnnxModel
from agent.evaluate import detection_metrics
from agent.paths import workspace_path,sha256,atomic_json


def evaluate(manifest_path,output):
    manifest_path=workspace_path(manifest_path,exists=True)
    frozen_hash=sha256(manifest_path)
    model=OnnxModel(manifest_path);artifact=dict(model.manifest)
    if not artifact.get("research_only") or artifact.get("task") not in ("pothole","redaction"):
        raise ValueError("research_detector_required")
    release=workspace_path(artifact["source_release_path"],exists=True)
    rows=json.loads(release.read_text(encoding="utf-8"))
    state=validate_research_release(rows)
    if not state["valid"] or artifact["dataset_release"]!="manifest-sha256:"+sha256(release):
        raise ValueError("invalid_or_changed_research_release")
    chosen=sorted([r for r in rows if r["split"]=="test"],key=lambda r:r["sample_id"])
    if not chosen:raise ValueError("test_required")
    if any(r.get("ignore_bboxes_xywh_px") for r in chosen):raise ValueError("ignored_region_protocol_requires_explicit_evaluator")
    # AP benchmark floor only, in memory. The selected operating threshold and
    # frozen artifact are never rewritten or selected using these test results.
    confidence=artifact.get("confidence",.35)
    if confidence<.01:raise ValueError("operating_confidence_below_benchmark_floor")
    model.manifest=dict(artifact,confidence=.01)
    truth=[];predictions=[];started=time.perf_counter()
    classes=artifact["classes"]
    for row in chosen:
        known=row.get("known_label_coverage",classes)
        truth.extend({"sample_id":row["sample_id"],"class":o["class"],"bbox_xywh_px":o["bbox_xywh_px"]}
                     for o in row["objects"] if o["class"] in known)
        image=cv2.imread(str(workspace_path(row["path"],exists=True)))
        for detection in model.predict(image):
            if detection["class"] in known:predictions.append(dict(detection,sample_id=row["sample_id"]))
    if sha256(manifest_path)!=frozen_hash:raise ValueError("model_changed_during_holdout")
    metrics=detection_metrics(truth,predictions,classes)
    operating=detection_metrics(truth,[p for p in predictions if p["confidence"]>=confidence],classes)
    for value in operating["classes"].values():
        value["precision"]=value["tp"]/(value["tp"]+value["fp"]) if value["tp"]+value["fp"] else 0.
    report={"scope":"offline_research_actual_application_adapter","evaluated_at":datetime.now(timezone.utc).isoformat(),
            "task":artifact["task"],"samples":len(chosen),"manifest_sha256":frozen_hash,"weights_sha256":artifact["sha256"],
            "source_release_sha256":sha256(release),"backend":model.backend,"backend_version":model.backend_version,
            "evaluator_code_sha256":sha256(workspace_path("scripts/evaluate_detector_model.py")),
            "application_adapter_code_sha256":sha256(workspace_path("agent/models.py")),
            "input_size":artifact["input_size"],"ap_confidence_floor":.01,"frozen_operating_confidence":confidence,
            "heldout_metrics":metrics,"operating_metrics":operating,"elapsed_seconds_including_metrics":time.perf_counter()-started,
            "known_class_conditional":any("known_label_coverage" in r for r in chosen),
            "test_used_for_model_or_threshold_selection":False,"production_approved":False,"project_acceptance_passed":False,
            "independent_route_verified":False,"human_privacy_reviewed":False,
            "limits":["Public publisher labels plus machine QC; no independent vehicle acceptance",
                      "Unknown face/plate class annotations are not treated as exhaustive negatives",
                      "No route-event recall, wear-ratio MAE or full-pipeline FPS claim"]}
    output=workspace_path(output)
    atomic_json(output,report)
    atomic_json(output.with_name(output.stem+"-predictions.json"),{"truth":truth,"predictions":predictions,"report_sha256":sha256(output)})
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("model");parser.add_argument("output");args=parser.parse_args()
    print(json.dumps(evaluate(args.model,args.output),ensure_ascii=False,indent=2))
