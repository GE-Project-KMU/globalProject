"""Publish aggregate research results without images, identities or raw predictions."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.paths import workspace_path, sha256, atomic_json


def read(path):
    return json.loads(workspace_path(path, exists=True).read_text(encoding="utf-8"))


def artifact(path):
    model = read(path)
    if not model.get("research_only") or model.get("production_approved") is not False:
        raise ValueError("explicit_research_nonapproval_required")
    if sha256(workspace_path(model["weights"], exists=True)) != model["sha256"]:
        raise ValueError("weights_changed")
    release = workspace_path(model["source_release_path"], exists=True)
    if model["dataset_release"] != "manifest-sha256:" + sha256(release):
        raise ValueError("source_release_changed")
    rows = read(release)
    verification_path = str(Path(path).parent / "verification-final.json")
    verification = read(verification_path)
    if verification["manifest_sha256"] != sha256(workspace_path(path)) or verification["weight_sha256"] != model["sha256"]:
        raise ValueError("verification_does_not_match_frozen_model")
    return {"manifest": path, "manifest_sha256": sha256(workspace_path(path)),
            "weights": model["weights"], "weights_sha256": model["sha256"],
            "release_sha256": sha256(release), "split_counts": dict(Counter(r["split"] for r in rows)),
            "classes": model["classes"], "input_size": model["input_size"],
            "source_licenses": model["source_licenses"], "research_only": True,
            "production_approved": False, "independent_route_verified": False,
            "earliest_source_expiry": min(r["expires_at"] for r in rows),
            "verification_report_sha256": sha256(workspace_path(verification_path)),
            "verification": {k: verification[k] for k in ("status", "numerical_parity_passed", "app_backend", "app_backend_version",
                "app_vs_ort", "app_vs_torch", "ort_vs_torch", "mask_agreement", "detection_postprocess_parity") if k in verification}}


def lane(folder, manifest="lane.json", holdout="heldout.json"):
    base = "data/training/" + folder + "/"
    result = artifact(base + manifest)
    training, evaluated = read(base + "training.json"), read(base + holdout)
    result.update(task="lane", epochs_completed=len(training["history"]),
                  best_validation_training_iou=max(r["validation_paint_iou"] for r in training["history"]),
                  paint_threshold=evaluated["paint_threshold"],
                  test_metrics={k: evaluated[k] for k in ("paint_iou", "paint_precision", "paint_recall", "paint_dice")},
                  test_confusion=evaluated["confusion"],
                  research_target_passed=evaluated["target_passed_on_this_research_holdout"],
                  project_acceptance_passed=False, retention_mae_pp=None,
                  test_report_sha256=sha256(workspace_path(base + holdout)),
                  training_code_sha256=training.get("code_sha256"),
                  training_code_commit="47b950d" if folder == "lane-rtk-v1" else "70bcf06")
    return result


def detector(folder):
    base = "data/training/" + folder + "/"
    result = artifact(base + "model.json")
    config, history, summary = (read(base + name) for name in ("training-config.json", "training-history.json", "summary.json"))
    application = read(base + "app-heldout.json")
    metadata_audit = read(base + "manifest-metadata-audit.json")
    if application["manifest_sha256"] != result["manifest_sha256"]:
        raise ValueError("application_evaluation_model_mismatch")
    result.update(task=read(base + "model.json")["task"], epochs_completed=len(history["history"]),
                  best_epoch=summary["best_epoch"], best_validation_map50=summary["validation_map50"],
                  native_gpu_test_metrics=summary["heldout_metrics"],
                  test_metrics=application["heldout_metrics"], operating_metrics=application["operating_metrics"],
                  application_evaluation_sha256=sha256(workspace_path(base + "app-heldout.json")),
                  confidence=summary["confidence_calibration"]["selected_confidence"],
                  known_class_conditional_evaluation=config["known_class_evaluation"],
                  project_acceptance_passed=False,
                  summary_sha256=sha256(workspace_path(base + "summary.json")),
                  training_code_commit=metadata_audit["source_repo_commit"],
                  training_code_sha256=metadata_audit["executed_training_script_sha256"],
                  metadata_only_revision=metadata_audit)
    result["research_target_passed"] = (application["heldout_metrics"]["map50"] >= .8 if result["task"] == "pothole"
                                         else all((r.get("recall") or 0) >= .95 for r in application["operating_metrics"]["classes"].values()))
    return result


def main():
    # Every requested candidate must have finished before a final report exists.
    report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
              "scope": "private_offline_public_data_research", "actual_gradient_training": True,
              "app_backend": "ONNX Runtime CPU", "external_weights_published": False,
              "project_acceptance_passed": False,
              "models": [lane("lane-rtk-v1", "lane-ort.json", "heldout-ort.json"), lane("lane-a2d2-v1"),
                         detector("pothole-nano-rdd2020-v1"), detector("redaction-nano-v3")],
              "preserved_failures": [
                  "RDD2022 third-party converted labels: semantic mismatch; run stopped and quarantined, not final weights",
                  "OpenCV5 default DNN engine: RTK logits overflow; earlier zero-IoU invalidated, same weights reevaluated with ORT",
                  "Redaction v2 cropped test failed original full-scene challenge; v3 uses original frames and newly frozen unseen source test",
                  "Strict numerical verification failures remain recorded; raw allclose and postprocess agreement are distinct"],
              "limits": ["No independent ZED 2i road/vehicle acceptance", "No expected-outline/gap/valid GT for wear-ratio MAE",
                         "Face/plate metrics conditional on known class; other-class annotation incomplete",
                         "No fabricated human privacy review, commercial-use clearance or external redistribution permission",
                         "Source expiry/revocation still blocks model reuse; no automatic expiry extension"]}
    atomic_json(workspace_path("docs/reports/MODEL-TRAINING-SUMMARY.json"), report)
    print(json.dumps({"status": "research_summary_written", "models": len(report["models"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
