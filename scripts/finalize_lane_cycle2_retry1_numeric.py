"""Publish aggregate CPU VAL parity only; preserve artifacts and private proofs."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.paths import atomic_json, sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    folder = ROOT / "data/training/lane-cycle2-v5-retry1"
    manifest_path = folder / "lane.json"
    info = read(manifest_path)
    full_path = folder / "full-validation-131.json"
    full = read(full_path)
    queue = read(folder / "CPU-numeric-queue-receipt.json")
    supervisor = read(ROOT / "data/training/public-nextcycle/lane-retry1-training-supervisor.json")
    history = read(folder / "history.json")
    if (supervisor.get("status") != "completed" or supervisor.get("child_exit_code") != 0
            or supervisor.get("own_child_running_after_cleanup") is not False
            or supervisor.get("sources_unchanged") is not True):
        raise ValueError("completed_bound_training_and_cleanup_required")
    if (full.get("complete") is not True or len(full["cases"]) != 179
            or full.get("all_raw_passed") is not True or full.get("all_post_passed") is not True
            or any(not c["raw_passed"] or not c["post_passed"] for c in full["cases"])):
        raise ValueError("complete_strict_179_VAL_required")
    if (full["manifest_sha256"] != sha256(manifest_path) or full["weights_sha256"] != info["sha256"]
            or full["checkpoint_sha256"] != sha256(folder / "lane_best.pt")
            or sha256(ROOT / info["weights"]) != info["sha256"]):
        raise ValueError("unchanged_artifact_proofs_required")
    if (full["rtol"] != .001 or full["atol"] != .0001
            or full["reference_runtime"]["ort"] != "1.31.1"
            or full["app_runtime"]["version"] != "1.31.1"
            or full["reference_runtime"].get("GPU_initialized") is not False):
        raise ValueError("fixed_CPU_runtime_tolerance_required")
    if (queue.get("complete") is not True or queue.get("sources_unchanged") is not True
            or queue.get("GPU_used") is not False or queue.get("TEST_used") is not False
            or queue.get("manifest_sha256") != sha256(manifest_path)
            or queue.get("checkpoint_sha256") != full["checkpoint_sha256"]
            or any(s["exit_code"] != 0 for s in queue["steps"])
            or list((folder / "numeric-working").glob("*.npy"))):
        raise ValueError("completed_CPU_queue_and_working_array_cleanup_required")
    canonical = []
    for index, name in enumerate(["verification-final.json", "verification-val-1.json", "verification-val-2.json"]):
        path = folder / name
        proof = read(path)
        if any(proof.get(k) is not True for k in ["artifact_contract_verified", "numerical_parity_passed", "postprocess_parity_passed"]):
            raise ValueError("canonical_raw_and_post_contract_required")
        if (proof.get("manifest_sha256") != sha256(manifest_path)
                or proof.get("weight_sha256") != info["sha256"]
                or proof.get("checkpoint_sha256") != full["checkpoint_sha256"]
                or queue["steps"][index].get("output_sha256") != sha256(path)):
            raise ValueError("canonical_same_artifact_and_queue_binding_required")
        canonical.append(sha256(path))
    report = {
        "format": "lane-cycle2-retry1-CPU-VAL-numeric-v1",
        "scope": "training_and_CPU_numeric_completed_before_final_accuracy_evaluation",
        "manifest_sha256": sha256(manifest_path), "weights_sha256": info["sha256"],
        "checkpoint_sha256": full["checkpoint_sha256"],
        "training_source_sha256": supervisor["learning_source_sha256"],
        "learning_plan_sha256": supervisor["learning_plan_sha256"],
        "numeric_source_sha256": queue["source_sha256"],
        "full_VAL_proof_sha256": sha256(full_path), "canonical_proof_sha256": canonical,
        "epochs_completed": len(history["history"]), "actual_optimizer_updates": history["actual_updates"],
        "selected_epoch": info["best_epoch"], "development_mean_recording_IoU": info["validation_paint_iou"],
        "paint_threshold": info["paint_threshold"], "development_score_is_independent_accuracy": False,
        "canonical_positive_VAL_cases": 3, "full_VAL_cases": 179, "raw_pair_comparisons_per_case": 3,
        "raw_rtol": .001, "raw_atol": .0001, "raw_failures": 0, "exact_mask_failures": 0,
        "supported_profile": info["required_runtime_profile"], "reference_runtime": full["reference_runtime"],
        "app_runtime": full["app_runtime"], "GPU_used_for_this_numeric_report": False,
        "GPU_validation_included_in_this_CPU_report": False,
        "working_arrays_remaining": 0, "numeric_workers_exited_successfully": True,
        "TEST_included_in_this_report": False, "production_approved": False,
        "project_acceptance_eligible": False, "source_expiries_extended": False,
        "limits": ["Windows CPU ORT1.31.1 only; no general portability or GPU claim",
                   "VAL numerical parity does not establish independent segmentation accuracy",
                   "Failed original concurrent-GPU experiments remain preserved; retry is a full restart"]}
    output = ROOT / "docs/reports/LANE-CYCLE2-RETRY1-NUMERIC.json"
    if output.exists():
        if read(output) != report:
            raise ValueError("existing_public_aggregate_must_not_change")
    else:
        atomic_json(output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
