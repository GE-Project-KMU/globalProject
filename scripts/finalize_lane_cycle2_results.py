"""Metadata-only final lane aggregation; never performs inference or model selection."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.paths import atomic_json, sha256
EXPECTED_ARTIFACT_DIGEST = "23f3b9eb6d1724ce91c842af10bc2c9eba54628c9906155b7bcdff64997860ce"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validated_metrics(value):
    c = value["confusion"]
    if set(c) != {"tp", "fp", "fn", "tn", "ignored"} or any(type(v) is not int or v < 0 for v in c.values()):
        raise ValueError("nonnegative_integer_confusion_required")
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    denominators = {"paint_iou": tp+fp+fn, "paint_precision": tp+fp,
                    "paint_recall": tp+fn, "paint_dice": 2*tp+fp+fn}
    metrics = {}
    for name, denominator in denominators.items():
        if denominator <= 0:
            raise ValueError("supported_paint_metric_denominator_required")
        number = (2*tp if name == "paint_dice" else tp) / denominator
        actual = value.get(name)
        if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, number, rel_tol=0, abs_tol=1e-12):
            raise ValueError("aggregate_metric_confusion_mismatch")
        metrics[name] = number
    return {"confusion": c, **metrics}


def collect():
    # This pure-metadata validator binds completed training, weights, checkpoint,
    # VAL selection, canonical3/all179 proofs and all original source pins.
    from scripts import evaluate_lane_retry1 as evaluation
    frozen = evaluation.validated_freeze()
    if frozen["artifact_digest"] != EXPECTED_ARTIFACT_DIGEST:
        raise ValueError("this_final_aggregate_requires_original_retry1_artifact_digest")
    folder = evaluation.DEST
    results, evidence = {}, {}
    for role, count in evaluation.COUNTS.items():
        plan_path = folder / f"{role}-plan.json"
        plan = read(plan_path)
        if plan != evaluation.comparison_plan(frozen, role):
            raise ValueError("original_one_shot_plan_changed")
        result_path = folder / f"{role}-comparison.json"
        result = read(result_path)
        attempt = read(folder / f"{role}-attempt.json")
        completion = read(folder / f"{role}-completion.json")
        binding = {"plan_sha256": sha256(plan_path), "artifact_digest": frozen["artifact_digest"], "selection_allowed": False}
        if attempt != {"status": "started", **binding} or completion != {"status": "completed", **binding, "samples": count, "result_sha256": sha256(result_path)}:
            raise ValueError("completed_one_shot_attempt_result_binding_required")
        if (result["samples"] != count or result["plan_sha256"] != binding["plan_sha256"]
                or result["benchmark_sha256"] != plan["benchmark_sha256"] or result["scope"] != plan["role"]
                or result.get("test_used_for_model_or_threshold_selection") is not False
                or result.get("physical_route_independence_verified") is not False
                or result.get("project_acceptance_passed") is not False
                or result.get("checkpoint_raw_tensor_parity_all_models_passed") is not True
                or result.get("diagnostic_accuracy_only") is not False):
            raise ValueError("frozen_accuracy_result_scope_required")
        comparison_freeze = read(folder / f"{role}-comparison-freeze.json")
        if comparison_freeze["plan_sha256"] != binding["plan_sha256"] or comparison_freeze["benchmark_sha256"] != plan["benchmark_sha256"] or comparison_freeze.get("test_selection_allowed") is not False:
            raise ValueError("comparison_freeze_binding_required")
        per_image_path = folder / f"{role}-comparison-per-image.json"
        per_image = read(per_image_path)
        if len(per_image) != count or len({r["sample_id"] for r in per_image}) != count:
            raise ValueError("complete_unique_private_per_image_results_required")
        safe_models = {}
        for name in ("baseline", "candidate"):
            value = result["results"][name]
            metadata = comparison_freeze["models"][name]
            if any(value.get(k) != v for k, v in metadata.items()):
                raise ValueError("comparison_model_metadata_changed")
            ref = frozen[f"{name}_manifest"]
            model = read(ROOT / ref["path"])
            if value["manifest_sha256"] != ref["sha256"] or value["weights_sha256"] != model["sha256"] or value["paint_threshold"] != model["paint_threshold"]:
                raise ValueError("selected_model_or_threshold_changed")
            aggregate = validated_metrics(value)
            summed = {k: sum(row[name]["confusion"][k] for row in per_image) for k in aggregate["confusion"]}
            if summed != aggregate["confusion"]:
                raise ValueError("private_per_image_aggregate_mismatch")
            safe_models[name] = {"manifest_sha256": ref["sha256"], "weights_sha256": value["weights_sha256"], "paint_threshold": value["paint_threshold"], **aggregate}
        change = safe_models["candidate"]["paint_iou"] - safe_models["baseline"]["paint_iou"]
        if not math.isclose(change, result["paint_iou_change"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("IoU_change_mismatch")
        results[role] = {"samples": count, "scope": result["scope"], "models": safe_models,
                         "paint_iou_change_percentage_points": 100*change}
        evidence[role] = {suffix: sha256(folder / f"{role}-{suffix}.json") for suffix in ["plan", "attempt", "completion", "comparison", "comparison-freeze", "comparison-per-image"]}
    info = read(ROOT / frozen["candidate_manifest"]["path"])
    history = read(ROOT / frozen["training_history"]["path"])
    fresh_iou = results["fresh"]["models"]["candidate"]["paint_iou"]
    return {"format": "lane-cycle2-retry1-final-public-accuracy-v1", "artifact_digest": frozen["artifact_digest"],
            "aggregation_source_sha256": sha256(Path(__file__)), "evaluation_source_sha256": frozen["evaluation_source_sha256"],
            "evaluation_policy_sha256": frozen["parent_policy"]["sha256"],
            "candidate_manifest_sha256": frozen["candidate_manifest"]["sha256"], "candidate_weights_sha256": frozen["weights_sha256"],
            "candidate_checkpoint_sha256": frozen["checkpoint"]["sha256"],
            "canonical_proof_sha256": [r["sha256"] for r in frozen["canonical_proofs"]], "all179_proof_sha256": frozen["all179_proof"]["sha256"],
            "epochs_completed": len(history["history"]), "actual_optimizer_updates": history["actual_updates"],
            "selected_epoch": info["best_epoch"], "development_mean_recording_IoU": info["validation_paint_iou"],
            "development_score_is_independent_accuracy": False, "results": results, "evidence_sha256": evidence,
            "paint_IoU_target": .75, "fresh_paint_IoU_target_passed": fresh_iou >= .75,
            "fresh_target_shortfall_percentage_points": 100*max(0, .75-fresh_iou),
            "TEST_used_for_selection": False, "post_TEST_reselection_or_training_performed": False,
            "additional_inference_performed_by_aggregation": False, "GPU_used_by_aggregation": False,
            "physical_route_independence_verified": False, "wear_ratio_accuracy_validated": False,
            "production_approved": False, "project_acceptance_eligible": False, "operational_promotion_performed": False,
            "source_expiries_extended": False, "raw_images_or_per_case_metadata_published": False,
            "limits": ["Fresh recording-heldout only; physical roads/dates/person independence unverified",
                       "Publisher paint segmentation proxy; no lane outline/gap/valid wear-ratio ground truth",
                       "Existing180 cohort is regression only; no independent-performance claim",
                       "Each model uses its own frozen VAL-selected threshold; training and calibration gains are not isolated",
                       "Shared-host single-model CPU latency is not10FPS full-pipeline evidence",
                       "Numerical CPU parity does not establish GPU or production support"]}


def markdown(report):
    rows = []
    for role, label in [("fresh", "새 기록 분리 시험"), ("regression", "기존 회귀 시험")]:
        value = report["results"][role]
        rows.append(f"| {label} | {value['samples']} | {100*value['models']['baseline']['paint_iou']:.2f}% | {100*value['models']['candidate']['paint_iou']:.2f}% | +{value['paint_iou_change_percentage_points']:.2f}%p |")
    return "\n".join(["# 차선 cycle2 retry1 평가 결과", "", "차선 paint 분할 IoU는 새 기록 분리 시험에서 **68.92% → 73.07%(+4.15%p)**로 개선됐습니다. 목표 **75%에는 1.93%p 부족**하여 운영 승격은 하지 않았습니다.", "",
        "| 평가 | 이미지 | 기존 v4 IoU | retry1 IoU | 변화 |", "|---|---:|---:|---:|---:|", *rows, "",
        "공식 A2D2 자료에서 촬영 기록을 나눈 새 시험119장과 기존 회귀180장을 각각 한 번 평가했습니다. 물리적으로 다른 도로·촬영일의 독립성은 확인하지 못했습니다. 기존 회귀 자료는 독립 성능 근거로 해석하지 않습니다.", "",
        "사전 고정한20epoch를 완료했고 실제 optimizer 업데이트는13,952회입니다. checkpoint는 개발VAL179장의 기록별 평균 IoU로 epoch18을 선택했습니다. 개발 수치76.48%는 독립 시험 정확도가 아닙니다. 기존 모델 threshold.2와 새 모델.6은 각각 VAL에서 고정했으며 TEST 결과를 본 뒤 모델·threshold를 다시 선택하거나 학습하지 않았습니다. 개선폭은 학습과 VAL 보정의 합친 효과이며 둘을 분리한 실험은 아닙니다.", "",
        "실제 양성VAL3장과 전체VAL179장에서 Windows AMD64 CPU ORT1.31.1, 격리 ORT1.31.1, 원 FP32 PyTorch를 비교했습니다. 기본 raw 허용오차(rtol.001/atol.0001)와 정확한 마스크 일치 검사를 모두 통과했습니다. 허용오차를 완화하지 않았으며 이 결과를 GPU 지원으로 소급하지 않습니다.", "",
        "평가 정답은 원 제공자의 paint 분할 proxy입니다. 마모차선 윤곽·점선 간격·유효 관측영역의 실제 마모율 정답 검증은 포함하지 않습니다. 공유 호스트에서 측정한 단일 모델 CPU 지연은 전체 파이프라인10FPS 달성 근거가 아닙니다. 모델은 연구용이며 개인정보 검수·배포 승인·운영 승격은 수행하지 않았습니다. 원 보관기한도 연장하지 않았습니다.", "",
        "원 동시GPU 실패와 환경 import 실패 자료는 보존했습니다. 이 문서 작성에는 추가 추론·GPU 실행·TEST 재평가를 하지 않았습니다. 사진·개별 sample ID·경로·GT 좌표는 공개 집계에서 제외했습니다.", "",
        "로컬의 고정된 비공개 증거가 있을 때 다음 명령으로 집계 연결을 다시 검증할 수 있습니다. 이 명령은 학습이나 추론을 실행하지 않습니다.", "", "```powershell", ".venv/Scripts/python.exe scripts/finalize_lane_cycle2_results.py", "```", "",
        "모델 가중치·개별 평가자료는 Git에 포함하지 않습니다. 공개 수치와 증거 SHA는 [집계 JSON](reports/LANE-CYCLE2-RETRY1-RESULTS.json), CPU 수치 검증은 [검증 집계](reports/LANE-CYCLE2-RETRY1-NUMERIC.json)에 기록했습니다.", ""])


def main():
    report = collect()
    output = ROOT / "docs/reports/LANE-CYCLE2-RETRY1-RESULTS.json"
    if output.exists():
        if read(output) != report:
            raise ValueError("existing_public_result_must_not_change")
    else:
        atomic_json(output, report)
    text = markdown(report)
    doc = ROOT / "docs/LANE-CYCLE2-RESULTS.md"
    if doc.exists() and doc.read_text(encoding="utf-8") != text:
        doc = ROOT / "docs/LANE-CYCLE2-RETRY1-RESULTS.md"
    if doc.exists() and doc.read_text(encoding="utf-8") != text:
        raise ValueError("existing_results_document_must_be_preserved")
    if not doc.exists():
        doc.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "aggregate_verified", "report_sha256": sha256(output), "document": doc.relative_to(ROOT).as_posix(), "additional_inference": False}))


if __name__ == "__main__":
    main()
