"""Release/annotation validators and connected-group dataset splitting."""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import random
import uuid
import numpy as np
from .paths import workspace_path, sha256
from .lane import measure_patch


def identity_tokens(row):
    tokens = []
    for key in ("sample_id", "source_sample_id", "parent_sample_id"):
        if row.get(key):
            tokens.append(("sample", row[key]))
    for key in ("leakage_group_id", "physical_object_id", "lane_segment_id", "source_hash", "run_id"):
        if row.get(key):
            tokens.append((key, row[key]))
    for obj in row.get("potholes", []):
        if obj.get("physical_object_id"):
            tokens.append(("physical_object_id", obj["physical_object_id"]))
    for lane in row.get("lanes", []):
        for key in ("lane_segment_id", "patch_id"):
            if lane.get(key):
                tokens.append((key, lane[key]))
    return tokens


def decode_rle(value):
    h, w = value["size"]
    if type(h) is not int or type(w) is not int or h <= 0 or w <= 0 or h * w > 16_777_216:
        raise ValueError("invalid_rle_size")
    counts = value["counts"]
    if not isinstance(counts, list) or any(type(x) is not int or x < 0 for x in counts) or sum(counts) != h * w:
        raise ValueError("invalid_rle_counts")
    flat = np.concatenate([np.full(n, i % 2, np.uint8) for i, n in enumerate(counts)])
    return flat.reshape((h, w), order="F")


def validate_annotation(sample):
    errors = []
    try:
        uuid.UUID(sample["sample_id"])
        if sample["dataset_schema_version"] != "1.0":
            errors.append("unsupported_dataset_schema")
        width, height = sample["image"]["width"], sample["image"]["height"]
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            errors.append("invalid_image_size")
        for obj in sample.get("potholes", []):
            uuid.UUID(obj["annotation_id"])
            uuid.UUID(obj["physical_object_id"])
            x, y, w, h = obj["bbox_xywh_px"]
            if not all(math.isfinite(v) for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
                errors.append("bbox_out_of_bounds")
            if obj.get("certainty") not in ("confirmed", "uncertain"):
                errors.append("invalid_certainty")
            if obj.get("certainty") == "uncertain" and sample.get("detection_eligible"):
                errors.append("uncertain_image_cannot_export_as_plain_negative")
        if sample.get("lane_masks"):
            masks = sample["lane_masks"]
            decoded = [decode_rle(masks[key]) for key in ("outline", "gap", "paint", "valid")]
            if sample["image"].get("coord_space", "left_original") == "left_original" and any(mask.shape != (height, width) for mask in decoded):
                errors.append("mask_image_size_mismatch")
            if np.any((decoded[2] > 0) & ~(decoded[0] > 0)):
                errors.append("paint_outside_outline")
            result = measure_patch(*decoded, outline_supported=sample.get("outline_supported", False))
            if "retention_pct" in sample and sample["retention_pct"] != result["retention_pct"]:
                errors.append("retention_label_mismatch")
        if sample.get("annotation_status") not in ("draft", "held", "qc_passed", "rejected"):
            errors.append("invalid_annotation_status")
        if sample.get("annotation_status") == "qc_passed" and sample.get("annotator_id") == sample.get("reviewer_id"):
            errors.append("independent_reviewer_required")
    except (KeyError, ValueError, TypeError) as exc:
        errors.append("malformed_annotation:" + str(exc))
    return errors


def validate_release(samples, *, now=None, purpose="deployment"):
    errors, groups, identifiers = [], defaultdict(set), set()
    now = now or datetime.now(timezone.utc)
    if purpose not in ("deployment","offline_research"):
        raise ValueError("unsupported_release_purpose")
    if not isinstance(samples, list):
        return {"valid": False, "samples": 0, "errors": [{"code": "release_must_be_sample_list"}]}
    for sample in samples:
        if not isinstance(sample, dict):
            errors.append({"code": "sample_must_be_object"})
            continue
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str):
            errors.append({"code": "sample_id_required"})
            continue
        try:
            uuid.UUID(sample_id)
        except ValueError:
            errors.append({"sample_id": sample_id, "code": "invalid_sample_uuid"})
        for field in ("source_id", "source_hash", "leakage_group_id", "path", "sha256", "annotation_version"):
            if not sample.get(field):
                errors.append({"sample_id": sample_id, "code": "missing_" + field})
        if sample_id in identifiers:
            errors.append({"sample_id": sample_id, "code": "duplicate_sample_id"})
        identifiers.add(sample_id)
        for token in identity_tokens(sample):
            groups[token].add(sample.get("split"))
        if sample.get("split") not in ("train", "validation", "test"):
            errors.append({"sample_id": sample_id, "code": "invalid_split"})
        if purpose=="deployment":
            if sample.get("privacy_status") != "approved" or sample.get("qc_status") != "qc_passed" or sample.get("release_scope")=="offline_research":
                errors.append({"sample_id": sample_id, "code": "approval_or_qc_missing"})
        else:
            if (sample.get("privacy_status")!="restricted_research" or sample.get("qc_status")!="publisher_annotations_machine_checked"
                    or sample.get("release_scope")!="offline_research" or sample.get("human_privacy_reviewed") is not False
                    or sample.get("not_for_deployment") is not True):
                errors.append({"sample_id":sample_id,"code":"research_scope_required"})
            try:
                evidence=workspace_path(sample["license_evidence_path"],exists=True)
                if sha256(evidence)!=sample["license_evidence_sha256"]:raise ValueError()
                policy=json.loads(evidence.read_text(encoding="utf-8"))
                if policy.get("research_allowed") is not True or not policy.get("license_id") or not policy.get("source_url") or not policy.get("license_url"):raise ValueError()
            except (KeyError,ValueError,TypeError,FileNotFoundError):
                errors.append({"sample_id":sample_id,"code":"research_license_evidence_missing"})
        if not sample.get("allowed_use"):
            errors.append({"sample_id": sample_id, "code": "usage_conditions_missing"})
        try:
            expiry = datetime.fromisoformat(sample["expires_at"].replace("Z", "+00:00"))
            if expiry.tzinfo is None or expiry <= now:
                errors.append({"sample_id": sample_id, "code": "expired_or_naive_timestamp"})
            if sample.get("path") and sha256(workspace_path(sample["path"], exists=True)) != sample.get("sha256"):
                errors.append({"sample_id": sample_id, "code": "file_hash_mismatch"})
        except (KeyError, ValueError, FileNotFoundError):
            errors.append({"sample_id": sample_id, "code": "invalid_expiry_or_file"})
    for key, splits in groups.items():
        if len(splits) > 1:
            errors.append({"group": key, "code": "split_leakage"})
    return {"valid": not errors, "samples": len(samples), "errors": errors}

def validate_research_release(samples, *, now=None):
    """Publisher labels + machine checks do not impersonate deployment approvals."""
    return validate_release(samples,now=now,purpose="offline_research")


def split_connected(samples, seed=2026):
    """Union all cross-frame identities before seeded group allocation."""
    parent = list(range(len(samples)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(samples):
        for token in identity_tokens(row):
            if token in seen:
                parent[root(i)] = root(seen[token])
            seen[token] = i
    components = defaultdict(list)
    for i, row in enumerate(samples):
        components[root(i)].append(row)
    buckets = list(components.items())
    random.Random(seed).shuffle(buckets)
    # Ratios are group-level targets; independence takes precedence over count.
    n = len(buckets)
    if n < 3:
        raise ValueError("at_least_three_independent_groups_required")
    test_n = max(1, round(n * .15))
    val_n = max(1, round(n * .15))
    result = []
    for index, (component_id, group) in enumerate(buckets):
        split = "test" if index < test_n else "validation" if index < test_n + val_n else "train"
        for sample in group:
            result.append(sample | {"split": split, "leakage_group_id": f"component-{component_id:06d}"})
    return result
