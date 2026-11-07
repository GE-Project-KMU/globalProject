"""Validate AI visual drafts; validation never grants training or evaluation approval."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent.dataset import validate_annotation
DEADLINE = datetime(2026, 11, 14, 14, 59, tzinfo=timezone.utc)
MAX_BYTES = 40_000_000
MAX_PIXELS = 16_777_216
SHA = re.compile(r"[0-9a-f]{64}\Z")


def scoped_file(value):
    """Reject lexical escapes before resolving any filesystem path."""
    if not isinstance(value, str) or not value:
        raise ValueError("invalid_path")
    root = os.path.normcase(os.path.abspath(ROOT))
    candidate = os.path.abspath(os.path.join(root, value))
    try:
        if os.path.commonpath([root, os.path.normcase(candidate)]) != root:
            raise ValueError("path_outside_workspace")
    except ValueError:
        raise ValueError("path_outside_workspace") from None
    path = Path(candidate)
    relative = path.relative_to(Path(os.path.abspath(ROOT)))
    current = Path(os.path.abspath(ROOT))
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError("linked_path_forbidden")
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("invalid_workspace_file")
    return path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("invalid_timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("naive_timestamp")
    return parsed.astimezone(timezone.utc)


def identifier(value):
    return isinstance(value, str) and str(uuid.UUID(value)) == value.lower()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def bbox(value, width, height):
    if not isinstance(value, list) or len(value) != 4 or not all(finite(v) for v in value):
        raise ValueError("invalid_bbox")
    x, y, w, h = value
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError("bbox_out_of_bounds")


def polygon(points, width, height):
    if not isinstance(points, list) or not 3 <= len(points) <= 256:
        raise ValueError("invalid_polygon")
    if any(not isinstance(p, list) or len(p) != 2 or not all(finite(v) for v in p)
           or not (0 <= p[0] <= width and 0 <= p[1] <= height) for p in points):
        raise ValueError("polygon_out_of_bounds")
    pts = [tuple(p) for p in points]
    if len(set(pts)) != len(pts):
        raise ValueError("duplicate_polygon_point")
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def on(a, b, p):
        return cross(a, b, p) == 0 and min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    for i, a in enumerate(pts):
        b = pts[(i+1) % len(pts)]
        for j in range(i+1, len(pts)):
            if j == i+1 or (i == 0 and j == len(pts)-1):
                continue
            c, d = pts[j], pts[(j+1) % len(pts)]
            ca, cb, cc, cd = cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)
            if (ca*cb < 0 and cc*cd < 0) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b):
                raise ValueError("self_intersecting_polygon")
    if sum(pts[i][0]*pts[(i+1)%len(pts)][1] - pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts))) == 0:
        raise ValueError("zero_area_polygon")


def validate_bundle(bundle, *, now=None, verify_frame_hashes=False):
    """Metadata + permission checksum by default; frame hashes require explicit opt-in.

    Errors contain only row indexes and fixed codes. No image decoding or model calls.
    All AI drafts remain ineligible for training, negatives, and independent evaluation.
    """
    now = now or datetime.now(timezone.utc)
    errors = []
    def error(code, index=None):
        errors.append({"code": code} | ({"sample_index": index} if index is not None else {}))
    if not isinstance(bundle, dict):
        return {"valid": False, "samples": 0, "errors": [{"code": "bundle_must_be_object"}]}
    try:
        if bundle.get("format") != "visual-ai-annotation-bundle-v1" or bundle.get("purpose") != "human_review_only":
            error("invalid_draft_purpose")
        for key in ("training_export_enabled", "evaluation_ground_truth_eligible"):
            if bundle.get(key) is not False:
                error("forbidden_" + key)
        if any(bundle.get(key) not in (None, False) for key in ("human_reviewed", "qc_passed", "AIqc_passed")):
            error("ai_qc_forbidden")
        if not identifier(bundle.get("leakage_group_id")):
            error("invalid_leakage_group_id")
        expiry = timestamp(bundle["expires_at"])
        source_expiry = timestamp(bundle["source_expires_at"])
        if now.tzinfo is None or not now < expiry <= source_expiry <= DEADLINE:
            error("invalid_bundle_expiry")
        permission = scoped_file(bundle["permission_path"])
        expected = bundle["permission_sha256"]
        if not isinstance(expected, str) or not SHA.fullmatch(expected) or digest(permission) != expected:
            error("permission_hash_mismatch")
    except (ValueError, KeyError, TypeError, OSError, AttributeError):
        error("invalid_bundle_permission_or_metadata")
        expiry = None
    samples = bundle.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 10_000:
        error("invalid_samples")
        return {"valid": False, "samples": 0, "errors": errors}
    ids, hashes, annotation_ids, pairs = set(), set(), set(), {}
    for index, sample in enumerate(samples):
        try:
            if not isinstance(sample, dict):
                raise ValueError("sample_must_be_object")
            if not identifier(sample.get("sample_id")) or sample["sample_id"] in ids:
                raise ValueError("duplicate_or_invalid_sample_id")
            ids.add(sample["sample_id"])
            if sample.get("leakage_group_id") != bundle.get("leakage_group_id"):
                raise ValueError("same_trip_group_required")
            if sample.get("split") not in (None, "unassigned"):
                raise ValueError("draft_split_forbidden")
            if sample.get("annotator_id") != "AIassistant" or sample.get("reviewer_id") is not None:
                raise ValueError("invalid_ai_authorship")
            for flag in ("human_reviewed", "detection_eligible", "labels_complete", "negative_eligible"):
                if sample.get(flag) is not False:
                    raise ValueError("draft_false_flags_required")
            if (sample.get("annotation_status") not in ("draft", "held") or sample.get("qc_status") in ("qc_passed", "AIqc_passed")
                    or any(sample.get(key) not in (None, False) for key in ("qc_passed", "AIqc_passed"))):
                raise ValueError("ai_qc_forbidden")
            if any(sample.get(k) not in (None, False) for k in ("training_export", "training_export_enabled", "evaluation_ground_truth_eligible", "independent_evaluation_eligible")):
                raise ValueError("draft_export_or_evaluation_forbidden")
            end, source = timestamp(sample["expires_at"]), timestamp(sample["source_expires_at"])
            if expiry is None or not now < end <= min(source, expiry) or source > source_expiry:
                raise ValueError("invalid_sample_expiry")
            image = sample["image"]
            width, height = image["width"], image["height"]
            if type(width) is not int or type(height) is not int or min(width,height) <= 0 or width*height > MAX_PIXELS:
                raise ValueError("invalid_image_size")
            if image.get("coord_space") != "left_original":
                raise ValueError("original_image_coordinates_required")
            frame_hash = image["sha256"]
            if not isinstance(frame_hash, str) or not SHA.fullmatch(frame_hash) or frame_hash in hashes:
                raise ValueError("duplicate_or_invalid_frame_hash")
            hashes.add(frame_hash)
            frame = scoped_file(image["path"])
            if verify_frame_hashes and digest(frame) != frame_hash:
                raise ValueError("frame_hash_mismatch")
            uncertain = bool(sample.get("uncertain_regions"))
            for key in ("potholes", "hard_negatives", "uncertain_regions", "privacy_boxes", "lane_polygons"):
                entries = sample.get(key, [])
                if not isinstance(entries, list) or len(entries) > 10_000:
                    raise ValueError("invalid_annotation_list")
                for obj in entries:
                    if not isinstance(obj, dict):
                        raise ValueError("invalid_annotation_object")
                    if key != "lane_polygons":
                        aid = obj.get("annotation_id")
                        if not identifier(aid) or aid in annotation_ids:
                            raise ValueError("duplicate_or_invalid_annotation_id")
                        annotation_ids.add(aid)
                        bbox(obj["bbox_xywh_px"], width, height)
                    if key == "privacy_boxes" and (obj.get("class") not in ("face", "license_plate") or obj.get("certainty") not in ("confirmed", "uncertain")):
                        raise ValueError("invalid_privacy_box")
                    if key in ("potholes", "privacy_boxes"):
                        uncertain |= obj.get("certainty") == "uncertain"
                    if key in ("hard_negatives", "uncertain_regions", "privacy_boxes", "lane_polygons") and (not isinstance(obj.get("reason"), str) or not obj["reason"].strip()):
                        raise ValueError("annotation_reason_required")
                    if key == "hard_negatives" and (not isinstance(obj.get("type"), str) or not obj["type"].strip()):
                        raise ValueError("hard_negative_type_required")
                    if key == "uncertain_regions" and obj.get("target") not in ("pothole", "lane", "privacy", "face", "license_plate"):
                        raise ValueError("invalid_uncertain_target")
                    if key == "lane_polygons":
                        if obj.get("kind") not in ("lane_paint", "lane_outline", "intentional_gap", "valid_region"):
                            raise ValueError("invalid_lane_polygon_kind")
                        polygon(obj["points"], width, height)
            if uncertain and sample["annotation_status"] != "held":
                raise ValueError("uncertain_requires_held")
            if "lane_masks" in sample:
                masks = sample["lane_masks"]
                if not isinstance(masks, dict) or set(masks) != {"outline", "gap", "paint", "valid"}:
                    raise ValueError("incomplete_lane_masks")
                for mask in masks.values():
                    if not isinstance(mask, dict) or mask.get("size") != [height,width]:
                        raise ValueError("mask_original_resolution_required")
                    counts = mask.get("counts")
                    if not isinstance(counts,list) or not 1 <= len(counts) <= min(2*width*height+1, 1_000_000) or any(type(c) is not int or c < 0 for c in counts) or sum(counts) != width*height:
                        raise ValueError("invalid_rle_counts")
                # dataset.decode_rle uses COCO column-major (F), alternating 0/1 runs.
                from agent.dataset import decode_rle
                outline, valid = decode_rle(masks["outline"]), decode_rle(masks["valid"])
                if (valid.astype(bool) & ~outline.astype(bool)).any():
                    raise ValueError("valid_mask_outside_manual_outline")
            if validate_annotation(sample):
                raise ValueError("dataset_annotation_invalid")
            if sample.get("stereo_pair_id") is not None:
                pair = sample["stereo_pair_id"]
                view = sample.get("view")
                if not identifier(pair) or view not in ("left", "right"):
                    raise ValueError("invalid_stereo_identity")
                meta = (sample.get("trip_id"), width, height, sample["leakage_group_id"])
                previous = pairs.setdefault(pair, {"meta": meta, "views": set()})
                if previous["meta"] != meta or view in previous["views"]:
                    raise ValueError("inconsistent_stereo_pair")
                previous["views"].add(view)
        except (ValueError, KeyError, TypeError, OSError, AttributeError, OverflowError) as exc:
            code = str(exc) if type(exc) is ValueError and re.fullmatch(r"[a-z_]+", str(exc)) else "malformed_sample"
            error(code, index)
    return {"valid": not errors, "samples": len(samples), "errors": errors,
            "frame_hashes_verified": bool(verify_frame_hashes), "human_reviewed": False,
            "training_export_enabled": False, "evaluation_ground_truth_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--verify-frame-hashes", action="store_true", help="Hash approved extracted files; never decode pixels")
    args = parser.parse_args()
    try:
        path = scoped_file(args.bundle)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("bundle_too_large")
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES+1)
        if len(raw) > MAX_BYTES:
            raise ValueError("bundle_too_large")
        result = validate_bundle(json.loads(raw), verify_frame_hashes=args.verify_frame_hashes)
    except (ValueError, OSError, TypeError):
        result = {"valid": False, "errors": [{"code": "invalid_bundle_file"}]}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
