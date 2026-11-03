"""Measurement from explicit outline/gap/paint/valid masks, not guessed geometry."""
import numpy as np
from .paths import finite


def measure_patch(outline, gap, paint, valid, *, outline_supported=True):
    arrays = [np.asarray(x) for x in (outline, gap, paint, valid)]
    if any(x.ndim != 2 or x.shape != arrays[0].shape for x in arrays):
        raise ValueError("mask_shape_mismatch")
    if any(not np.isin(x, [0, 1, 255]).all() for x in arrays):
        raise ValueError("mask_must_be_binary")
    # 255 in O/G/P is a foreground value for ordinary binary PNG. Valid=255 is
    # deliberately NOT accepted: valid is explicit 0/1, avoiding ignore confusion.
    if not np.isin(arrays[3], [0, 1]).all():
        raise ValueError("valid_mask_must_be_zero_or_one")
    o, g, p, v = [x > 0 for x in arrays]
    expected = o & ~g
    total = int(expected.sum())
    effective = expected & v
    valid_area = int(effective.sum())
    painted = int((p & effective).sum())
    result = {"outline_supported": bool(outline_supported), "expected_area": total,
              "effective_area": valid_area, "paint_area": painted,
              "valid_area_ratio": valid_area / total if total and outline_supported else None,
              "retention_pct": None, "grade": "unassessable", "unassessable_reason": None}
    if not outline_supported or total == 0:
        result["unassessable_reason"] = "missing_outline_basis"
    elif valid_area / total < .7:
        result["unassessable_reason"] = "insufficient_valid_area"
    else:
        ratio = 100 * painted / valid_area
        result.update(retention_pct=ratio, grade=grade(ratio))
    return result


def grade(ratio):
    ratio = finite(ratio, "retention")
    if not 0 <= ratio <= 100:
        raise ValueError("retention_out_of_range")
    return "visible_good" if ratio >= 80 else "monitor" if ratio >= 60 else "worn_candidate"


def aggregate_segment(patches, *, segment_support_ratio, estimated_length_m):
    unique = {}
    for patch in patches:
        key = patch.get("patch_id")
        if not key:
            raise ValueError("patch_id_required")
        if key in unique and unique[key] != patch:
            raise ValueError("conflicting_duplicate_patch")
        unique[key] = patch
    base = {"retention_pct": None, "grade": "unassessable", "segment_support_ratio": segment_support_ratio,
            "estimated_length_m": estimated_length_m, "patch_ids": list(unique), "unassessable_reason": None}
    if segment_support_ratio is None or estimated_length_m is None:
        return base | {"unassessable_reason": "missing_distance_basis"}
    support = finite(segment_support_ratio, "segment_support")
    length = finite(estimated_length_m, "segment_length")
    if not 0 <= support <= 1 or length <= 0:
        raise ValueError("invalid_segment_geometry")
    if support < .8:
        return base | {"unassessable_reason": "insufficient_segment_support"}
    accepted = [p for p in unique.values() if p.get("grade") != "unassessable"]
    area = sum(p["effective_area"] for p in accepted)
    if not area:
        return base | {"unassessable_reason": "no_valid_patch"}
    ratio = 100 * sum(p["paint_area"] for p in accepted) / area
    return base | {"retention_pct": ratio, "grade": grade(ratio)}
