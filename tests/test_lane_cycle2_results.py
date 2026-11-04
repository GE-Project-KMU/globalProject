import copy
import pytest
from scripts.finalize_lane_cycle2_results import validated_metrics


def valid():
    return {"confusion": {"tp": 6, "fp": 2, "fn": 3, "tn": 100, "ignored": 7},
            "paint_iou": 6/11, "paint_precision": 6/8,
            "paint_recall": 6/9, "paint_dice": 12/17}


def test_pooled_pixel_metrics_recomputed_from_confusion():
    result = validated_metrics(valid())
    assert result["paint_iou"] == 6/11
    assert result["paint_dice"] == 12/17


@pytest.mark.parametrize("field,value", [("paint_iou", .9), ("paint_precision", float("nan")),
                                         ("paint_recall", float("inf")), ("paint_dice", True)])
def test_inconsistent_or_nonfinite_metric_rejected(field, value):
    item = valid()
    item[field] = value
    with pytest.raises(ValueError, match="aggregate_metric_confusion_mismatch"):
        validated_metrics(item)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_confusion_counts_are_nonnegative_integers(value):
    item = valid()
    item["confusion"]["fp"] = value
    with pytest.raises(ValueError, match="nonnegative_integer_confusion_required"):
        validated_metrics(item)


def test_unsupported_zero_paint_population_not_fabricated_as_perfect():
    item = copy.deepcopy(valid())
    item["confusion"].update(tp=0, fp=0, fn=0)
    with pytest.raises(ValueError, match="supported_paint_metric_denominator_required"):
        validated_metrics(item)
