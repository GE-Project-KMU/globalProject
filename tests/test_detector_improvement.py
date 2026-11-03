"""Synthetic conditional-label and sealed holdout checks, not model accuracy."""
import uuid
import pytest
from agent.paths import ROOT,sha256,atomic_json
from scripts.compare_detector_models import known_classes,score


def test_unknown_class_excluded_without_removing_known_class_missed_ground_truth():
    classes=["face","license_plate"]
    known=known_classes({"evaluation_class":"face"},classes)
    truth=[{"sample_id":"synthetic","class":"face","bbox_xywh_px":[10,10,20,20]}]
    predictions=[{"sample_id":"synthetic","class":"license_plate","bbox_xywh_px":[10,10,20,20],"confidence":.9}]
    result=score(truth,[p for p in predictions if p["class"] in known],classes,.01)
    assert result["classes"]["face"]["fn"]==1 and result["classes"]["face"]["recall"]==0
    assert result["classes"]["license_plate"]["fp"]==0 and result["classes"]["license_plate"]["ap50"] is None
    with pytest.raises(ValueError):known_classes({"known_label_coverage":["unknown"]},classes)


def test_reference_operating_point_keeps_low_confidence_false_negatives():
    truth=[{"sample_id":"synthetic","class":"pothole","bbox_xywh_px":[0,0,10,10]}]
    predictions=[dict(truth[0],confidence=.2)]
    assert score(truth,predictions,["pothole"],.01)["map50"]==1
    assert score(truth,predictions,["pothole"],.35)["classes"]["pothole"]["fn"]==1


def test_detector_comparison_refuses_development_image_before_inference(monkeypatch):
    from scripts import compare_detector_models as comparison
    from test_research import research_sample
    row,_=research_sample();row["split"]="test"
    folder=ROOT/".cache/detector-improvement-tests"/uuid.uuid4().hex;folder.mkdir(parents=True)
    bench=folder/"bench.json";atomic_json(bench,[row])
    source=folder/"source.json";atomic_json(source,[row|{"split":"train"}])
    modelpath=folder/"model.json";atomic_json(modelpath,{"synthetic":"leakage-sentinel"})
    class Model:
        manifest={"research_only":True,"production_approved":False,"task":"redaction","classes":["face","license_plate"],
                  "source_release_path":str(source),"dataset_release":"manifest-sha256:"+sha256(source)}
        def predict(self,image):raise AssertionError("must never infer leaked image")
    monkeypatch.setattr(comparison,"OnnxModel",lambda path:Model())
    plan=folder/"plan.json";atomic_json(plan,{"benchmark_manifest":str(bench),"benchmark_sha256":sha256(bench),"models":{"baseline":str(modelpath),"candidate":str(modelpath)}})
    with pytest.raises(ValueError,match="holdout_development_leakage"):comparison.compare(plan,folder/"result.json")
    assert not (folder/"result.json").exists()
