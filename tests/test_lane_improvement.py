"""Synthetic resolution/calibration contracts, not road accuracy evidence."""
import json,math,random,uuid
import cv2
import numpy as np
import pytest
from agent.paths import ROOT,atomic_json,sha256
from agent.lane_metrics import PAINT_THRESHOLDS,threshold_confusions


def test_histogram_sweep_exactly_matches_application_threshold_then_nearest_resize():
    rng=np.random.default_rng(20261104)
    truth=rng.choice([0,1,255],size=(47,71)).astype(np.uint8)
    delta=rng.normal(size=(11,17)).astype(np.float32)
    delta[0,:len(PAINT_THRESHOLDS)]=[math.log(t/(1-t)) for t in PAINT_THRESHOLDS]
    actual=threshold_confusions(truth,delta)
    for t in PAINT_THRESHOLDS:
        mask=cv2.resize((delta>math.log(t/(1-t))).astype(np.uint8),(71,47),interpolation=cv2.INTER_NEAREST)
        assert actual[t]=={"tp":int(((truth==1)&(mask==1)).sum()),"fp":int(((truth==0)&(mask==1)).sum()),
                           "fn":int(((truth==1)&(mask==0)).sum()),"tn":int(((truth==0)&(mask==0)).sum()),"ignored":int((truth==255).sum())}


def test_paint_sweep_rejects_invalid_thresholds_and_nonfinite_logits():
    truth=np.zeros((4,4),np.uint8);delta=np.zeros((4,4),np.float32)
    with pytest.raises(ValueError):threshold_confusions(truth,delta,[.5,.1])
    with pytest.raises(ValueError):threshold_confusions(truth,np.full((4,4),np.nan,np.float32))


def test_native_patch_keeps_paint_color_mask_alignment_and_ignores_padding():
    from scripts.improve_lane import training_patch
    folder=ROOT/".cache/lane-improvement-tests"/uuid.uuid4().hex;folder.mkdir(parents=True)
    image=np.zeros((240,320,3),np.uint8);image[80:160,120:200]=[0,0,255]
    truth=np.zeros((240,320),np.uint8);truth[80:160,120:200]=1
    picture=folder/"synthetic.png";mask=folder/"mask.png";cv2.imwrite(str(picture),image);cv2.imwrite(str(mask),truth)
    before=(sha256(picture),sha256(mask));random.seed(4)
    pixels,labels=training_patch({"path":str(picture),"mask_path":str(mask)},size=640,view_size=512)
    raw=pixels.transpose(1,2,0)*np.asarray([.229,.224,.225])+np.asarray([.485,.456,.406])
    assert set(np.unique(labels))<={0,1,255} and (labels==1).any() and (labels==255).any()
    assert raw[labels==1,0].mean()>raw[labels==1,1].mean()+.5
    assert before==(sha256(picture),sha256(mask))


@pytest.mark.parametrize("development_only",[False,True])
def test_calibration_uses_manifest_resolution_and_preserves_frozen_artifact(monkeypatch,development_only):
    from scripts import calibrate_lane_model as calibration
    folder=ROOT/".cache/lane-improvement-tests"/uuid.uuid4().hex;folder.mkdir(parents=True)
    picture=folder/"synthetic.png";mask=folder/"mask.png"
    cv2.imwrite(str(picture),np.zeros((16,24,3),np.uint8));cv2.imwrite(str(mask),np.zeros((16,24),np.uint8))
    release=folder/"release.json";atomic_json(release,[{"split":"validation","source_id":"synthetic","path":str(picture),"mask_path":str(mask)}])
    weights=folder/"weights.onnx";weights.write_bytes(b"synthetic no graph; explicit fake inference")
    model_path=folder/"model.json";artifact={"research_only":True,"dataset_release":"manifest-sha256:"+sha256(release),"weights":str(weights),"sha256":sha256(weights),"input_size":[32,32]}
    if development_only:
        corpus=folder/"corpus.json";atomic_json(corpus,{"synthetic_full_release_hash_only":True})
        artifact.update(dataset_release="manifest-sha256:"+sha256(corpus),source_release_path=str(corpus),training_development_release_sha256=sha256(release))
    atomic_json(model_path,artifact)
    monkeypatch.setattr(calibration,"validate_research_release",lambda rows:{"valid":True})
    class Net:
        def setInput(self,blob):assert blob.shape==(1,3,32,32)
        def forward(self):return np.zeros((1,2,32,32),np.float32)
    class Model:
        net=Net();backend="synthetic-contract"
    monkeypatch.setattr(calibration,"OnnxModel",lambda path:Model())
    calibration.calibrate(release,model_path,folder/"calibration.json")
    assert json.loads(model_path.read_text())["threshold_calibration"]["test_images_used"]==0
    after=sha256(model_path)
    with pytest.raises(ValueError,match="already_frozen"):calibration.calibrate(release,model_path,folder/"second.json")
    assert sha256(model_path)==after


def test_comparison_rejects_any_training_image_in_holdout_before_inference(monkeypatch):
    from scripts import compare_lane_models as comparison
    from test_research import research_sample
    row,_=research_sample();row["split"]="test"
    folder=ROOT/".cache/lane-improvement-tests"/uuid.uuid4().hex;folder.mkdir(parents=True)
    bench=folder/"benchmark.json";atomic_json(bench,[row])
    source=folder/"training.json";atomic_json(source,[row|{"split":"train"}])
    model_path=folder/"model.json";atomic_json(model_path,{"synthetic":"leakage-guard-only"})
    class Model:
        manifest={"task":"lane","research_only":True,"production_approved":False,"threshold_calibration":{"selected_on":"validation"},
                  "source_release_path":str(source),"dataset_release":"manifest-sha256:"+sha256(source)}
        def predict(self,image):raise AssertionError("must not infer leaked data")
    monkeypatch.setattr(comparison,"OnnxModel",lambda path:Model())
    plan=folder/"plan.json";atomic_json(plan,{"benchmark_manifest":str(bench),"benchmark_sha256":sha256(bench),"models":{"baseline":str(model_path),"candidate":str(model_path)}})
    with pytest.raises(ValueError,match="holdout_training_or_validation_leakage"):
        comparison.compare(plan,folder/"result.json")
    assert not (folder/"result.json").exists()


def test_frozen_comparison_cannot_change_model_after_interrupted_evaluation():
    from scripts.compare_lane_models import preserve_freeze
    folder=ROOT/".cache/lane-improvement-tests"/uuid.uuid4().hex;folder.mkdir(parents=True)
    path=folder/"receipt.json";original={"candidate_sha256":"a"*64,"threshold":.7}
    preserve_freeze(path,original);before=sha256(path);preserve_freeze(path,original)
    with pytest.raises(ValueError,match="no_reselection"):preserve_freeze(path,dict(original,threshold=.8))
    assert sha256(path)==before


def test_comparison_lock_rejects_parallel_writer_and_cleans_up_exception(monkeypatch):
    from scripts import compare_lane_models as comparison
    folder=ROOT/".cache/lane-improvement-tests"/uuid.uuid4().hex;folder.mkdir(parents=True)
    output=folder/"result.json";lock=folder/"result.json.lock";lock.write_text("synthetic existing owner")
    with pytest.raises(ValueError,match="comparison_busy"):comparison.compare("unused",output)
    lock.unlink()
    def failed(*args):
        assert lock.exists()
        raise ValueError("synthetic execution failure")
    monkeypatch.setattr(comparison,"_compare",failed)
    with pytest.raises(ValueError,match="synthetic execution failure"):comparison.compare("unused",output)
    assert not lock.exists()


def test_final_comparison_requires_current_real_image_torch_and_postprocess_proof():
    from scripts.compare_lane_models import verified_lane_artifact
    folder=ROOT/".cache/lane-improvement-tests"/uuid.uuid4().hex;folder.mkdir(parents=True)
    model=folder/"model.json";atomic_json(model,{"synthetic":"proof-guard","sha256":"c"*64})
    proof={"manifest_sha256":sha256(model),"numerical_parity_passed":True,"torch_checkpoint_compared":True,
           "synthetic_input":False,"lane_adapter_postprocess_matches_reference":True,"weight_sha256":"c"*64,
           "artifact_contract_verified":True,"mask_agreement":{name:{"pixel_agreement":1.0} for name in ("app","ort","torch")}}
    path=folder/"verification-final.json";atomic_json(path,proof);verified_lane_artifact(model)
    for patch in ({"manifest_sha256":"0"*64},{"numerical_parity_passed":False},{"torch_checkpoint_compared":False},
                  {"synthetic_input":True},{"lane_adapter_postprocess_matches_reference":False}):
        atomic_json(path,proof|patch)
        with pytest.raises(ValueError,match="verification_required"):verified_lane_artifact(model)


def test_diagnostic_accuracy_never_turns_a_failed_checkpoint_check_into_a_pass():
    from scripts.compare_lane_models import verified_lane_artifact
    folder=ROOT/".cache/lane-improvement-tests"/uuid.uuid4().hex;folder.mkdir(parents=True)
    model=folder/"model.json";path=folder/"verification-final.json"
    metadata={"research_only":True,"production_approved":False,"project_acceptance_eligible":False,"sha256":"c"*64,"diagnostic_accuracy_only":True,
              "checkpoint_raw_tensor_parity_passed":False,"trained_checkpoint_sha256":"a"*64,
              "deployment_blocked_reasons":["checkpoint_raw_tensor_parity_unresolved"]}
    proof={"numerical_parity_passed":False,"artifact_contract_verified":False,"torch_checkpoint_compared":True,
           "synthetic_input":False,"lane_adapter_postprocess_matches_reference":True,"checkpoint_sha256":"a"*64,"weight_sha256":"c"*64,
           "app_vs_ort":{"passed":True},"mask_agreement":{name:{"pixel_agreement":1.0,"paint_pixels":10} for name in ("app","ort","torch")}}
    def write(m,p):
        atomic_json(model,m);atomic_json(path,p|{"manifest_sha256":sha256(model)})
    write(metadata,proof)
    with pytest.raises(ValueError,match="verification_required"):verified_lane_artifact(model)
    result=verified_lane_artifact(model,diagnostic=True)
    assert result["checkpoint_raw_tensor_parity_passed"] is False and result["diagnostic_accuracy_only"] is True
    assert json.loads(path.read_text())["artifact_contract_verified"] is False
    for mutation in ({"app_vs_ort":{"passed":False}},{"artifact_contract_verified":True},
                     {"checkpoint_sha256":"b"*64},{"weight_sha256":"b"*64},{"synthetic_input":True},
                     {"mask_agreement":{**proof["mask_agreement"],"torch":{"pixel_agreement":.99999,"paint_pixels":10}}}):
        write(metadata,proof|mutation)
        with pytest.raises(ValueError):verified_lane_artifact(model,diagnostic=True)
    for mutation in ({"production_approved":True},{"research_only":False},{"project_acceptance_eligible":True},
                     {"checkpoint_raw_tensor_parity_passed":True},{"diagnostic_accuracy_only":False},{"deployment_blocked_reasons":[]}):
        write(metadata|mutation,proof)
        with pytest.raises(ValueError):verified_lane_artifact(model,diagnostic=True)
