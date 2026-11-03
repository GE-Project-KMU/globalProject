"""Artifact contract/security tests; no fixture is claimed as a trained model."""
import copy,json,tempfile
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.verify_trained_models import ROOT,digest,validate_metadata,verify,check_output,compare

@pytest.fixture
def artifact():
    base=ROOT/".cache/model-verification-tests";base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        folder=Path(directory);release=folder/"release.json";provenance=folder/"provenance.json";weights=folder/"weights.onnx";manifest=folder/"model.json"
        release.write_text(json.dumps([{"sample_id":split,"split":split} for split in ("train","validation","test")]),encoding="utf-8")
        provenance.write_text(json.dumps({"release_sha256":digest(release),"test_split_exported":False,"production_approved":False,"assets":[{"sample_id":"train","identity_hashes":["a"*64],"source_sha256":"b"*64,"expires_at":(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()}]}),encoding="utf-8")
        weights.write_bytes(b"SYNTHETIC METADATA FIXTURE - NOT VALID ONNX")
        model={"format":"onnx","task":"lane","classes":["background","paint"],"weights":weights.relative_to(ROOT).as_posix(),"sha256":digest(weights),"input_size":[512,512],"target_trained":True,"research_only":True,"production_approved":False,"project_acceptance_eligible":False,"test_split_used":False,"independent_route_verified":False,"source_licenses":["synthetic-test-only"],"dataset_release":"manifest-sha256:"+digest(release),"source_release_path":release.relative_to(ROOT).as_posix(),"training_provenance":provenance.relative_to(ROOT).as_posix()}
        manifest.write_text(json.dumps(model),encoding="utf-8")
        yield SimpleNamespace(folder=folder,release=release,provenance=provenance,weights=weights,manifest=manifest,model=model)

def test_metadata_only_does_not_open_unfinished_weights(artifact):
    artifact.weights.unlink()
    result=verify(SimpleNamespace(manifest=artifact.manifest,metadata_only=True))
    assert result["metadata_valid"] and not result["weight_hash_verified"]
    assert result["production_approved"] is False and result["accuracy_evaluation_performed"] is False
    assert result["split_counts"]=={"train":1,"validation":1,"test":1}

@pytest.mark.parametrize("change",[{"research_only":False},{"production_approved":True},{"source_licenses":[]},{"weights":"../outside.onnx"},{"weights":"C:/outside/model.onnx"},{"sha256":"invalid"},{"classes":["paint"]},{"project_acceptance_eligible":True}])
def test_invalid_or_overclaimed_manifest_rejected(artifact,change):
    artifact.manifest.write_text(json.dumps({**artifact.model,**change}),encoding="utf-8")
    with pytest.raises(ValueError):validate_metadata(artifact.manifest)

def test_holdout_leakage_expiry_and_source_hash_tampering_rejected(artifact):
    original=json.loads(artifact.provenance.read_text())
    for mutation in ("holdout","expiry","hash"):
        p=copy.deepcopy(original)
        if mutation=="holdout":p["assets"][0]["sample_id"]="test"
        elif mutation=="expiry":p["assets"][0]["expires_at"]="2000-01-01T00:00:00+00:00"
        else:p["release_sha256"]="0"*64
        artifact.provenance.write_text(json.dumps(p),encoding="utf-8")
        with pytest.raises(ValueError):validate_metadata(artifact.manifest)

def test_weight_tampering_rejected_before_model_loading(artifact):
    artifact.weights.write_bytes(b"changed")
    with pytest.raises(ValueError,match="weight_hash_mismatch"):verify(SimpleNamespace(manifest=artifact.manifest,metadata_only=False))

def test_finite_shape_and_decoded_yolox_contract():
    check_output("lane",["background","paint"],np.zeros((1,2,4,4),np.float32))
    valid=np.array([[[10.,10.,5.,5.,.8,.9]]],np.float32)
    check_output("pothole",["pothole"],valid)
    for value in (np.full((1,2,4,4),np.nan),np.zeros((1,3,4,4)),np.zeros((1,2,0,4))):
        with pytest.raises(ValueError):check_output("lane",["background","paint"],value)
    invalid=valid.copy();invalid[0,0,4]=1.1
    with pytest.raises(ValueError):check_output("pothole",["pothole"],invalid)
    assert compare(valid,valid.copy(),.001,.0001)["passed"]
    assert not compare(valid,valid+1,.001,.0001)["passed"]

def package_fixture(artifact):
    from scripts.model_artifacts import build
    artifact.model["artifact_redistribution_allowed"]=True
    artifact.manifest.write_text(json.dumps(artifact.model),encoding="utf-8")
    original=json.loads(artifact.provenance.read_text());original["assets"][0].update(source_id="PRIVATE_PERSON",source_path="PRIVATE/raw_face.jpg",files=[{"path":"PRIVATE/raw_face.jpg"}])
    artifact.provenance.write_text(json.dumps(original),encoding="utf-8")
    report=artifact.folder/"verification.json"
    report.write_text(json.dumps({"manifest_sha256":digest(artifact.manifest),"weight_sha256":artifact.model["sha256"],"metadata_valid":True,"weight_hash_verified":True,"finite_output":True,"preprocess_contract_verified":True,"numerical_parity_passed":True,"app_vs_ort":{"passed":True,"max_abs_error":0.,"mean_abs_error":0.,"rtol":.001,"atol":.0001}}),encoding="utf-8")
    output=artifact.folder/"research.zip";receipt=build(artifact.manifest,output,report)
    return output,receipt,report

def test_distribution_requires_explicit_rights_and_full_matching_verification(artifact):
    from scripts.model_artifacts import build
    with pytest.raises(ValueError,match="redistribution"):build(artifact.manifest,artifact.folder/"blocked.zip",artifact.folder/"missing.json")
    output,receipt,report=package_fixture(artifact)
    value=json.loads(report.read_text());value["weight_sha256"]="0"*64;report.write_text(json.dumps(value),encoding="utf-8")
    with pytest.raises(ValueError,match="verification_artifact_hash"):build(artifact.manifest,artifact.folder/"blocked.zip",report)

def test_portable_archive_sanitizes_provenance_and_installs_without_original_sources(artifact):
    from scripts.model_artifacts import inspect_archive,install
    import shutil,uuid
    output,receipt,report=package_fixture(artifact);bundle,payload=inspect_archive(output,receipt["sha256"])
    metadata=b"".join(value for name,value in payload.items() if name.endswith(".json"))
    assert b"PRIVATE_PERSON" not in metadata and b"raw_face.jpg" not in metadata
    artifact.release.unlink();artifact.provenance.unlink();artifact.weights.unlink()
    identifier="test-"+uuid.uuid4().hex;target=ROOT/"data/models/research"/identifier
    try:
        installed=install(output,receipt["sha256"],identifier)
        info=validate_metadata(ROOT/installed["manifest"])
        assert info["split_counts"]=={"train":1,"validation":1,"test":1}
        assert digest(info["weight_path"])==artifact.model["sha256"]
        assert installed["origin_manifest_sha256"]==digest(artifact.manifest)
        assert info["provenance"]["assets"][0]["identity_hashes"]==["a"*64]
        with pytest.raises(ValueError,match="new_project_model_destination"):install(output,receipt["sha256"],identifier)
    finally:
        if target.resolve().is_relative_to(ROOT) and target.exists():shutil.rmtree(target)

def test_archive_hash_traversal_extra_files_and_identifier_rejected(artifact):
    from scripts.model_artifacts import inspect_archive,install
    import zipfile
    output,receipt,report=package_fixture(artifact)
    with pytest.raises(ValueError,match="archive_hash"):inspect_archive(output,"0"*64)
    with pytest.raises(ValueError,match="invalid_artifact_id"):install(output,receipt["sha256"],"../outside")
    with zipfile.ZipFile(output,"a") as archive:archive.writestr("../../unexpected.env","SYNTHETIC_SECRET")
    with pytest.raises(ValueError,match="unexpected"):inspect_archive(output,digest(output))

def test_archive_member_tampering_and_symlink_rejected(artifact):
    from scripts.model_artifacts import inspect_archive
    import zipfile
    output,receipt,report=package_fixture(artifact)
    with zipfile.ZipFile(output) as archive:items={name:archive.read(name) for name in archive.namelist()}
    tampered=artifact.folder/"tampered.zip"
    with zipfile.ZipFile(tampered,"w") as archive:
        for name,value in items.items():archive.writestr(name,b"tampered" if name=="weights.onnx" else value)
    with pytest.raises(ValueError,match="bundle_file_hash"):inspect_archive(tampered,digest(tampered))
    linked=artifact.folder/"linked.zip"
    with zipfile.ZipFile(linked,"w") as archive:
        for name,value in items.items():
            entry=zipfile.ZipInfo(name)
            if name=="weights.onnx":entry.create_system=3;entry.external_attr=0o120777<<16
            archive.writestr(entry,value)
    with pytest.raises(ValueError,match="links_not_allowed"):inspect_archive(linked,digest(linked))

def test_validation_threshold_metadata_is_preserved_and_test_tuning_rejected(artifact):
    from scripts.model_artifacts import inspect_archive
    artifact.model.update(paint_threshold=.9,threshold_calibration={"selected_on":"validation","report_sha256":"c"*64,"validation_samples":3,"test_images_used":0})
    artifact.manifest.write_text(json.dumps(artifact.model),encoding="utf-8")
    output,receipt,_=package_fixture(artifact)
    _,payload=inspect_archive(output,receipt["sha256"])
    assert json.loads(payload["model.json"])["paint_threshold"]==.9
    artifact.model["threshold_calibration"]["selected_on"]="test"
    artifact.manifest.write_text(json.dumps(artifact.model),encoding="utf-8")
    with pytest.raises(ValueError,match="validation_only"):validate_metadata(artifact.manifest)

def test_publisher_pinned_download_checks_hash_and_rights_before_install(artifact,monkeypatch):
    from scripts.model_artifacts import download
    import io
    output,receipt,_=package_fixture(artifact)
    catalog=artifact.folder/"catalog.json"
    entry={"id":"synthetic-test","url":"https://example.invalid/model.zip","sha256":receipt["sha256"],"research_only":True,"artifact_redistribution_allowed":True}
    catalog.write_text(json.dumps({"schema_version":1,"artifacts":[entry]}),encoding="utf-8")
    class Response(io.BytesIO):url=entry["url"]
    monkeypatch.setattr("urllib.request.urlopen",lambda *a,**k:Response(output.read_bytes()))
    downloaded=artifact.folder/"downloaded.zip"
    assert download(catalog,entry["id"],downloaded)["sha256"]==digest(downloaded)
    monkeypatch.setattr("urllib.request.urlopen",lambda *a,**k:Response(b"tampered"))
    with pytest.raises(ValueError,match="archive_hash"):download(catalog,entry["id"],artifact.folder/"bad.zip")
    assert not (artifact.folder/"bad.zip").exists() and not list(artifact.folder.glob("*.part"))
    entry["artifact_redistribution_allowed"]=False
    catalog.write_text(json.dumps({"schema_version":1,"artifacts":[entry]}),encoding="utf-8")
    with pytest.raises(ValueError,match="redistribution"):download(catalog,entry["id"],artifact.folder/"rights.zip")

def test_actual_ort_default_executes_known_conv_and_preserves_lane_contract(artifact):
    """Tiny known Conv tests execution semantics, not road-model accuracy."""
    import base64,onnxruntime
    from agent.models import OnnxModel,ModelNotReady,load_network,runtime_metadata,readiness
    wire="CAg6zgEKMwoGaW1hZ2VzCgF3CgFiEgZsb2dpdHMiBENvbnYqFQoMa2VybmVsX3NoYXBlQAFAAaABBxIXc3ludGhldGljLWNvbnRyYWN0LW9ubHkqJwgCCAMIAQgBEAFCAXdKGAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAACoRCAIQAUIBYkoIAAAAAAAAAABaIAoGaW1hZ2VzEhYKFAgBEhAKAggBCgIIAwoCCBAKAggQYiAKBmxvZ2l0cxIWChQIARIQCgIIAQoCCAIKAggQCgIIEEIECgAQEQ=="
    artifact.weights.write_bytes(base64.b64decode(wire));artifact.model.update(sha256=digest(artifact.weights),input_size=[16,16])
    artifact.manifest.write_text(json.dumps(artifact.model),encoding="utf-8")
    model=OnnxModel(artifact.manifest)
    assert model.backend=="onnxruntime-cpu" and model.backend_version==onnxruntime.__version__
    assert model.net.session.get_providers()==["CPUExecutionProvider"]
    runtime=runtime_metadata({"lane":model})
    assert runtime["engine"]==model.backend and runtime["version"]==onnxruntime.__version__ and runtime["device"]=="cpu"
    assert runtime_metadata({"custom":SimpleNamespace()})["engine"]=="custom-unverified"
    assert model.predict(np.full((16,16,3),255,np.uint8)).all()
    assert not model.predict(np.zeros((16,16,3),np.uint8)).any()
    with pytest.raises(ModelNotReady,match="invalid_model_input"):model.net.setInput(np.full((1,3,16,16),np.nan,np.float32))
    with pytest.raises(ModelNotReady,match="model_input_required"):model.net.forward()
    with pytest.raises(ValueError,match="unsupported_inference_backend"):load_network(artifact.weights,"opencv-auto")
    artifact.manifest.write_text("[]",encoding="utf-8")
    assert readiness(artifact.manifest)["reason"]=="model_manifest_object_required"

def test_training_python_default_matches_project_platform_paths():
    from scripts.verify_trained_models import default_training_python
    assert default_training_python("nt")==ROOT/".venv-training/python.exe"
    assert default_training_python("posix")==ROOT/".venv-training/bin/python"

def test_detector_reference_tensors_use_real_classwise_nms_and_report_deltas():
    from scripts.verify_trained_models import postprocess_detection,compare_detections
    manifest={"task":"redaction","classes":["face","license_plate"],"input_size":[32,32],"output_layout":"yolox_decoded_xywh","confidence":.35,"nms_iou":.5}
    # Two overlapping face anchors collapse, while the plate is retained by
    # classwise NMS. These are synthetic tensors, never accuracy evidence.
    raw=np.array([[[10,10,8,8,.9,.9,.1],[10.1,10.1,8,8,.8,.9,.1],[24,24,6,4,.9,.1,.95],[5,5,2,2,.1,.9,.1]]],np.float32)
    image=np.zeros((32,32,3),np.uint8)
    app=postprocess_detection(manifest,raw,image)
    assert len(app)==2 and {r["class"] for r in app}=={"face","license_plate"}
    nearby=raw.copy();nearby[0,0,0]+=.01;nearby[0,0,4]+=.000001
    result=compare_detections(app,postprocess_detection(manifest,nearby,image))
    assert result["passed"] and result["matched_count"]==2 and 0<result["max_abs_bbox_error_px"]<.05
    assert result["application_class_counts"]=={"face":1,"license_plate":1}
    missing=raw.copy();missing[0,2,4]=0
    assert not compare_detections(app,postprocess_detection(manifest,missing,image))["passed"]
    wrong_class=raw.copy();wrong_class[0,2,5:]=[.95,.1]
    assert not compare_detections(app,postprocess_detection(manifest,wrong_class,image))["passed"]
    bad_box=raw.copy();bad_box[0,2,0]+=1
    assert not compare_detections(app,postprocess_detection(manifest,bad_box,image))["passed"]
    bad_score=raw.copy();bad_score[0,2,4]-=.01
    assert not compare_detections(app,postprocess_detection(manifest,bad_score,image))["passed"]
    empty=compare_detections([],[])
    assert empty["passed"] and empty["no_positive_predictions"] and empty["min_matched_iou"] is None

def test_lane_reference_factory_keeps_legacy_and_explicit_dilation_contract(monkeypatch):
    from scripts.verify_trained_models import build_torch_reference,validate_architecture
    import agent.lane_network as factory
    calls=[];sentinel=object()
    monkeypatch.setattr(factory,'make_lane_network',lambda **kwargs:calls.append(kwargs) or sentinel)
    assert build_torch_reference({'task':'lane'},ROOT/'.cache/vendor/YOLOX') is sentinel
    assert calls[-1]=={'pretrained':False,'aspp_dilations':[12,24,36]}
    explicit={'task':'lane','architecture':'deeplabv3_mobilenet_v3_large','aspp_dilations':[6,12,18]}
    assert build_torch_reference(explicit,ROOT/'.cache/vendor/YOLOX') is sentinel
    assert calls[-1]['aspp_dilations']==[6,12,18]
    for values in (None,[6,12],[True,12,18],[0,12,18],[6,12,65]):
        with pytest.raises(ValueError,match='aspp_dilations'):validate_architecture(explicit|{'aspp_dilations':values})
    for architecture in (None,'resnet50','unknown'):
        with pytest.raises(ValueError,match='unsupported_lane_architecture'):validate_architecture(explicit|{'architecture':architecture})

def test_specialist_reference_factory_and_exact_output_contract(monkeypatch):
    from scripts.verify_trained_models import build_torch_reference,validate_architecture,check_model_output
    import agent.redaction_network as factory
    info={'task':'redaction','classes':['face','license_plate'],'architecture':'two YOLOX-Nano specialists','input_size':[960,960],'face_input_size':960,'plate_input_size':640}
    calls=[];sentinel=object()
    monkeypatch.setattr(factory,'build_redaction_network',lambda **kwargs:calls.append(kwargs) or sentinel)
    assert build_torch_reference(info,ROOT/'.cache/vendor/YOLOX') is sentinel
    assert calls==[{'yolox_root':ROOT/'.cache/vendor/YOLOX','face_input_size':960,'plate_input_size':640}]
    check_model_output(info,np.zeros((1,27300,7),np.float32))
    with pytest.raises(ValueError,match='specialist_output'):check_model_output(info,np.zeros((1,18900,7),np.float32))
    for patch in ({'task':'pothole'},{'input_size':[640,640]},{'plate_input_size':960},{'face_input_size':None},{'architecture':'unknown'}):
        with pytest.raises(ValueError):validate_architecture(info|patch)
    validate_architecture({'task':'pothole','architecture':'YOLOX-Nano'})
    validate_architecture({'task':'redaction'})

@pytest.mark.parametrize('structure',[
    {'architecture':'deeplabv3_mobilenet_v3_large','aspp_dilations':[6,12,18]},
    {'task':'redaction','classes':['face','license_plate'],'architecture':'two YOLOX-Nano specialists','input_size':[960,960],'face_input_size':960,'plate_input_size':640,'output_layout':'yolox_decoded_xywh'},
    {'task':'redaction','classes':['face','license_plate'],'architecture':'two YOLOX-Nano specialists global plus four tiles','input_size':[1920,1920],'face_input_size':960,'plate_input_size':640,'shared_input_size':1920,'face_tile_grid':2,'output_layout':'yolox_decoded_xywh'},
])
def test_artifact_package_preserves_non_parameter_structure_contract(artifact,structure):
    from scripts.model_artifacts import inspect_archive
    artifact.model.update(structure);artifact.manifest.write_text(json.dumps(artifact.model),encoding='utf-8')
    output,receipt,_=package_fixture(artifact)
    _,payload=inspect_archive(output,receipt['sha256']);restored=json.loads(payload['model.json'])
    for key,value in structure.items():assert restored[key]==value

def test_tiled_factory_metadata_cannot_silently_become_standard(monkeypatch):
    from scripts.verify_trained_models import build_torch_reference,validate_architecture,check_model_output
    import agent.redaction_tiled_network as factory
    info={'task':'redaction','classes':['face','license_plate'],'architecture':factory.ARCHITECTURE,'input_size':[1920,1920],'face_input_size':960,'plate_input_size':640,'shared_input_size':1920,'face_tile_grid':2}
    calls=[];sentinel=object()
    monkeypatch.setattr(factory,'build_redaction_tiled_network',lambda **kwargs:calls.append(kwargs) or sentinel)
    assert build_torch_reference(info,ROOT/'.cache/vendor/YOLOX') is sentinel
    assert calls==[{'yolox_root':ROOT/'.cache/vendor/YOLOX','shared_input_size':1920,'face_input_size':960,'plate_input_size':640}]
    check_model_output(info,np.zeros((1,102900,7),np.float32))
    with pytest.raises(ValueError,match='tiled_output'):check_model_output(info,np.zeros((1,27300,7),np.float32))
    for key in ('shared_input_size','face_tile_grid','face_input_size','plate_input_size','input_size'):
        missing=info.copy();missing.pop(key)
        with pytest.raises(ValueError):validate_architecture(missing)
    for patch in ({'shared_input_size':1920.0},{'shared_input_size':True},{'face_tile_grid':2.0},{'face_tile_grid':True},{'face_tile_grid':3},{'input_size':[960,960]},{'input_size':[1920.0,1920.0]},{'architecture':'two YOLOX-Nano specialists'},{'architecture':'YOLOX-Nano'},{'architecture':None}):
        with pytest.raises(ValueError):validate_architecture(info|patch)
