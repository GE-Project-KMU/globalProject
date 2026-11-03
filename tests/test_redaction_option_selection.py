"""Synthetic sealed-source option workflow; never opens trained weights or TEST."""
import json,shutil,tempfile
from pathlib import Path
from types import SimpleNamespace
import pytest
import numpy as np
from scripts import select_redaction_inference as s

@pytest.fixture
def fixture(monkeypatch):
    real=s.ROOT;base=real/'.cache/redaction-option-selection-tests';base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as folder:
        root=Path(folder);monkeypatch.setattr(s,'ROOT',root)
        for name in ('agent/redaction_network.py','agent/redaction_tiled_network.py','scripts/improve_redaction_train.py'):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(real/name,path)
        training=root/'data/training/new-specialists';training.mkdir(parents=True)
        source=root/'data/training/research-redaction-v4';source.mkdir(parents=True)
        release=source/'release.json';release.write_bytes(b'OPAQUE SEALED TEST METADATA: NOT JSON')
        release_sha=s.sha(release)
        image=source/'PRIVATE-SYNTHETIC-ONLY.png';image.write_bytes(b'SYNTHETIC NOT AN IMAGE')
        rows=[{'sample_id':kind+'-synthetic','source_id':kind+'-synthetic','split':'validation','evaluation_class':kind,'path':s.rel(image),'sha256':s.sha(image),'privacy_status':'restricted_research','qc_status':'publisher_annotations_machine_checked','release_scope':'offline_research','human_privacy_reviewed':False,'not_for_deployment':True,'expires_at':'2030-01-01T00:00:00Z','image':{'width':8,'height':8},'objects':[{'class':kind,'bbox_xywh_px':[1,1,2,2]}]} for kind in s.CLASSES]
        val=source/'validation-only.json';s.atomic_json(val,rows);val_sha=s.sha(val)
        proof=source/'validation-only.source.json';s.atomic_json(proof,{'parent_manifest_sha256':release_sha,'validation_manifest_sha256':val_sha,'only_split':'validation','source_expiries_not_extended':True,'counts':{'face':1,'license_plate':1},'source_expiry_min':'2030-01-01T00:00:00Z','training_manifest_unchanged':True,'test_images_or_predictions_opened':False})
        config={'architecture':'two YOLOX-Nano specialists','shared_input_size':960,'face_input_size':960,'plate_input_size':640,'release_sha256':release_sha,'research_only':True,'production_approved':False,'confidence_grid':list(s.GRID),'nms_iou':.5,'minimum_recall_goal':.95,'factory_sha256':s.STANDARD_FACTORY,'training_script_sha256':s.sha(root/'scripts/improve_redaction_train.py')}
        s.atomic_json(training/'training-config.json',config)
        calibration={'grid':list(s.GRID),'minimum_recall_goal':.95,'selection_data':'validation only','selected_confidence':.01}
        s.atomic_json(training/'confidence-calibration.json',calibration)
        (training/'best.pt').write_bytes(b'SYNTHETIC CHECKPOINT NOT TORCH')
        (training/'model.onnx').write_bytes(b'SYNTHETIC MODEL NOT ONNX')
        sealed=training/'model.json';sealed.write_text('SEALED TEST SCORE MUST NEVER BE OPENED')
        summary=training/'summary.json';summary.write_text('SEALED TEST SCORE MUST NEVER BE OPENED')
        receipt={'checkpoint_sha256':s.sha(training/'best.pt'),'onnx_sha256':s.sha(training/'model.onnx'),'calibration_sha256':s.sha(training/'confidence-calibration.json'),'model_manifest_sha256':'a'*64,'test_evaluated':False}
        s.atomic_json(training/'candidate-freeze.json',receipt)
        provenance={'format':'training-provenance-v1','task':'redaction','release_path':s.rel(release),'release_sha256':release_sha,'test_split_exported':False,'production_approved':False,'data_revoked':False,'needs_revalidation':False,'assets':[{'sample_id':'synthetic-train','identity_hashes':['b'*64],'expires_at':'2030-01-01T00:00:00Z'}],'models':[]}
        s.atomic_json(training/'provenance.json',provenance)
        def exporter(checkpoint,output):output.write_bytes(b'SYNTHETIC TILED ONNX NOT EXECUTABLE')
        monkeypatch.setattr(s,'export_tiled_checkpoint',exporter)
        def linker(provenance,manifest):
            value,_=s.json_snapshot(provenance);value['models'].append(s.rel(manifest));s.atomic_json(provenance,value)
        monkeypatch.setattr(s,'link_option_model',linker)
        original_open=Path.open
        def guarded_open(path,*args,**kwargs):
            if path in (sealed,summary):raise AssertionError('SEALED_TEST_ARTIFACT_OPENED')
            return original_open(path,*args,**kwargs)
        monkeypatch.setattr(Path,'open',guarded_open)
        def export(output=None):return s.export_options(training,release,release_sha,val,val_sha,output or root/'data/training/options',expected_counts={'face':1,'license_plate':1})
        yield SimpleNamespace(root=root,training=training,release=release,release_sha=release_sha,val=val,val_sha=val_sha,proof=proof,rows=rows,export=export,provenance=provenance)

def install_fake_app(monkeypatch,*,bad_box=False):
    class Model:
        backend='synthetic-unit-double';backend_version='none'
        def __init__(self,path):self.value,_=s.json_snapshot(path)
        def predict(self,image):
            label='face' if image[0,0,0]==1 else 'license_plate';other='license_plate' if label=='face' else 'face'
            return [{'class':label,'bbox_xywh_px':[-1,1,2,2] if bad_box else [1,1,2,2],'confidence':.8},{'class':other,'bbox_xywh_px':[0,0,1,1],'confidence':.9}]
    monkeypatch.setattr(s,'app_model',Model)
    monkeypatch.setattr(s,'decode_validation_image',lambda row:np.full((8,8,3),1 if row['evaluation_class']=='face' else 2,np.uint8))

def test_export_uses_safe_allowlist_and_keeps_test_sealed(fixture):
    old={name:s.sha(fixture.training/name) for name in s.SAFE_TRAINING_FILES if name!='provenance.json'}
    result=fixture.export();output=s.scoped(result['options_root'])
    receipt,_=s.json_snapshot(output/'export-receipt.json',result['export_receipt_sha256'])
    assert receipt['test_inputs_read'] is False and receipt['original_final_model_json_read'] is False
    assert receipt['shared_checkpoint_sha256']==old['best.pt']
    assert all(s.sha(fixture.training/name)==value for name,value in old.items())
    provenance,_=s.json_snapshot(fixture.training/'provenance.json')
    assert s.provenance_core(provenance)==s.provenance_core(fixture.provenance) and len(provenance['models'])==2
    for option in ('standard','tiled'):
        value,_=s.json_snapshot(output/('option-'+option)/'model.json')
        assert value['confidence']==.01 and value['no_new_gradient_steps'] is True and 'test_evaluated' not in value
    with pytest.raises(ValueError,match='output_already_exists'):fixture.export()

@pytest.mark.parametrize('target',['best.pt','model.onnx','confidence-calibration.json'])
def test_candidate_receipt_rejects_changed_source_before_export(fixture,target):
    (fixture.training/target).write_bytes(b'CHANGED SYNTHETIC INPUT')
    with pytest.raises((ValueError,json.JSONDecodeError)):fixture.export()
    assert not (fixture.root/'data/training/options').exists()

def test_validation_only_and_parent_binding_are_enforced(fixture):
    value,_=s.json_snapshot(fixture.val);value[0]['split']='test';s.atomic_json(fixture.val,value)
    with pytest.raises(ValueError,match='pinned_file_changed'):fixture.export()
    with pytest.raises(ValueError,match='validation_only'):s.load_validation(fixture.val,s.sha(fixture.val),{'face':1,'license_plate':1})
    s.atomic_json(fixture.val,fixture.rows)
    value,_=s.json_snapshot(fixture.proof);value['parent_manifest_sha256']='0'*64;s.atomic_json(fixture.proof,value)
    with pytest.raises(ValueError,match='parent_binding'):fixture.export()

def test_locked_or_tampered_export_is_not_committed(fixture,monkeypatch):
    output=fixture.root/'data/training/options';lock=output.with_name('.options.export.lock')
    with s.lock(lock):
        with pytest.raises(ValueError,match='already_locked'):fixture.export()
    def exporter(checkpoint,path):path.write_bytes(b'FAKE');lock.write_text('CHANGED TOKEN')
    monkeypatch.setattr(s,'export_tiled_checkpoint',exporter)
    with pytest.raises(ValueError,match='lock_changed'):fixture.export()
    assert not (output/'export-receipt.json').exists()

@pytest.mark.parametrize('target',['export-receipt.json','option-standard/model.json','option-tiled/model.onnx'])
def test_validation_rejects_receipt_manifest_and_weight_tampering(fixture,target):
    exported=fixture.export();output=s.scoped(exported['options_root']);(output/target).write_bytes(b'CHANGED SYNTHETIC OPTION')
    with pytest.raises((ValueError,json.JSONDecodeError)):s.validate_options(output,exported['export_receipt_sha256'])
    assert not (output/'option-selection-freeze.json').exists()

def test_known_class_scope_freezes_thresholds_models_and_standard_tie(fixture,monkeypatch):
    exported=fixture.export();output=s.scoped(exported['options_root']);install_fake_app(monkeypatch)
    public=fixture.root/'docs/reports/synthetic-aggregate.json'
    freeze=s.validate_options(output,exported['export_receipt_sha256'],public_report=public)
    assert freeze['selected_option']=='standard' and freeze['test_images_or_metrics_read'] is False
    aggregate,_=s.json_snapshot(public)
    assert aggregate['unknown_other_class_predictions_not_scored']=={'standard':2,'tiled':2}
    for option in ('standard','tiled'):
        value,_=s.json_snapshot(output/('option-'+option)/'model.json')
        assert value['confidence']==.5 and value['option_selection_pending'] is False
        assert s.sha(output/('option-'+option)/'model.json')==freeze['options'][option]['model_manifest_sha256']
        assert freeze['options'][option]['model_manifest_sha256']!=freeze['options'][option]['export_stage_manifest_sha256']
        assert all(c['fp']==0 for c in aggregate['calibrations'][option]['selected']['classes'].values())
    assert 'PRIVATE-SYNTHETIC' not in json.dumps(aggregate) and 'bbox_xywh_px' not in json.dumps(aggregate)
    assert (output/'validation-details.json').exists()
    with pytest.raises(ValueError,match='selection_output_already_exists'):s.validate_options(output,exported['export_receipt_sha256'])

def test_bad_prediction_does_not_freeze_a_selection(fixture,monkeypatch):
    exported=fixture.export();output=s.scoped(exported['options_root']);install_fake_app(monkeypatch,bad_box=True)
    with pytest.raises(ValueError,match='prediction_box'):s.validate_options(output,exported['export_receipt_sha256'])
    assert not (output/'option-selection-freeze.json').exists()

def test_image_decode_checks_fixed_bytes_before_decoding(fixture):
    row=fixture.rows[0]|{'sha256':'0'*64}
    with pytest.raises(ValueError,match='image_hash_mismatch'):s.decode_validation_image(row)

@pytest.mark.parametrize('standard,tiled,expected',[
    ((.95,.5),(.94,1.),'standard'),
    ((.95,.5),(.99,.4),'standard'),
    ((.8,.2),(.79,1.),'standard'),
    ((.8,.2),(.81,.1),'tiled'),
    ((.96,.6),(.96,.6),'standard'),
    ((.7,.6),(.7,.6),'standard'),
])
def test_predeclared_option_selection_policy(standard,tiled,expected):
    def calibration(point):return {'selected':{'min_class_recall':point[0],'macro_precision':point[1]}}
    assert s.select_option({'standard':calibration(standard),'tiled':calibration(tiled)})==expected

def test_calibration_one_to_one_matching_and_grid_tie():
    truth=[{'sample_id':label,'class':label,'bbox_xywh_px':[0,0,2,2]} for label in s.CLASSES]
    predictions=[row|{'confidence':.8} for row in truth]
    predictions.append(truth[0]|{'confidence':.6})
    result=s.calibrate(truth,predictions)
    assert result['selected']['confidence']==.5 and result['selected']['classes']['face']['tp']==1
    assert result['selected']['classes']['face']['fp']==1 and result['goal_achieved'] is True
    assert result['selected_on']=='validation_only'

@pytest.mark.parametrize('patch',[
    {'release_sha256':'0'*64},
    {'shared_input_size':1920},
    {'face_input_size':True},
    {'confidence_grid':[.01,.5]},
    {'nms_iou':.4},
    {'production_approved':True},
    {'factory_sha256':'0'*64},
])
def test_frozen_training_schema_rejects_incompatible_or_changed_configuration(fixture,patch):
    config,_=s.json_snapshot(fixture.training/'training-config.json');s.atomic_json(fixture.training/'training-config.json',config|patch)
    with pytest.raises(ValueError):fixture.export()
    assert not (fixture.root/'data/training/options').exists()

def test_changed_selection_lock_never_freezes_the_option(fixture,monkeypatch):
    exported=fixture.export();output=s.scoped(exported['options_root']);install_fake_app(monkeypatch)
    original=s.decode_validation_image
    def decoder(row):
        (output/'.selection.lock').write_text('MUTATED TOKEN')
        return original(row)
    monkeypatch.setattr(s,'decode_validation_image',decoder)
    with pytest.raises(ValueError,match='lock_changed'):s.validate_options(output,exported['export_receipt_sha256'])
    assert not (output/'option-selection-freeze.json').exists()
    for name in ('standard','tiled'):
        manifest,_=s.json_snapshot(output/('option-'+name)/'model.json')
        assert manifest['confidence']==.01 and manifest['option_selection_pending'] is True

def test_validation_rejects_missing_export_without_creating_output(fixture):
    output=fixture.root/'data/training/never-exported'
    with pytest.raises(ValueError,match='exported_options_directory_required'):s.validate_options(output,'0'*64)
    assert not output.exists()
