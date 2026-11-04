"""Synthetic portable-export policy tests; no trained checkpoint is loaded."""
import json,tempfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import export_lane_portable as e

@pytest.fixture
def fixture(monkeypatch):
    base=e.ROOT/'.cache/lane-portable-tests';base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as temporary:
        folder=Path(temporary);original=folder/'lane.json';weights=folder/'lane.onnx';checkpoint=folder/'best.pt';provenance=folder/'provenance.json';failure=folder/'verification-final.json'
        weights.write_bytes(b'SYNTHETIC ORIGINAL ONNX');checkpoint.write_bytes(b'SYNTHETIC NOT A REAL CHECKPOINT')
        body={'task':'lane','assets':[{'expires_at':'2030-01-01T00:00:00Z','sample_id':'synthetic'}],'models':[],'data_revoked':False,'needs_revalidation':False};e.write(provenance,body)
        model={'format':'onnx','task':'lane','classes':['background','paint'],'architecture':'deeplabv3_mobilenet_v3_large','aspp_dilations':[6,12,18],'weights':weights.relative_to(e.ROOT).as_posix(),'sha256':e.sha(weights),'training_provenance':provenance.relative_to(e.ROOT).as_posix(),'research_only':True,'production_approved':False,'project_acceptance_eligible':False,'input_size':[1024,1024],'paint_threshold':.4,'threshold_calibration':{'selected_on':'validation','report_sha256':'a'*64}}
        e.write(original,model);e.write(failure,{'manifest_sha256':e.sha(original),'weight_sha256':e.sha(weights),'numerical_parity_passed':False,'torch_checkpoint_compared':True})
        def exporter(checkpoint,model,target):target.write_bytes(b'SYNTHETIC PORTABLE REDUCEMEAN');return {'global1_pools_replaced':9,'reduce_mean_operators':9,'global_average_pool_operators':0,'strict_checkpoint_load':True}
        monkeypatch.setattr(e,'export_checkpoint',exporter)
        def linker(provenance,manifest):
            body=e.read(provenance);body['models'].append(manifest.relative_to(e.ROOT).as_posix());e.write(provenance,body)
        monkeypatch.setattr(e,'link',linker)
        def run():return e.export(original,e.sha(original),checkpoint,e.sha(checkpoint),folder/'portable')
        yield SimpleNamespace(folder=folder,original=original,weights=weights,checkpoint=checkpoint,provenance=provenance,failure=failure,model=model,run=run)

def test_new_artifact_preserves_original_failure_and_blocks_deployment(fixture):
    paths=(fixture.original,fixture.weights,fixture.checkpoint,fixture.failure);before=[e.sha(p) for p in paths];core=e.core_hash(e.read(fixture.provenance))
    receipt=fixture.run();model=e.read(fixture.folder/'portable/lane.json')
    assert [e.sha(p) for p in paths]==before and e.core_hash(e.read(fixture.provenance))==core
    assert model['original_onnx_sha256']==before[1] and model['original_checkpoint_sha256']==before[2]
    assert model['numerical_parity_passed'] is False and model['checkpoint_raw_tensor_parity_unresolved'] is True
    assert model['checkpoint_raw_tensor_parity_passed'] is False and model['trained_checkpoint_sha256']==before[2]
    assert model['diagnostic_accuracy_only'] is True and e.REASON in model['deployment_blocked_reasons']
    assert model['production_approved'] is False and model['project_acceptance_eligible'] is False
    assert model['paint_threshold']==.4 and 'threshold_calibration' not in model
    assert model['original_threshold_calibration']==fixture.model['threshold_calibration']
    assert model['export_transform']['global_pools_replaced']==9 and model['no_new_gradient_steps'] is True
    assert receipt['test_accessed'] is False and receipt['gpu_operations']==0
    with pytest.raises(ValueError,match='already_exists'):fixture.run()

def test_hash_pins_are_not_inferred_from_changed_files(fixture):
    old=e.sha(fixture.original);fixture.original.write_text('{}')
    with pytest.raises(ValueError,match='source_or_checkpoint_changed'):e.export(fixture.original,old,fixture.checkpoint,e.sha(fixture.checkpoint),fixture.folder/'portable')
    assert not (fixture.folder/'portable').exists()

def test_nonfailure_or_wrong_model_report_cannot_be_used(fixture):
    body=e.read(fixture.failure);body['numerical_parity_passed']=True;e.write(fixture.failure,body)
    with pytest.raises(ValueError,match='original_raw_parity_failure'):fixture.run()

def test_expired_training_evidence_prevents_export(fixture):
    body=e.read(fixture.provenance);body['assets'][0]['expires_at']='2020-01-01T00:00:00Z';e.write(fixture.provenance,body)
    with pytest.raises(ValueError,match='expired_or_revoked'):fixture.run()

def test_output_cannot_replace_original_or_sibling_workspace(fixture):
    with pytest.raises(ValueError,match='new_child'):e.export(fixture.original,e.sha(fixture.original),fixture.checkpoint,e.sha(fixture.checkpoint),fixture.folder)

def test_provenance_assets_or_deadline_mutation_is_refused(fixture,monkeypatch):
    def bad_link(provenance,manifest):
        body=e.read(provenance);body['assets'][0]['expires_at']='2035-01-01T00:00:00Z';e.write(provenance,body)
    monkeypatch.setattr(e,'link',bad_link)
    with pytest.raises(ValueError,match='assets_or_ttl_changed'):fixture.run()

def test_concurrent_export_lock_is_refused(fixture):
    with e.lock(fixture.folder/'.portable.export.lock'):
        with pytest.raises(ValueError,match='export_locked'):fixture.run()
