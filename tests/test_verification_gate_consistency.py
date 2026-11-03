"""Fail-closed verification consistency and public-summary privacy, synthetic only."""
import copy,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import verify_trained_models as verify
from scripts.model_artifacts import verification_summary

def passing_lane_result():
    return {'task':'lane','metadata_valid':True,'weight_hash_verified':True,'finite_output':True,'preprocess_contract_verified':True,'torch_checkpoint_compared':True,
            'app_repeatability':{'passed':True,'max_abs_error':0.},'app_vs_ort':{'passed':True,'max_abs_error':0.},'app_vs_torch':{'passed':True,'max_abs_error':0.},'ort_vs_torch':{'passed':True,'max_abs_error':0.},
            'lane_adapter_postprocess_matches_reference':True,
            'mask_agreement':{name:{'pixel_agreement':1.,'paint_pixels':17,'paint_iou_against_app':1.} for name in ('app','ort','torch')},
            'image_sha256':'b'*64,'checkpoint_sha256':'c'*64,'privatecoords':[{'x':3,'y':7,'sample_id':'SYNTHETIC_PRIVATE_SAMPLE'}]}

def finalize(value):return verify.finalize_verification(copy.deepcopy(value),'lane')

def test_raw_pass_cannot_override_lane_adapter_failure():
    value=passing_lane_result();value['lane_adapter_postprocess_matches_reference']=False
    result=finalize(value)
    assert result['numerical_parity_passed'] is True
    assert result['postprocess_parity_passed'] is False and result['artifact_contract_verified'] is False
    assert result['status']=='postprocess_mismatch'
    assert verify.lane_postprocess_passed(value) is False

@pytest.mark.parametrize('name',['app','ort','torch'])
def test_missing_reference_mask_is_a_postprocess_failure(name):
    value=passing_lane_result();value['mask_agreement'].pop(name)
    result=finalize(value)
    assert result['numerical_parity_passed'] is True and result['postprocess_parity_passed'] is False
    assert result['artifact_contract_verified'] is False and result['status']=='postprocess_mismatch'

@pytest.mark.parametrize('name',['app','ort','torch'])
def test_nonexact_mask_does_not_pass(name):
    value=passing_lane_result();value['mask_agreement'][name]['pixel_agreement']=.999999
    result=finalize(value)
    assert result['postprocess_parity_passed'] is False and result['artifact_contract_verified'] is False
    assert result['status']=='postprocess_mismatch'

@pytest.mark.parametrize('name',['app_repeatability','app_vs_ort','app_vs_torch','ort_vs_torch'])
def test_missing_required_numeric_reference_fails_closed(name):
    value=passing_lane_result();value.pop(name)
    result=finalize(value)
    assert result['numerical_parity_passed'] is False and result['postprocess_parity_passed'] is True
    assert result['artifact_contract_verified'] is False and result['status']=='numerical_mismatch'

def test_raw_failure_is_preserved_even_with_exact_positive_masks():
    value=passing_lane_result();value['app_vs_torch']['passed']=False;value['app_vs_torch']['max_abs_error']=.087
    result=finalize(value)
    assert result['numerical_parity_passed'] is False and result['postprocess_parity_passed'] is True
    assert result['artifact_contract_verified'] is False and result['status']=='numerical_mismatch'
    assert result['app_vs_torch']==value['app_vs_torch']

def test_all_required_numeric_and_mask_contracts_pass_together():
    value=passing_lane_result();result=finalize(value)
    assert result['numerical_parity_passed'] is True and result['postprocess_parity_passed'] is True
    assert result['artifact_contract_verified'] is True and result['status']=='artifact_verified'
    # Local proof identity must be retained; public-summary filtering is separate.
    assert result['image_sha256']==value['image_sha256']

def test_ort_only_contract_does_not_fabricate_a_torch_comparison():
    value=passing_lane_result();value['torch_checkpoint_compared']=False
    for name in ('app_vs_torch','ort_vs_torch'):value.pop(name)
    value['mask_agreement'].pop('torch')
    result=finalize(value)
    assert result['artifact_contract_verified'] is True and result['status']=='artifact_verified'
    assert 'app_vs_torch' not in result and 'ort_vs_torch' not in result

@pytest.fixture
def proof_fixture():
    base=verify.ROOT/'.cache/verification-gate-consistency-tests';base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as temporary:
        folder=Path(temporary);assert folder.resolve().is_relative_to(verify.ROOT)
        manifest=folder/'model.json';model={'task':'lane','sha256':'a'*64};manifest.write_text(json.dumps(model),encoding='utf-8')
        proof=folder/'proof.json';value=finalize(passing_lane_result());value.update(manifest_sha256=verify.digest(manifest),weight_sha256=model['sha256'])
        def save(body):proof.write_text(json.dumps(body),encoding='utf-8')
        save(value)
        yield SimpleNamespace(path=proof,value=value,save=save,info={'manifest_path':manifest,'manifest':model})

@pytest.mark.parametrize('failure',['adapter','missing_ort','mismatched_torch','artifact_flag','numeric_flag','postprocess_flag'])
def test_public_package_summary_refuses_lane_failures_even_if_other_flags_claim_pass(proof_fixture,failure):
    body=copy.deepcopy(proof_fixture.value)
    if failure=='adapter':body['lane_adapter_postprocess_matches_reference']=False
    elif failure=='missing_ort':body['mask_agreement'].pop('ort')
    elif failure=='mismatched_torch':body['mask_agreement']['torch']['pixel_agreement']=.999
    elif failure=='artifact_flag':body['artifact_contract_verified']=False
    elif failure=='numeric_flag':body['numerical_parity_passed']=False
    else:body['postprocess_parity_passed']=False
    proof_fixture.save(body)
    with pytest.raises(ValueError):verification_summary(proof_fixture.path,proof_fixture.info)

def test_public_verification_summary_allowlist_omits_probe_hash_and_private_coordinates(proof_fixture):
    body=copy.deepcopy(proof_fixture.value)
    body['app_vs_ort'].update(image_sha256='b'*64,privatecoords=[1,2,3],max_abs_error=.001)
    body.update(app_backend='onnxruntime-cpu',app_backend_version='1.31.1',runtime_python='3.14.7',opencv_preprocessing_version='5.0.0')
    proof_fixture.save(body);summary=verification_summary(proof_fixture.path,proof_fixture.info);encoded=json.dumps(summary)
    assert 'image_sha256' not in encoded and 'privatecoords' not in encoded and 'SYNTHETIC_PRIVATE_SAMPLE' not in encoded
    assert 'b'*64 not in encoded and 'c'*64 not in encoded
    assert summary['manifest_sha256']==proof_fixture.value['manifest_sha256'] and summary['weight_sha256']=='a'*64
    assert summary['numerical_parity_passed'] is True and summary['production_approved'] is False
    assert summary['accuracy_evaluation_performed'] is False and summary['app_vs_ort']['max_abs_error']==.001
    # This test concerns the verification summary only, not intentionally retained
    # minimal training identity/source hashes in the separate provenance payload.
