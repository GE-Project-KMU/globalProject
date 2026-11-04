"""Synthetic metadata binding and tolerance gates, no image inference."""
import copy,json,shutil,uuid
import pytest
from agent.paths import ROOT,sha256,atomic_json
from scripts.evaluate_public_refinement import strict_proof,strict_full_validation
@pytest.fixture
def files():
    folder=ROOT/'.cache/test-refinement-numeric'/uuid.uuid4().hex;folder.mkdir(parents=True)
    manifest=folder/'model.json';proof=folder/'proof.json';atomic_json(manifest,{'sha256':'a'*64})
    comparisons={key:{'passed':True,'rtol':.001,'atol':.0001} for key in ['app_repeatability','app_vs_ort','app_vs_torch','ort_vs_torch']}
    value={'manifest_sha256':sha256(manifest),'weight_sha256':'a'*64,'checkpoint_sha256':'b'*64,'image_sha256':'c'*64,
        'numerical_parity_passed':True,'postprocess_parity_passed':True,'artifact_contract_verified':True,'torch_checkpoint_compared':True,'synthetic_input':False,'adapter_result_count':1,**comparisons}
    atomic_json(proof,value);yield manifest,proof,value
    if not folder.resolve().is_relative_to(ROOT.resolve()):raise ValueError('unsafe_cleanup')
    shutil.rmtree(folder)
def test_bound_real_default_tolerance_positive_proof(files):
    manifest,proof,_=files;strict_proof(manifest,proof,image_sha='c'*64,checkpoint_sha='b'*64,positive=True)
@pytest.mark.parametrize('field',['numerical_parity_passed','postprocess_parity_passed','artifact_contract_verified','torch_checkpoint_compared'])
def test_any_failed_gate_stays_failed(files,field):
    manifest,proof,value=files;value[field]=False;atomic_json(proof,value)
    with pytest.raises(ValueError):strict_proof(manifest,proof)
@pytest.mark.parametrize('field',['manifest_sha256','weight_sha256'])
def test_artifact_binding(files,field):
    manifest,proof,value=files;value[field]='d'*64;atomic_json(proof,value)
    with pytest.raises(ValueError,match='artifact_binding'):strict_proof(manifest,proof)
@pytest.mark.parametrize('kwargs',[{'image_sha':'d'*64},{'checkpoint_sha':'d'*64}])
def test_predeclared_source_and_checkpoint_binding(files,kwargs):
    manifest,proof,_=files
    with pytest.raises(ValueError):strict_proof(manifest,proof,**kwargs)
@pytest.mark.parametrize('key',['app_repeatability','app_vs_ort','app_vs_torch','ort_vs_torch'])
def test_relaxed_tolerance_cannot_satisfy_gate(files,key):
    manifest,proof,value=files;value[key]['rtol']=.01;atomic_json(proof,value)
    with pytest.raises(ValueError,match='default_tolerance'):strict_proof(manifest,proof)
def test_synthetic_and_empty_predictions_not_positive_evidence(files):
    manifest,proof,value=files;value['synthetic_input']=True;atomic_json(proof,value)
    with pytest.raises(ValueError,match='real_validation'):strict_proof(manifest,proof)
    value['synthetic_input']=False;value['adapter_result_count']=0;atomic_json(proof,value)
    with pytest.raises(ValueError,match='nonempty_positive'):strict_proof(manifest,proof,positive=True)

def full_report():
    return {'manifest_sha256':'M','checkpoint_sha256':'C','complete':True,'all_raw_passed':True,'all_post_passed':True,'rtol':.001,'atol':.0001,
        'reference_runtime':{'ort':'1.31.1','providers':['CPUExecutionProvider'],'GPU_initialized':False},'app_runtime':{'engine':'onnxruntime-cpu','version':'1.31.1'},
        'cases':[{'index':i,'raw_passed':True,'post_passed':True,'raw':{n:{'passed':True,'rtol':.001,'atol':.0001} for n in ['app_vs_original_FP32_torch','app_vs_reference_ORT131','reference_ORT131_vs_original_FP32_torch']},
            'post':{n:{'exact_match':True,'mismatching_pixels':0} for n in ['app','torch','ort']}} for i in range(100)]}
def test_full_positive_contract():strict_full_validation(full_report(),'M','C')
@pytest.mark.parametrize('problem',['hidden_raw','hidden_post','relaxed_tol','duplicated_case','old_runtime','missing_comparison','false_summary'])
def test_full_scope_failures_cannot_be_overridden_by_summary(problem):
    r=full_report()
    if problem=='hidden_raw':r['cases'][4]['raw']['app_vs_original_FP32_torch']['passed']=False
    elif problem=='hidden_post':r['cases'][4]['post']['torch']['mismatching_pixels']=1
    elif problem=='relaxed_tol':r['cases'][4]['raw']['app_vs_original_FP32_torch']['rtol']=.01
    elif problem=='duplicated_case':r['cases'][99]['index']=98
    elif problem=='old_runtime':r['reference_runtime']['ort']='1.20.1'
    elif problem=='missing_comparison':r['cases'][4]['raw']={}
    else:r['all_raw_passed']=1
    with pytest.raises(ValueError):strict_full_validation(r,'M','C')
