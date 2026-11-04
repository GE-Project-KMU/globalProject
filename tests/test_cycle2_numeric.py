import copy,pytest
from scripts.evaluate_cycle2 import full_gate

def proof(count=3):
 names=['app_vs_original_FP32_torch','app_vs_reference_ORT131','reference_ORT131_vs_original_FP32_torch']
 return {'manifest_sha256':'m','checkpoint_sha256':'c','complete':True,'all_raw_passed':True,'all_post_passed':True,'rtol':.001,'atol':.0001,'reference_runtime':{'ort':'1.31.1','providers':['CPUExecutionProvider'],'GPU_initialized':False},'app_runtime':{'engine':'onnxruntime-cpu','version':'1.31.1'},'cases':[{'index':i,'raw_passed':True,'post_passed':True,'raw':{n:{'passed':True,'rtol':.001,'atol':.0001} for n in names},'post':{n:{'exact_match':True,'mismatching_pixels':0} for n in ['app','torch','ort']}} for i in range(count)]}

def test_all_real_count_cases_are_required_without_padding():full_gate(proof(180),'m','c',180)

@pytest.mark.parametrize('change',[
 lambda p:p.update(manifest_sha256='changed'),lambda p:p.update(checkpoint_sha256='changed'),lambda p:p.update(complete=False),lambda p:p.update(all_raw_passed=False),lambda p:p.update(rtol=.01),lambda p:p['reference_runtime'].update(ort='1.20.1'),lambda p:p['reference_runtime'].update(GPU_initialized=True),lambda p:p['cases'][2].update(index=1),lambda p:p['cases'][1].update(index=True),lambda p:p['cases'][0]['raw']['app_vs_original_FP32_torch'].update(passed=False),lambda p:p['cases'][0]['raw']['app_vs_original_FP32_torch'].update(atol=.001),lambda p:p['cases'][2]['post']['torch'].update(exact_match=False,mismatching_pixels=1),lambda p:p['cases'][0]['raw'].pop('app_vs_reference_ORT131')])
def test_aggregate_success_cannot_hide_a_single_raw_or_mask_failure(change):
 p=proof();change(p)
 with pytest.raises(ValueError):full_gate(p,'m','c',3)

def test_missing_case_fails():
 with pytest.raises(ValueError):full_gate(proof(2),'m','c',3)

@pytest.mark.parametrize('n',[True,0,-1,3.])
def test_expected_count_is_positive_integer(n):
 with pytest.raises(ValueError):full_gate(proof(),'m','c',n)
