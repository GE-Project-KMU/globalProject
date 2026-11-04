"""Aggregate frozen VAL evidence only; never changes a model, threshold or proof."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256
from agent.models import SUPPORTED_CPU_PROFILE
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
    folder=ROOT/'data/training/lane-a2d2-supported-ort131-v3';info=read(folder/'lane.json');full=read(folder/'full-validation-results.json')
    if info['required_runtime_profile']!=SUPPORTED_CPU_PROFILE:raise ValueError('exact_profile_required')
    if full.get('complete') is not True or len(full['cases'])!=100 or not full['all_raw_passed'] or not full['all_post_passed']:raise ValueError('strict_complete_VAL_required')
    if full['manifest_sha256']!=sha256(folder/'lane.json') or full['weights_sha256']!=info['sha256']:raise ValueError('proof_artifact_changed')
    if full['reference_runtimes']['ort']!='1.31.1' or full['app_backend_version']!='1.31.1':raise ValueError('actual_supported_versions_required')
    canonical=[]
    for i in range(3):
        p=folder/f'verification-val-{i}.json';v=read(p)
        if not v.get('artifact_contract_verified') or not v.get('numerical_parity_passed'):raise ValueError('canonical_raw_and_post_required')
        canonical.append(sha256(p))
    legacy=[]
    for name in ['lane-a2d2-fp64-mean-v3','lane-a2d2-fp64-head-mean-v3']:
        f=ROOT/'data/training'/name;v=read(f/'full-validation-results.json')
        if not v.get('complete') or v.get('all_post_passed') is not False:raise ValueError('legacy_failure_must_be_preserved')
        receipt=dict(status='not_portably_verified',validation_images=100,raw_failures=sum(not x['raw_passed'] for x in v['cases']),post_failures=sum(not x['post_passed'] for x in v['cases']),runtime='ORT1.20.1',original_proof_sha256=sha256(f/'full-validation-results.json'),accuracy_acceptance_passed=False)
        atomic_json(f/'artifact-status.json',receipt);legacy.append(receipt)
    rejected=read(folder/'legacy-runtime-rejection.json')
    if rejected.get('ready') is not False or 'runtime_profile_mismatch' not in rejected.get('reason',''):raise ValueError('actual_legacy_rejection_required')
    report=dict(format='lane-numeric-supported-profile-v1',status='verified_in_restricted_environment',required_runtime_profile=SUPPORTED_CPU_PROFILE,
        manifest_sha256=sha256(folder/'lane.json'),onnx_sha256=info['sha256'],checkpoint_sha256=info['trained_checkpoint_sha256'],
        full_validation_proof_sha256=sha256(folder/'full-validation-results.json'),canonical_proof_sha256=canonical,
        runtime_guard_sha256=sha256(ROOT/'agent/models.py'),eval_setup_sha256=sha256(ROOT/'scripts/setup_lane_eval.py'),
        validator_sha256=sha256(ROOT/'scripts/check_lane_precision_runtime.py'),
        original_torch_reference='FP32 unchanged trained state; CPU torch2.7.1',rtol=.001,atol=.0001,threshold=.4,
        full_validation_images=100,raw_comparison_pairs_per_image=3,raw_failures=0,post_exact_failures=0,canonical_positive_VAL=3,
        learned_state_unchanged=True,gradient_steps=0,threshold_reselected=False,test_accessed=False,accuracy_improvement_evidence=False,
        production_approved=False,project_acceptance_eligible=False,legacy_failures=legacy,
        actual_legacy_runtime_rejected=True,guard_contract_tests=20,vision_and_model_regression_tests=109,
        limits=['Only tested Windows AMD64 ORT1.31.1 CPU profile is supported; no general portability claim','VAL numerical parity is not independent accuracy validation','Existing TEST cohorts are regression only; no TEST accessed in this diagnostic','Original ORT1.20.1 failure remains a failure; no tolerance or mask criterion relaxed'])
    atomic_json(ROOT/'docs/reports/LANE-SUPPORTED-RUNTIME-NUMERIC.json',report)
    atomic_json(folder/'artifact-status.json',dict(status=report['status'],full_validation_proof_sha256=report['full_validation_proof_sha256'],accuracy_acceptance_passed=False,production_approved=False))
    print(json.dumps(report))
if __name__=='__main__':main()
