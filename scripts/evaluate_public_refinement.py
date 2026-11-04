"""Final-only actual-app comparisons after strict predeclared VAL gates.

Fixed output roots keep individual source identities/annotations private.
This tool never trains, calibrates thresholds or changes model manifests.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from scripts.compare_lane_models import compare as compare_lane
from scripts.compare_detector_models import compare as compare_detector
DEST=ROOT/'data/training/refinement-evaluation'
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def strict_proof(manifest_path,proof_path,*,image_sha=None,checkpoint_sha=None,positive=False):
    info=read(manifest_path);proof=read(proof_path)
    if proof.get('manifest_sha256')!=sha256(manifest_path) or proof.get('weight_sha256')!=info['sha256']:raise ValueError('proof_artifact_binding_mismatch')
    for key in ['numerical_parity_passed','postprocess_parity_passed','artifact_contract_verified','torch_checkpoint_compared']:
        if proof.get(key) is not True:raise ValueError('real_raw_and_postprocess_proof_required')
    if proof.get('synthetic_input') is not False:raise ValueError('real_validation_image_required')
    if image_sha is not None and proof.get('image_sha256')!=image_sha:raise ValueError('predeclared_VAL_case_mismatch')
    if checkpoint_sha is not None and proof.get('checkpoint_sha256')!=checkpoint_sha:raise ValueError('proof_checkpoint_binding_mismatch')
    for key in ['app_repeatability','app_vs_ort','app_vs_torch','ort_vs_torch']:
        value=proof.get(key,{})
        if value.get('passed') is not True or value.get('rtol')!=.001 or value.get('atol')!=.0001:raise ValueError('default_tolerance_proof_required')
    if positive and proof.get('adapter_result_count',0)<1:raise ValueError('nonempty_positive_validation_prediction_required')

def strict_full_validation(full,manifest_sha,checkpoint_sha):
    if full.get('manifest_sha256')!=manifest_sha or full.get('checkpoint_sha256')!=checkpoint_sha:raise ValueError('full_VAL_artifact_binding_required')
    if any(full.get(key) is not True for key in ['complete','all_raw_passed','all_post_passed']) or len(full.get('cases',[]))!=100:raise ValueError('complete_strict_100_VAL_gate_required')
    if full.get('rtol')!=.001 or full.get('atol')!=.0001:raise ValueError('full_VAL_default_tolerance_required')
    if (full.get('reference_runtime',{}).get('ort')!='1.31.1' or full.get('reference_runtime',{}).get('providers')!=['CPUExecutionProvider'] or
        full.get('reference_runtime',{}).get('GPU_initialized') is not False or full.get('app_runtime')!={'engine':'onnxruntime-cpu','version':'1.31.1'}):raise ValueError('actual_restricted_runtime_version_required')
    if {c.get('index') for c in full['cases']}!=set(range(100)):raise ValueError('full_VAL_unique_case_indices_required')
    names={'app_vs_original_FP32_torch','app_vs_reference_ORT131','reference_ORT131_vs_original_FP32_torch'}
    for case in full['cases']:
        if case.get('raw_passed') is not True or case.get('post_passed') is not True or set(case.get('raw',{}))!=names or set(case.get('post',{}))!={'app','torch','ort'}:raise ValueError('full_VAL_failure_cannot_be_hidden')
        if any(v.get('passed') is not True or v.get('rtol')!=.001 or v.get('atol')!=.0001 for v in case['raw'].values()):raise ValueError('full_VAL_raw_comparison_failed_or_relaxed')
        if any(v.get('exact_match') is not True or v.get('mismatching_pixels')!=0 for v in case['post'].values()):raise ValueError('full_VAL_exact_postprocess_required')

def gates(kind,plan):
    cases_path=DEST/'positive-validation-cases.json';numeric_plan=read(ROOT/'docs/reports/REFINEMENT-NUMERIC-GATE-PLAN.json')
    if sha256(cases_path)!=numeric_plan['positive_validation_cases_manifest_sha256']:raise ValueError('frozen_positive_cases_changed')
    cases=read(cases_path)
    if kind=='lane':
        for name,path in plan['models'].items():
            manifest=workspace_path(path,exists=True)
            checkpoint=manifest.parent/'lane_best.pt' if name=='candidate' else ROOT/'data/training/lane-a2d2-v2/lane_best.pt'
            for index,case in enumerate(cases['lane']):
                filename='verification-final.json' if index==0 else f'verification-val-{index}.json'
                strict_proof(manifest,manifest.parent/filename,image_sha=case['sha256'],checkpoint_sha=sha256(checkpoint),positive=True)
        candidate=workspace_path(plan['models']['candidate']);full=read(candidate.parent/'full-validation-131.json')
        strict_full_validation(full,sha256(candidate),sha256(candidate.parent/'lane_best.pt'))
        fullplan=candidate.parent/'full-validation-131-plan.json'
        if sha256(fullplan)!=full['plan_sha256']:raise ValueError('full_VAL_plan_changed')
        valplan=read(fullplan)
        if len(valplan['cases'])!=100 or any(c['image_sha256']!=valplan['cases'][c['index']]['sha256'] for c in full['cases']):raise ValueError('full_VAL_source_cases_changed')
    else:
        for name,path in plan['models'].items():
            manifest=workspace_path(path,exists=True)
            checkpoint=manifest.parent/'best.pt' if name=='candidate' else ROOT/'data/training/redaction-specialists-v4-efficient/best.pt'
            for label,filename in [('face','verification-final.json'),('license_plate','verification-plate.json')]:
                strict_proof(manifest,manifest.parent/filename,image_sha=cases['privacy'][label]['sha256'] if name=='candidate' else None,checkpoint_sha=sha256(checkpoint),positive=True)

def publish(kind):
    reports={role:read(DEST/f'{kind}-{role}.json') for role in ['fresh','regression'] if (DEST/f'{kind}-{role}.json').exists()}
    clean={}
    for role,report in reports.items():
        if kind=='lane':
            models={name:{k:value for k,value in result.items() if k in ['manifest_sha256','weights_sha256','paint_threshold','input_size','backend','backend_version','confusion','paint_iou','paint_precision','paint_recall','paint_dice','latency_seconds']} for name,result in report['results'].items()}
            difference={'paint_iou_change':report['paint_iou_change']}
        else:
            models=report['results'];difference={'map50_change':report['map50_change']}
        clean[role]={'role':report['scope'],'samples':report['samples'],'models':models,**difference}
    output=ROOT/f'docs/reports/{kind.upper()}-REFINEMENT-RESULTS.json'
    atomic_json(output,{'format':'public-model-refinement-results-v1','task':kind,'cohorts':clean,'physical_person_or_route_independence_verified':False,
        'old_VAL_is_reused_development':True,'old_TEST_is_regression_only':True,'test_used_for_model_or_threshold_selection':False,
        'user_recordings_used':False,'project_acceptance_eligible':False,'production_approved':False,'existing_source_expiries_extended':False,
        'limits':['New unseen source/recording cohort, not physical route/person independence','Known publisher paint proxy or conditional face/plate labels, not wear/complete privacy ground truth','Single-model CPU timings under shared host load; not full-pipeline or field FPS','No causal attribution to one architecture/loss change without ablation']})

def run(kind,role):
    plan_path=ROOT/f'docs/reports/{kind.upper()}-REFINEMENT-{role.upper()}-PLAN.json';plan=read(plan_path);gates(kind,plan)
    output=DEST/f'{kind}-{role}.json'
    result=(compare_lane if kind=='lane' else compare_detector)(plan_path,output)
    publish(kind);print(json.dumps({'kind':kind,'cohort':role,'samples':result['samples'],'output_written_private':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['lane','privacy']);p.add_argument('role',choices=['fresh','regression']);a=p.parse_args();run(a.kind,a.role)
