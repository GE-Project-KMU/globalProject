"""Separate retry1 policy -> CPU-gated artifact receipt -> one-shot TEST run.

Original cycle2 policies/workers stay immutable. Policy creation reads metadata
only and never needs an unfinished candidate artifact or holdout image bytes.
"""
from __future__ import annotations
import argparse,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import gpu_model_set as g
from scripts.evaluate_cycle2 import full_gate
from scripts.compare_lane_models import compare
from agent.governance import effective_expiry

PARENTS={role:ROOT/f'docs/reports/LANE-CYCLE2-{role.upper()}-PLAN.json' for role in ('fresh','regression')}
NUMERIC=ROOT/'docs/reports/LANE-CYCLE2-NUMERIC-PLAN.json'
CASES=ROOT/'data/training/cycle2-evaluation/lane-positive-VAL-cases.json'
LEARNING=ROOT/'docs/reports/LANE-CYCLE2-RETRY1-PLAN.json'
PARENT_LEARNING=ROOT/'docs/reports/LANE-CYCLE2-PLAN.json'
SUPERVISION=ROOT/'docs/reports/LANE-CYCLE2-RETRY1-SUPERVISION-PLAN-V2.json'
SUPERVISOR_RECEIPT=ROOT/'data/training/public-nextcycle/lane-retry1-training-supervisor.json'
ORIGINAL_DEST=ROOT/'data/training/cycle2-evaluation'
CANDIDATE=ROOT/'data/training/lane-cycle2-v5-retry1/lane.json'
DEST=ROOT/'data/training/lane-cycle2-retry1-evaluation'
POLICY=ROOT/'docs/reports/LANE-CYCLE2-RETRY1-EVALUATION-POLICY.json'
COUNTS={'fresh':119,'regression':180}
SOURCES=('scripts/evaluate_lane_retry1.py','scripts/evaluate_cycle2.py','scripts/compare_lane_models.py',
    'scripts/verify_trained_models.py','scripts/gpu_model_set.py','scripts/check_lane_cycle2_runtime.py',
    'agent/models.py','agent/dataset.py','scripts/evaluate_lane_model.py')


def read(path):return g.read(path)
def pointer(path):return g.pointer(path)
def pinned(ref):return g.pinned(ref)
def source_hashes():return {name:g.sha256(ROOT/name) for name in SOURCES}


def write_new(path,value):
    path=g.workspace_path(path);raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def source_alive(value):
    clock=datetime.fromisoformat(value.replace('Z','+00:00'))
    if clock.tzinfo is None or clock<=datetime.now(timezone.utc):raise ValueError('original_lane_source_expired_or_naive')
    return clock


def original_TEST_zero():
    for role in COUNTS:
        for suffix in ('-attempt.json','.json','-freeze.json','-per-image.json'):
            if (ORIGINAL_DEST/('lane-'+role+suffix)).exists():raise ValueError('original_cycle2_TEST_started_new_retry_policy_refused')


def cohort(parent,role):
    path=g.workspace_path(parent['benchmark_manifest'],exists=True)
    if g.sha256(path)!=parent['benchmark_sha256']:raise ValueError('original_lane_benchmark_changed')
    rows=read(path)
    if len(rows)!=COUNTS[role] or any(row.get('split')!='test' for row in rows) or len({r['sha256'] for r in rows})!=len(rows) or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('complete_unique_lane_TEST_metadata_required')
    for row in rows:source_alive(row['expires_at'])
    return rows


def policy():
    original_TEST_zero();numeric=read(NUMERIC);learning=read(LEARNING);parents={role:read(path) for role,path in PARENTS.items()}
    if numeric.get('task')!='lane' or numeric.get('positive_VAL_cases')!=3 or numeric.get('all_lane_VAL_count')!=179 or numeric.get('raw_rtol')!=.001 or numeric.get('raw_atol')!=.0001 or numeric.get('exact_postprocess_required') is not True or numeric.get('CPU_ORT')!='1.31.1' or numeric.get('TEST_used') is not False:raise ValueError('original_strict_numeric_policy_required')
    if g.sha256(CASES)!=numeric['positive_cases_sha256'] or len(read(CASES))!=3:raise ValueError('original_three_positive_case_plan_required')
    if learning.get('validation_count')!=179 or learning.get('optimizer_state_resumed') is not False or learning.get('data_unchanged_from_parent') is not True:raise ValueError('frozen_retry1_learning_policy_required')
    if learning.get('parent_plan_sha256')!=g.sha256(PARENT_LEARNING):raise ValueError('original_learning_parent_SHA_required')
    baseline=parents['fresh']['models']['baseline']
    for role,parent in parents.items():
        if parent.get('created_before_final_predictions') is not True or parent.get('models',{}).get('baseline')!=baseline or parent.get('candidate_release_sha256')!=learning['release_sha256'] or parent.get('numeric_plan_sha256')!=g.sha256(NUMERIC) or parent.get('raw_rtol')!=.001 or parent.get('raw_atol')!=.0001 or parent.get('test_used_for_fitting_or_selection') is not False or parent.get('verification_mode') is not None:raise ValueError('original_evaluation_parent_contract_changed')
        cohort(parent,role)
        if parent['baseline_manifest_sha256']!=g.sha256(baseline):raise ValueError('original_baseline_manifest_changed')
    fresh=cohort(parents['fresh'],'fresh');regression=cohort(parents['regression'],'regression')
    if {r['sha256'] for r in fresh}&{r['sha256'] for r in regression}:raise ValueError('fresh_regression_metadata_overlap')
    return {'format':'lane-cycle2-retry1-predeclared-evaluation-policy-v1','parent_comparison_policies':{role:pointer(path) for role,path in PARENTS.items()},
        'revision_reason':'Handoff said regression280; immutable original benchmark has180 unique rows. Keep original cohort/SHA and change only the candidate to retry1; no images were added or duplicated.',
        'parent_numeric_policy':pointer(NUMERIC),'parent_original_learning_policy':pointer(PARENT_LEARNING),'parent_retry_learning_policy':pointer(LEARNING),'parent_supervision_policy':pointer(SUPERVISION),
        'positive_VAL_case_plan':pointer(CASES),'candidate_manifest':g.workspace_path(CANDIDATE).relative_to(ROOT).as_posix(),
        'baseline_manifest':pointer(baseline),'cohort_samples':COUNTS,'default_raw_tolerance':{'rtol':.001,'atol':.0001},
        'exact_mask_required':True,'candidate_CPU_positive_cases':3,'candidate_full_VAL_count':179,
        'source_sha256':source_hashes(),'candidate_hash_bound_in_separate_receipt_after_completion':True,
        'TEST_used_for_selection':False,'automatic_retry_allowed':False,'original_cycle2_TEST_attempts':0,
        'actual_TEST_inference_requires_explicit_root_approval':True,'physical_route_independence_verified':False,
        'project_acceptance_eligible':False,'production_approved':False}


def prepare_policy():
    write_new(POLICY,policy());return {'policy_sha256':g.sha256(POLICY),'candidate_artifact_read':False,'TEST_images_read':0}


def validate_policy():
    if read(POLICY)!=policy():raise ValueError('retry1_evaluation_policy_or_parent_changed')
    return pointer(POLICY)


def research_model(ref):
    manifest=pinned(ref)
    if manifest.get('task')!='lane' or manifest.get('research_only') is not True or manifest.get('production_approved') is not False or g.sha256(manifest['weights'])!=manifest['sha256']:raise ValueError('frozen_research_lane_weights_required')
    if manifest.get('threshold_calibration',{}).get('selected_on')!='validation':raise ValueError('VAL_only_threshold_required')
    rows=read(manifest['source_release_path'])
    if manifest.get('dataset_release')!='manifest-sha256:'+g.sha256(manifest['source_release_path']):raise ValueError('frozen_training_release_required')
    provenance=read(manifest['training_provenance'])
    if provenance.get('release_sha256')!=g.sha256(manifest['source_release_path']) or any(value.get(key) is True for value in (manifest,provenance) for key in ('data_revoked','needs_revalidation')):raise ValueError('valid_training_provenance_required')
    for row in rows:source_alive(effective_expiry(row))
    for asset in provenance.get('assets',[]):source_alive(asset['expires_at'])
    return manifest,rows


def frozen_artifacts():
    parent=validate_policy();policy_value=read(POLICY);learning=read(LEARNING);supervision=read(SUPERVISION);receipt=read(SUPERVISOR_RECEIPT)
    if receipt.get('format')!='lane-exclusive-retry1-training-supervision-v1' or receipt.get('status')!='completed' or type(receipt.get('child_exit_code')) is not int or receipt['child_exit_code']!=0 or receipt.get('sources_unchanged') is not True or receipt.get('own_child_running_after_cleanup') is not False:raise ValueError('completed_retry_training_before_evaluation_required')
    if receipt.get('learning_plan_sha256')!=g.sha256(LEARNING) or receipt.get('supervision_plan_sha256')!=g.sha256(SUPERVISION) or receipt.get('learning_source_sha256')!=learning['training_code_sha256'] or receipt.get('supervisor_source_sha256')!=supervision['supervisor_source_sha256']:raise ValueError('retry_training_source_plan_binding_required')
    if supervision.get('learning_plan_sha256')!=g.sha256(LEARNING) or supervision.get('training_source_sha256')!=learning['training_code_sha256']:raise ValueError('supervision_learning_source_binding_required')
    if g.sha256(ROOT/'scripts/train_lane_cycle2_retry1.py')!=learning['training_code_sha256'] or g.sha256(ROOT/'scripts/supervise_lane_cycle2_retry1.py')!=supervision['supervisor_source_sha256']:raise ValueError('frozen_training_supervisor_source_changed')
    candidate=pointer(CANDIDATE);model,source=research_model(candidate);checkpoint=pointer(CANDIDATE.parent/'lane_best.pt')
    if candidate['path']!=policy_value['candidate_manifest'] or model['dataset_release']!='manifest-sha256:'+learning['release_sha256']:raise ValueError('retry_candidate_source_binding_required')
    dev_path=g.workspace_path(model['training_development_release_path'],exists=True)
    if g.sha256(dev_path)!=model.get('training_development_release_sha256') or g.sha256(dev_path)!=learning['development_sha256']:raise ValueError('retry_development_changed')
    rows=[r for r in read(dev_path) if r['split']=='validation']
    if len(rows)!=179 or len({r['sha256'] for r in rows})!=179:raise ValueError('all179_unique_VAL_required')
    cases=read(CASES);by_hash={row['sha256']:row for row in rows};proofs=[]
    if len({case['sha256'] for case in cases})!=3:raise ValueError('distinct_three_canonical_cases_required')
    for index,case in enumerate(cases):
        row=by_hash.get(case['sha256'])
        if row is None or row.get('positive_pixels',0)<=0 or any(case[key]!=row[key] for key in ('path','sha256','expires_at')):raise ValueError('original_positive_VAL_case_binding_required')
        path=CANDIDATE.parent/('verification-final.json' if index==0 else f'verification-val-{index}.json')
        ref=pointer(path);proof=pinned(ref);g.proof_passed(proof,model,candidate['sha256'])
        if proof.get('image_sha256')!=case['sha256'] or proof.get('checkpoint_sha256')!=checkpoint['sha256']:raise ValueError('canonical_case_checkpoint_binding_required')
        proofs.append(ref)
    full_path=CANDIDATE.parent/'full-validation-131.json';full=read(full_path);full_plan_path=CANDIDATE.parent/'full-validation-131-plan.json';full_plan=read(full_plan_path)
    full_gate(full,candidate['sha256'],checkpoint['sha256'],179)
    if full.get('plan_sha256')!=g.sha256(full_plan_path) or full.get('weights_sha256')!=model['sha256'] or full.get('worker_source_sha256')!=g.sha256(ROOT/'scripts/check_lane_cycle2_runtime.py') or full.get('test_accessed') is not False:raise ValueError('all179_plan_source_weight_binding_required')
    expected=[{key:r[key] for key in ('path','sha256','expires_at')} for r in sorted(rows,key=lambda r:r['source_id'])]
    if full_plan.get('manifest_sha256')!=candidate['sha256'] or full_plan.get('checkpoint_sha256')!=checkpoint['sha256'] or full_plan.get('development_sha256')!=g.sha256(dev_path) or full_plan.get('cases')!=expected or full_plan.get('test_accessed') is not False or full_plan.get('rtol')!=.001 or full_plan.get('atol')!=.0001 or any(c['image_sha256']!=expected[c['index']]['sha256'] for c in full['cases']):raise ValueError('all179_original_case_plan_required')
    calibration=pointer(CANDIDATE.parent/'best-validation.json');best=pinned(calibration);history=read(CANDIDATE.parent/'history.json')
    if model['threshold_calibration'].get('report_sha256')!=calibration['sha256'] or model['threshold_calibration'].get('validation_samples')!=179 or model['threshold_calibration'].get('test_images_used')!=0 or model['paint_threshold']!=best['selected']['paint_threshold'] or model['best_epoch']!=best['epoch'] or model['best_epoch']!=history['best_epoch'] or history.get('test_used') is not False:raise ValueError('frozen_VAL_checkpoint_threshold_selection_required')
    base_ref=policy_value['baseline_manifest'];baseline,base_source=research_model(base_ref)
    baseline_proof=pointer(g.workspace_path(base_ref['path']).parent/'verification-final.json');g.proof_passed(pinned(baseline_proof),baseline,base_ref['sha256'])
    for role in COUNTS:
        benchmark=cohort(read(PARENTS[role]),role);hashes={r['sha256'] for r in benchmark};groups={r['leakage_group_id'] for r in benchmark}
        for pool in (source,base_source):
            development=[r for r in pool if r['split'] in ('train','validation')]
            if hashes&{r['sha256'] for r in development} or groups&{r['leakage_group_id'] for r in development}:raise ValueError('TEST_training_or_VAL_metadata_leakage')
    return {'format':'lane-cycle2-retry1-completed-CPU-artifact-freeze-v1','parent_policy':parent,
        'candidate_manifest':candidate,'checkpoint':checkpoint,'weights_sha256':model['sha256'],'development_manifest':pointer(dev_path),
        'source_release':pointer(g.workspace_path(model['source_release_path'])),'canonical_proofs':proofs,
        'all179_proof':pointer(full_path),'all179_plan':pointer(full_plan_path),'calibration':calibration,
        'training_history':pointer(CANDIDATE.parent/'history.json'),'training_completion':pointer(SUPERVISOR_RECEIPT),
        'candidate_training_provenance':pointer(g.workspace_path(model['training_provenance'])),
        'baseline_training_provenance':pointer(g.workspace_path(baseline['training_provenance'])),
        'baseline_manifest':base_ref,'baseline_strict_proof':baseline_proof,'evaluation_source_sha256':source_hashes(),
        'paint_threshold':model['paint_threshold'],'TEST_used_for_selection':False,'numerical_gate_relaxed':False}


def freeze():
    if (DEST/'candidate-freeze.json').exists():raise ValueError('retry_artifact_already_frozen')
    value=frozen_artifacts();value['artifact_digest']=g.canonical(value);write_new(DEST/'candidate-freeze.json',value)
    return {'artifact_digest':value['artifact_digest'],'TEST_images_read':0}


def validated_freeze():
    value=read(DEST/'candidate-freeze.json');body={k:v for k,v in value.items() if k!='artifact_digest'}
    if value.get('artifact_digest')!=g.canonical(body) or body!=frozen_artifacts():raise ValueError('frozen_retry_artifact_or_proof_changed')
    return value


def comparison_plan(value,role):
    if role not in COUNTS:raise ValueError('known_lane_comparison_role_required')
    return read(PARENTS[role])|{'format':'lane-cycle2-retry1-one-shot-comparison-v1',
        'models':{'baseline':value['baseline_manifest']['path'],'candidate':value['candidate_manifest']['path']},
        'parent_original_policy':pointer(PARENTS[role]),'retry_policy':pointer(POLICY),
        'retry_artifact_freeze':pointer(DEST/'candidate-freeze.json'),'artifact_digest':value['artifact_digest'],
        'raw_rtol':.001,'raw_atol':.0001,'automatic_retry_allowed':False,'TEST_used_for_selection':False}


def prepare_plans():
    value=validated_freeze();plans={role:comparison_plan(value,role) for role in COUNTS}
    if any((DEST/(role+'-plan.json')).exists() for role in plans):raise ValueError('retry_comparison_plans_already_exist')
    for role,plan in plans.items():write_new(DEST/(role+'-plan.json'),plan)
    return {'plans':2,'TEST_images_read':0}


def run(role,*,approved=False):
    if approved is not True:raise ValueError('explicit_root_TEST_approval_required')
    value=validated_freeze();plan=DEST/(role+'-plan.json')
    if read(plan)!=comparison_plan(value,role):raise ValueError('frozen_retry_comparison_plan_changed')
    output=DEST/(role+'-comparison.json');attempt=DEST/(role+'-attempt.json')
    if output.exists() or attempt.exists() or output.with_name(output.stem+'-freeze.json').exists():raise ValueError('retry_TEST_already_started_no_auto_retry')
    record={'status':'started','plan_sha256':g.sha256(plan),'artifact_digest':value['artifact_digest'],'selection_allowed':False}
    write_new(attempt,record)
    result=compare(plan,output)
    write_new(DEST/(role+'-completion.json'),record|{'status':'completed','samples':result['samples'],'result_sha256':g.sha256(output)})
    return {'role':role,'samples':result['samples'],'TEST_used_for_selection':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['policy','freeze','prepare','run']);parser.add_argument('role',nargs='?',choices=list(COUNTS));parser.add_argument('--approved-test-inference',action='store_true');args=parser.parse_args()
    if args.action=='policy':value=prepare_policy()
    elif args.action=='freeze':value=freeze()
    elif args.action=='prepare':value=prepare_plans()
    else:
        if args.role is None:parser.error('run requires fresh or regression')
        value=run(args.role,approved=args.approved_test_inference)
    print(json.dumps(value))


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(json.dumps({'status':'retry1_evaluation_refused','error_type':type(error).__name__,'success':False}));raise SystemExit(2)
