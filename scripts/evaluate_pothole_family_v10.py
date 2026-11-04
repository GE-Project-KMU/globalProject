"""V10 geometry family: same actual-app CPU VAL652 ranking, then one TEST attempt.

V9 source/policy and native/GPU metrics are preserved, never used for V10 ranking.
Policy/family/plan actions read metadata/weights only, not source photographs.
"""
import argparse,json,math,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import evaluate_pothole_family as v9

RES='data/training/pothole-resolution-v10';DEST='data/training/pothole-family-v10'
POLICY='docs/reports/POTHOLE-FAMILY-V10-ONE-SHOT-PLAN.json'
TRIAL_SHA='9913b5cccee29b0ee665dc29fa27a80562cdf29f3b010bff4c7be35b314a1e4d'
V9_POLICY_SHA='732b850fe3873f56eaeb9859367ef92c41ad2703af7cd4a9c204dd12fc15bbac'
ORDER=('baseline','nano','small','nano960','nano1280')
RUNTIME={'engine':'onnxruntime-cpu','version':'1.31.1','providers':['CPUExecutionProvider'],'GPU_used':False}
EVAL_SOURCES=(*v9.EVAL_SOURCES,'scripts/evaluate_pothole_family_v10.py','scripts/evaluate_pothole_resolution.py')


def v9_test_zero():
    for role in ('fresh','regression'):
        for suffix in ('attempt.json','comparison.json','completion.json','comparison-freeze.json'):
            if v9.workspace_path(v9.DEST+'/'+role+'-'+suffix).exists():raise ValueError('v9_TEST_started_v10_cannot_reuse_cohort')


def membership(rows):
    return v9.digest([{key:row[key] for key in ('sample_id','sha256','development_domain')} for row in sorted(rows,key=lambda row:row['sample_id'])])


def trial():
    v9_test_zero();v9.validate_policy()
    if v9.sha256(v9.POLICY)!=V9_POLICY_SHA or v9.sha256(RES+'/trial-plan.json')!=TRIAL_SHA or v9.sha256('docs/reports/POTHOLE-RESOLUTION-V10-PLAN.json')!=TRIAL_SHA:raise ValueError('frozen_parent_v9_or_resolution_policy_changed')
    value=v9.read(RES+'/trial-plan.json')
    if value.get('format')!='pothole-resolution-v10-predeclared-plan-v1' or value.get('input_sizes')!=[640,960,1280] or value.get('new_exports')!=[960,1280] or value.get('CPU_family_members')!=list(ORDER) or value.get('new_training_updates')!=0 or value.get('parameter_or_checkpoint_changes') is not False or value.get('TEST_images')!=0 or any(value.get(key) is not False for key in ('TEST_used_for_choice','user_recordings_used','GPU_used','source_expiry_extended')) or value.get('confidence_floor')!=.01 or value.get('operating_confidence')!=.1 or value.get('nms_iou')!=.5:raise ValueError('predeclared_resolution_contract_changed')
    if value['source_sha256']!=v9.sha256('scripts/evaluate_pothole_resolution.py') or any(v9.sha256(path)!=pin for path,pin in value['dependencies'].items()):raise ValueError('frozen_resolution_execution_sources_changed')
    for key in ('parent_manifest','checkpoint','validation_release','parent_family_policy','baseline_manifest','small_learning_plan','small_retry_protocol'):
        ref=value[key]
        if v9.sha256(ref['path'])!=ref['sha256']:raise ValueError('frozen_resolution_parent_changed')
    if value['parent_family_policy']!=v9.pointer(v9.POLICY):raise ValueError('resolution_parent_policy_mismatch')
    rows=[row for row in v9.read(value['validation_release']['path']) if row['split']=='validation']
    if Counter(row['development_domain'] for row in rows)!={'original':402,'mwpd':250} or len({row['sample_id'] for row in rows})!=652 or value.get('VAL_membership_sha256')!=membership(rows):raise ValueError('exact_shared_VAL652_membership_required')
    for row in rows:v9.source_deadline(row['expires_at'],required=True)
    return value,rows


def protocol():
    value,_=trial()
    return {**v9.protocol(),'format':'pothole-family-v10-predeclared-evaluation-v1',
        'parent_v9_policy':v9.pointer(v9.POLICY),'parent_v9_module_sha256':v9.sha256('scripts/evaluate_pothole_family.py'),
        'resolution_trial':v9.pointer(RES+'/trial-plan.json'),'resolution_public_plan':v9.pointer('docs/reports/POTHOLE-RESOLUTION-V10-PLAN.json'),
        'selection':'All available options evaluated on the same actual-app ORTCPU1.31.1 VAL652; oldAP>=same-CPU-baseline-.03; maximum two-domain mean; exact ties use declared order',
        'same_CPU_tie_order':list(ORDER),'native_or_GPU_scores_used_for_family_selection':False,
        'checkpoint_training_selection_preserved':True,'CPU_report_used_to_construct_policy':False,
        'input_geometry_options':[640,960,1280],'new_training_updates':0,'selection_rule_predeclared_in_resolution_trial':True}


def prepare_policy():
    value=protocol();v9.write_new(POLICY,value)
    return {'policy_sha256':v9.sha256(POLICY),'TEST_inferences':0,'source_images_read':0}


def validate_policy():
    if v9.read(POLICY)!=protocol():raise ValueError('v10_policy_or_parent_changed')
    return v9.pointer(POLICY)


def input_members():
    retry=v9.retry_contract()
    values={'baseline':v9.baseline_member(),'tiny':v9.tiny_rejected(),'nano':v9.trained_member(v9.NANO,packaged=True),'small':v9.trained_member(v9.SMALL)}
    if values['nano']['status']!='available':raise ValueError('frozen_Nano_checkpoint_required_for_geometry_trial')
    # Native numbers certify only which checkpoint was frozen by its trainer.
    # Deliberately strip AP fields from V10 identity records and ranking inputs.
    clean={name:{key:item for key,item in value.items() if key!='validation'}|
        ({'training_selected_epoch':value['validation']['epoch']} if 'validation' in value else {}) for name,value in values.items()}
    return clean,retry


def exports(value):
    report=v9.read(RES+'/exports.json')
    if report.get('plan_sha256')!=TRIAL_SHA or report.get('checkpoint_unchanged') is not True or report.get('GPU_used') is not False or report.get('TEST_images')!=0 or set(report.get('models',{}))!={'nano960','nano1280'}:raise ValueError('frozen_CPU_resolution_exports_required')
    parent=v9.read(value['parent_manifest']['path']);result={}
    for size in (960,1280):
        name='nano'+str(size);folder=RES+'/'+name;entry=report['models'][name]
        if entry['manifest']!=v9.pointer(folder+'/model.json') or entry['checkpoint']!=v9.pointer(folder+'/best.pt') or entry['checkpoint']['sha256']!=value['checkpoint']['sha256']:raise ValueError('same_checkpoint_resolution_binding_required')
        artifact=v9.read(folder+'/model.json');v9.detector_contract(artifact)
        expected={key:item for key,item in parent.items() if key not in ('executed_export_manifest_sha256','selection_freeze_sha256','packaging_source_sha256')}
        expected.update(input_size=[size,size],weights=folder+'/model.onnx',sha256=v9.sha256(folder+'/model.onnx'),
            inference_parent_manifest_sha256=value['parent_manifest']['sha256'],inference_option_plan_sha256=TRIAL_SHA,inference_option_new_training_updates=0)
        if artifact!=expected or entry['weights_sha256']!=artifact['sha256'] or entry['input_size']!=[size,size] or entry['output_shape']!=[1,sum((size//stride)**2 for stride in (8,16,32)),6]:raise ValueError('resolution_changed_parameters_or_non_geometry_contract')
        result[name]={'status':'available','manifest':entry['manifest'],'checkpoint':entry['checkpoint'],'weights_sha256':entry['weights_sha256'],'input_size':[size,size]}
    return result


def cpu_score(entry):
    if set(entry.get('domains',{}))!={'original','mwpd'}:raise ValueError('two_CPU_validation_domains_required')
    values=[]
    for domain,count in (('original',402),('mwpd',250)):
        row=entry['domains'][domain];value=row['ap50']
        if type(row.get('images')) is not int or row['images']!=count or type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('finite_same_CPU_VAL652_scores_required')
        values.append(value)
    return tuple(values)


def choose_cpu(models):
    if set(models)!=set(ORDER) or any(models[name].get('status')!='available' for name in ('baseline','nano','nano960','nano1280')) or models['small'].get('status') not in ('available','rejected_no_artifact'):raise ValueError('complete_same_CPU_family_required')
    reference=cpu_score(models['baseline'])[0];eligible=[];audit={}
    for rank,name in enumerate(ORDER):
        entry=models[name]
        if entry['status']=='rejected_no_artifact':audit[name]={'eligible':False,'reason':'native_trainer_selected_zero_no_export'};continue
        old,new=cpu_score(entry);ok=old>=reference-.03
        audit[name]={'eligible':ok,'CPU_old_validation_ap50':old,'CPU_new_validation_ap50':new,'CPU_mean_validation_ap50':(old+new)/2}
        if ok:eligible.append(((old+new)/2,-rank,name))
    return {'selected_member':max(eligible)[2],'same_CPU_only':True,'old_validation_drop_limit':.03,'tie_order':list(ORDER),'members':audit}


def cpu_report(value,rows,members,variants):
    attempt=v9.read(RES+'/cpu-validation-attempt.json');report=v9.read(RES+'/cpu-validation.json')
    if attempt.get('plan_sha256')!=TRIAL_SHA or attempt.get('status')!='started' or attempt.get('TEST_images')!=0:raise ValueError('CPU_evaluation_attempt_binding_required')
    if report.get('format')!='pothole-resolution-v10-CPU-VAL652' or report.get('complete') is not True or report.get('plan_sha256')!=TRIAL_SHA or report.get('validation_release_sha256')!=value['validation_release']['sha256'] or report.get('VAL_membership_sha256')!=membership(rows) or report.get('runtime')!=RUNTIME or type(report.get('TEST_images')) is not int or report['TEST_images']!=0 or report.get('confidence_floor')!=.01 or report.get('nms_iou')!=.5 or set(report.get('models',{}))!=set(ORDER):raise ValueError('complete_same_actual_CPU_VAL652_report_required')
    identities={name:members[name] for name in ('baseline','nano','small')}|variants
    for name,identity in identities.items():
        entry=report['models'][name]
        if identity['status']=='rejected_no_artifact':
            if entry!={'status':'rejected_no_artifact','selection_freeze':identity['selection_freeze']}:raise ValueError('rejected_Small_cannot_supply_AP_or_artifact')
            continue
        artifact=v9.pinned(identity['manifest']);v9.detector_contract(artifact)
        expected_size=[int(name[4:]),int(name[4:])] if name in variants else [640,640]
        if entry.get('status')!='available' or entry.get('manifest')!=identity['manifest'] or entry.get('checkpoint')!=identity['checkpoint'] or entry.get('weights_sha256')!=identity['weights_sha256'] or entry.get('input_size')!=expected_size or artifact['input_size']!=expected_size or v9.sha256(artifact['weights'])!=entry['weights_sha256']:raise ValueError('CPU_report_artifact_or_input_binding_changed')
        cpu_score(entry)
    return report,identities


def family_core():
    policy=validate_policy();value,rows=trial();members,retry=input_members();variants=exports(value)
    report,identities=cpu_report(value,rows,members,variants)
    return {'format':'pothole-family-v10-CPU-validation-freeze-v1','parent_policy':policy,'parent_v9_policy':v9.pointer(v9.POLICY),
        'resolution_trial':v9.pointer(RES+'/trial-plan.json'),'CPU_validation_report':v9.pointer(RES+'/cpu-validation.json'),
        'CPU_validation_attempt':v9.pointer(RES+'/cpu-validation-attempt.json'),'exports':v9.pointer(RES+'/exports.json'),
        'members':identities,'tiny_excluded':members['tiny'],'retry_contract':retry,'selection':choose_cpu(report['models']),
        'native_or_GPU_AP_used_for_family_ranking':False,'selection_source_sha256':v9.sha256(Path(__file__)),
        'source_images_read_during_selection':0,'TEST_used_for_selection':False,'post_TEST_reselection_allowed':False}


def freeze_family():
    path=DEST+'/family-freeze.json'
    if v9.workspace_path(path).exists():raise ValueError('v10_family_already_frozen')
    value=family_core();value['family_digest']=v9.digest(value);v9.write_new(path,value)
    return {'family_digest':value['family_digest'],'selected_member':value['selection']['selected_member'],'TEST_inferences':0,'source_images_read':0}


def validated_family():
    value=v9.read(DEST+'/family-freeze.json');core={key:item for key,item in value.items() if key!='family_digest'}
    if value.get('family_digest')!=v9.digest(core) or core!=family_core():raise ValueError('v10_family_digest_or_CPU_report_changed')
    return value


def comparison_plan(family,role):
    if role not in v9.PARENTS:raise ValueError('unknown_comparison_role')
    policy=v9.pinned(family['parent_policy']);cohort=policy['cohorts'][role]
    if v9.sha256(cohort['path'])!=cohort['sha256']:raise ValueError('fixed_benchmark_changed')
    rows=v9.read(cohort['path'])
    if len(rows)!=cohort['samples'] or any(row['split']!='test' or any(obj['class']!='pothole' for obj in row['objects']) for row in rows) or sum(bool(row['objects']) for row in rows)!=cohort['positive'] or len({row['sha256'] for row in rows})!=len(rows) or len({row['sample_id'] for row in rows})!=len(rows):raise ValueError('fixed_TEST_cohort_contract_changed')
    for row in rows:v9.source_deadline(row['expires_at'],required=True)
    cases,proofs=v9.proof_refs(family);selected=family['members'][family['selection']['selected_member']];base=family['members']['baseline'];artifact=v9.pinned(selected['manifest'])
    return {'format':'pothole-family-v10-one-shot-comparison-v1','role':'US100_new_blind_source_proxy' if role=='fresh' else 'MWPD97_RDD48_historical_regression_only',
        'models':{'baseline':base['manifest']['path'],'candidate':selected['manifest']['path']},'baseline_manifest_sha256':base['manifest']['sha256'],
        'candidate_release_sha256':artifact['dataset_release'].removeprefix('manifest-sha256:'),'benchmark_manifest':cohort['path'],'benchmark_sha256':cohort['sha256'],
        'family_freeze':v9.pointer(DEST+'/family-freeze.json'),'family_digest':family['family_digest'],'parent_policy':family['parent_policy'],
        'parent_v9_policy':family['parent_v9_policy'],'resolution_trial':family['resolution_trial'],'CPU_validation_report':family['CPU_validation_report'],
        'parent_benchmark_plan':policy['benchmark_parent_plans'][role],'positive_VAL_case_plan':cases,'strict_proofs':proofs,
        'evaluation_sources':{path:v9.sha256(path) for path in EVAL_SOURCES},'TEST_used_for_selection':False,'automatic_retry_allowed':False}


def prepare_comparison_plans():
    family=validated_family();plans={role:comparison_plan(family,role) for role in v9.PARENTS}
    if any(v9.workspace_path(DEST+'/'+role+'-plan.json').exists() for role in plans):raise ValueError('v10_comparison_plans_already_prepared')
    for role,plan in plans.items():v9.write_new(DEST+'/'+role+'-plan.json',plan)
    return {'plans':2,'family_digest':family['family_digest'],'TEST_inferences':0}


def run(role):
    family=validated_family();path=DEST+'/'+role+'-plan.json';plan=v9.read(path)
    if plan!=comparison_plan(family,role):raise ValueError('v10_comparison_family_or_proofs_changed')
    output=v9.workspace_path(DEST+'/'+role+'-comparison.json');attempt=v9.workspace_path(DEST+'/'+role+'-attempt.json')
    if output.exists() or attempt.exists():raise ValueError('v10_one_shot_already_started')
    receipt={'status':'started','family_digest':family['family_digest'],'plan_sha256':v9.sha256(path),'CPU_report_sha256':family['CPU_validation_report']['sha256'],'selection_allowed':False}
    v9.write_new(attempt,receipt)
    result=v9.compare(path,output)
    v9.write_new(DEST+'/'+role+'-completion.json',receipt|{'status':'completed','samples':result['samples'],'comparison_sha256':v9.sha256(output)})
    return {'role':role,'samples':result['samples'],'TEST_used_for_selection':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['policy','freeze','prepare','run']);parser.add_argument('role',nargs='?',choices=['fresh','regression']);args=parser.parse_args()
    if args.action=='policy':value=prepare_policy()
    elif args.action=='freeze':value=freeze_family()
    elif args.action=='prepare':value=prepare_comparison_plans()
    else:
        if args.role is None:parser.error('run requires fresh or regression')
        value=run(args.role)
    print(json.dumps(value))


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,TypeError) as error:
        reason=str(error) if isinstance(error,ValueError) and all(char in 'abcdefghijklmnopqrstuvwxyz_0123456789' for char in str(error)) else 'metadata_or_filesystem_error'
        print(json.dumps({'status':'v10_family_refused','reason':reason,'success':False}));raise SystemExit(2)
