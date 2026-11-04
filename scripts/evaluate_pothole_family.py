"""VAL-only family freeze, then proof-gated one-shot US100/regression comparisons.

No training, calibration or image reads occur in policy/family/plan preparation.
Only the explicit run action calls the existing actual-app detector evaluator.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256
from agent.material_policy import source_deadline
from scripts.evaluate_public_refinement import strict_proof
from scripts.compare_detector_models import compare

BASE='data/training/pothole-mwpd-v6';TINY='data/training/pothole-tiny-v7'
NANO='data/training/pothole-localization-v8';ORIGINAL_SMALL='data/training/pothole-small-v9'
SMALL='data/training/pothole-small-v9-retry1';DEST='data/training/pothole-family-v9'
POLICY='docs/reports/POTHOLE-FAMILY-V9-ONE-SHOT-PLAN.json'
US100='data/training/research-pothole-rdd2022-us100-v7/release.json'
US100_SHA='592c83e5967073354031ade9b50eb0a24b7b2f65f33abf477fc3990611221703'
REGRESSION=BASE+'/final-benchmark.json'
REGRESSION_SHA='7497babaf45a3bcd652b5f0cf6a5807f5f0da1e636878475e6272792c999f415'
PARENTS={'fresh':'docs/reports/RDD2022-US100-BLIND-V7-PREBUILD-REVISION.json','regression':'docs/reports/POTHOLE-MWPD-FINAL-PLAN.json'}
US100_ORIGINAL_PLAN='docs/reports/RDD2022-US100-BLIND-V7-PLAN.json'
US100_ACQUISITION='docs/reports/RDD2022-US100-BLIND-V7-ACQUISITION.json'
US100_PROTOCOL='data/research/rdd2022-blind-v7/us100-offline-protocol-v2.json'
RETRY_PUBLIC='docs/reports/POTHOLE-SMALL-V9-RETRY1-PLAN.json'
RETRY_PUBLIC_SHA='805609357c979d3b37c1354127e4c8069486c0be0434e46e33978cf78b904ceb'
METRICS=('old_validation_ap50','new_validation_ap50')
PROOFS=('verification-final.json','verification-mwpd-positive-1.json','verification-mwpd-positive-2.json')
EVAL_SOURCES=('scripts/evaluate_pothole_family.py','scripts/compare_detector_models.py','scripts/evaluate_public_refinement.py',
    'scripts/verify_trained_models.py','agent/models.py','agent/evaluate.py','agent/dataset.py')


def read(path):
    path=workspace_path(path,exists=True)
    if not path.is_file() or path.stat().st_size>40_000_000:raise ValueError('bounded_metadata_required')
    return json.loads(path.read_text(encoding='utf-8'))


def pointer(path):
    path=workspace_path(path,exists=True);return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path)}


def pinned(ref):
    path=workspace_path(ref['path'],exists=True)
    if sha256(path)!=ref['sha256']:raise ValueError('pinned_metadata_changed')
    return read(path)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_new(path,value):
    path=workspace_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,indent=2,allow_nan=False)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())


def metric(row):
    values=tuple(row[key] for key in METRICS)
    if any(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1 for value in values):raise ValueError('finite_unit_validation_metrics_required')
    return values


def detector_contract(artifact):
    if artifact.get('format')!='onnx' or artifact.get('task')!='pothole' or artifact.get('classes')!=['pothole'] or artifact.get('research_only') is not True or artifact.get('production_approved') is not False or artifact.get('confidence')!=.1 or artifact.get('nms_iou',.5)!=.5:raise ValueError('unchanged_research_pothole_detector_contract_required')


def best_epoch(history):
    if not isinstance(history,list) or not history:raise ValueError('completed_validation_history_required')
    for index,row in enumerate(history):
        if type(row.get('epoch')) is not int or row['epoch']!=index:raise ValueError('ordered_validation_history_required')
        metric(row)
    eligible=[row for row in history if metric(row)[0]>=metric(history[0])[0]-.03]
    return max(eligible,key=lambda row:(sum(metric(row))/2,-row['epoch']))


def choose_family(members):
    if set(members)!={'baseline','tiny','nano','small'} or members['baseline']['status']!='available' or members['tiny']['status']!='rejected_no_artifact':raise ValueError('complete_predeclared_family_required')
    reference=metric(members['baseline']['validation'])[0];eligible=[];audit={}
    for rank,name in enumerate(('baseline','nano','small')):
        member=members[name]
        if member['status']=='rejected_no_artifact':audit[name]={'eligible':False,'reason':'no_selected_artifact'};continue
        if member['status']!='available':raise ValueError('all_family_training_must_complete')
        old,new=metric(member['validation']);ok=old>=reference-.03
        audit[name]={'eligible':ok,'old_validation_ap50':old,'new_validation_ap50':new,'mean_validation_ap50':(old+new)/2}
        if ok:eligible.append(((old+new)/2,-rank,name))
    return {'selected_member':max(eligible)[2],'validation_only':True,'old_validation_drop_limit':.03,'tie_order':['baseline','nano','small'],'members':audit}


def protocol():
    initial=read(US100_ORIGINAL_PLAN);revision=read(PARENTS['fresh']);acquired=read(US100_ACQUISITION)
    if initial['plan'].get('model_inferences_before_freeze')!=0 or initial['plan'].get('train_or_selection_use') is not False or revision['plan'].get('model_inferences_before_freeze')!=0 or revision['plan'].get('train_or_selection_use') is not False or revision['plan']['previous_pre_build_protocol_sha256']!=initial['protocol_sha256'] or acquired.get('model_inferences')!=0 or acquired.get('status')!='frozen' or acquired.get('release_sha256')!=US100_SHA or acquired.get('protocol_sha256')!=revision['protocol_sha256'] or sha256(US100_PROTOCOL)!=revision['protocol_sha256']:raise ValueError('US100_prebuild_revision_or_zero_inference_binding_required')
    return {'format':'pothole-family-v9-predeclared-evaluation-v1','selection':'oldVAL>=frozen-v6-.03; maximum mean(oldVAL,newVAL); exact tie baseline then Nano then Small',
        'candidate_members':{'baseline':BASE,'tiny_rejected':TINY,'nano':NANO,'small_retry_only':SMALL},
        'original_small_failed_experiment':ORIGINAL_SMALL,'retry_protocol':{'path':RETRY_PUBLIC,'sha256':RETRY_PUBLIC_SHA},
        'cohorts':{'fresh':{'path':US100,'sha256':US100_SHA,'samples':100,'positive':50,'negative':50},
                   'regression':{'path':REGRESSION,'sha256':REGRESSION_SHA,'samples':145,'positive':97,'negative':48}},
        'benchmark_parent_plans':{role:pointer(path) for role,path in PARENTS.items()},
        'US100_original_plan':pointer(US100_ORIGINAL_PLAN),'US100_acquisition_before_inference':pointer(US100_ACQUISITION),'US100_actual_protocol':pointer(US100_PROTOCOL),
        'AP_floor':.01,'operating_confidence':.1,'fixed_reference_confidence':.35,'nms_iou':.5,
        'new_candidate_TEST_inference_before_family_freeze':False,'TEST_selection_allowed':False,
        'one_permanent_attempt_per_cohort':True,'user_recordings_used':False,'physical_route_independence_verified':False,'production_approved':False}


def prepare_policy():
    value=protocol();write_new(POLICY,value);return {'policy_sha256':sha256(POLICY),'image_reads':0,'TEST_inferences':0}


def validate_policy():
    value=read(POLICY)
    if value!=protocol():raise ValueError('predeclared_family_policy_or_parent_plan_changed')
    return pointer(POLICY)


def retry_contract():
    if sha256(RETRY_PUBLIC)!=RETRY_PUBLIC_SHA:raise ValueError('frozen_retry_protocol_changed')
    spec=read(RETRY_PUBLIC)
    if read(SMALL+'/retry-plan.json')!=spec or spec.get('format')!='pothole-small-v9-sequential-full-restart-v1' or spec.get('restart_from_official_initialization') is not True or any(spec.get(key) is not False for key in ('optimizer_state_resumed','prior_partial_weights_used','learning_recipe_changed','TEST_used')):raise ValueError('unchanged_full_restart_protocol_required')
    copied=spec.get('copied_input_sha256',{})
    if set(copied)!={'plan.json','release.json','preflight.json','GPU-memory-preflight.json'}:raise ValueError('four_exact_copied_inputs_required')
    for name,pin in copied.items():
        if sha256(ORIGINAL_SMALL+'/'+name)!=pin or sha256(SMALL+'/'+name)!=pin:raise ValueError('retry_copied_inputs_changed')
    for path,pin in spec['stopped_parent_receipts'].items():
        if sha256(path)!=pin:raise ValueError('stopped_parent_receipt_changed')
        prior=read(path)
        if prior.get('own_child_running_after_cleanup') is not False or prior.get('sources_unchanged') is not True:raise ValueError('parent_writers_not_stopped_unchanged')
    failed=read(ORIGINAL_SMALL+'/shared-training-supervisor.json')
    if failed.get('format')!='small-v9-shared-supervision-v1' or failed.get('status')!='failed' or failed.get('own_child_stopped') is not True or failed.get('device_memory',{}).get('over_budget_observed') is not True:raise ValueError('original_small_failed_run_evidence_required')
    complete=read(SMALL+'/shared-training-supervisor.json')
    if complete.get('format')!='small-v9-sequential-restart-supervision-v1' or complete.get('status')!='completed' or type(complete.get('child_exit_code')) is not int or complete['child_exit_code']!=0 or complete.get('sources_unchanged') is not True or complete.get('own_child_running_after_cleanup') is not False or complete.get('training_math_changed') is not False or complete.get('device_memory',{}).get('over_budget_observed') is not False:raise ValueError('completed_retry_supervision_required_before_family_freeze')
    if complete.get('plan_sha256')!=copied['plan.json'] or complete.get('supervisor_sha256')!=spec['supervisor_sha256'] or sha256('scripts/train_pothole_small.py')!=spec['learning_source_sha256']:raise ValueError('retry_execution_source_or_plan_binding_changed')
    return {'protocol':pointer(RETRY_PUBLIC),'local_protocol':pointer(SMALL+'/retry-plan.json'),
        'completed_supervisor':pointer(SMALL+'/shared-training-supervisor.json'),'original_failed_supervisor':pointer(ORIGINAL_SMALL+'/shared-training-supervisor.json'),
        'copied_input_sha256':copied,'original_failed_selection_freeze_present':workspace_path(ORIGINAL_SMALL+'/selection-freeze.json').exists(),'original_failed_artifact_excluded':True}


def baseline_member():
    snap=read(BASE+'/selection-freeze.json');artifact=read(BASE+'/model.json')
    detector_contract(artifact)
    if snap.get('test_inference_performed') is not False:raise ValueError('baseline_validation_snapshot_not_preTEST')
    if sha256(BASE+'/model.json')!=snap['manifest_sha256'] or sha256(BASE+'/best.pt')!=snap['checkpoint_sha256'] or artifact['sha256']!=snap['onnx_sha256'] or sha256(artifact['weights'])!=artifact['sha256']:raise ValueError('baseline_artifact_changed')
    metric(snap['selected_validation'])
    return {'status':'available','validation':snap['selected_validation'],'selection_freeze':pointer(BASE+'/selection-freeze.json'),
        'manifest':pointer(BASE+'/model.json'),'checkpoint':pointer(BASE+'/best.pt'),'weights_sha256':artifact['sha256']}


def tiny_rejected():
    snap=read(TINY+'/selection-freeze.json');history=read(TINY+'/history.json')
    if snap.get('candidate_selected') is not False or snap.get('test_used') is not False or history.get('selected',{}).get('epoch')!=0 or best_epoch(history['history'])['epoch']!=0 or any(workspace_path(TINY+'/'+name).exists() for name in ('model.json','model-development.json')):raise ValueError('tiny_rejected_zero_no_artifact_required')
    return {'status':'rejected_no_artifact','selected_epoch':0,'TEST_used':False,
        'selection_freeze':pointer(TINY+'/selection-freeze.json'),'history':pointer(TINY+'/history.json')}


def trained_member(folder,*,packaged=False):
    snap=read(folder+'/selection-freeze.json');history=read(folder+'/history.json');plan=read(folder+'/plan.json')
    if snap.get('TEST_inference_performed') is not False:raise ValueError('new_candidate_TEST_must_be_zero_before_family_freeze')
    epoch=snap.get('epochs_completed')
    if type(epoch) is not int or not plan['minimum_epochs']<=epoch<=plan['maximum_epochs'] or len(history['history'])!=epoch+1:raise ValueError('completed_training_snapshot_required')
    selected=best_epoch(history['history'])
    if snap.get('selected')!=selected or history.get('selected')!=selected:raise ValueError('selected_checkpoint_not_validation_best')
    rows=read(folder+'/release.json')
    if not rows or {row['split'] for row in rows}!={'train','validation'}:raise ValueError('development_release_must_have_no_TEST')
    for row in rows:source_deadline(row['expires_at'],required=True)
    if plan['development_sha256']!=sha256(folder+'/release.json'):raise ValueError('frozen_development_release_changed')
    selected_flag='candidate_improved_on_VAL' if packaged else 'candidate_selected'
    if snap.get(selected_flag) is not (selected['epoch']>0):raise ValueError('selected_epoch_artifact_status_mismatch')
    result={'status':'available' if selected['epoch']>0 else 'rejected_no_artifact','validation':selected,
        'selection_freeze':pointer(folder+'/selection-freeze.json'),'history':pointer(folder+'/history.json'),'training_plan':pointer(folder+'/plan.json'),
        'development_release':pointer(folder+'/release.json'),'TEST_inference_performed':False}
    if selected['epoch']==0:return result
    parent=read(folder+'/model.json');manifest_path=folder+('/model-development.json' if packaged else '/model.json');artifact=read(manifest_path)
    detector_contract(artifact)
    if sha256(folder+'/model.json')!=snap['manifest_sha256'] or sha256(folder+'/best.pt')!=snap['checkpoint_sha256'] or artifact['sha256']!=snap['weights_sha256'] or sha256(artifact['weights'])!=artifact['sha256']:raise ValueError('selected_artifacts_changed')
    if packaged:
        expected=parent|{'development_only_release':True,'executed_export_manifest_sha256':sha256(folder+'/model.json'),
            'selection_freeze_sha256':sha256(folder+'/selection-freeze.json'),'packaging_source_sha256':sha256('scripts/package_development_export.py')}
        if artifact!=expected:raise ValueError('development_packaging_changed_model_contract')
    if artifact.get('development_only_release') is not True or artifact.get('exported_before_final_test') is not True or artifact.get('test_split_used') is not False or artifact.get('best_epoch')!=selected['epoch'] or artifact.get('trained_checkpoint_sha256')!=snap['checkpoint_sha256'] or artifact.get('refinement_plan_sha256')!=sha256(folder+'/plan.json'):raise ValueError('preTEST_selected_export_binding_required')
    if artifact['source_release_path']!=folder+'/release.json' or artifact['dataset_release']!='manifest-sha256:'+sha256(folder+'/release.json') or artifact.get('confidence')!=.1 or artifact.get('nms_iou',.5)!=.5:raise ValueError('frozen_source_or_threshold_contract_changed')
    result.update(manifest=pointer(manifest_path),checkpoint=pointer(folder+'/best.pt'),weights_sha256=artifact['sha256'])
    return result


def family_core():
    policy=validate_policy();retry=retry_contract()
    members={'baseline':baseline_member(),'tiny':tiny_rejected(),'nano':trained_member(NANO,packaged=True),'small':trained_member(SMALL)}
    return {'format':'pothole-family-v9-validation-freeze-v1','parent_policy':policy,'members':members,'retry_contract':retry,
        'selection':choose_family(members),'selection_source_sha256':sha256(Path(__file__)),
        'TEST_or_image_reads_during_selection':0,'post_TEST_reselection_allowed':False}


def freeze_family():
    path=DEST+'/family-freeze.json'
    if workspace_path(path).exists():raise ValueError('family_already_frozen_no_reselection')
    value=family_core();value['family_digest']=digest(value);write_new(path,value)
    return {'family_digest':value['family_digest'],'selected_member':value['selection']['selected_member'],'image_reads':0,'TEST_inferences':0}


def validated_family():
    value=read(DEST+'/family-freeze.json');core={key:item for key,item in value.items() if key!='family_digest'}
    if value.get('family_digest')!=digest(core) or core!=family_core():raise ValueError('frozen_family_digest_or_members_changed')
    return value


def proof_refs(family):
    cases_plan=pointer(BASE+'/canonical-plan.json');cases=pinned(cases_plan)['cases']
    if len(cases)!=3:raise ValueError('three_original_positive_VAL_cases_required')
    result={}
    for role,name in [('baseline','baseline'),('candidate',family['selection']['selected_member'])]:
        member=family['members'][name];manifest=workspace_path(member['manifest']['path']);checkpoint_sha=member['checkpoint']['sha256']
        source=read(read(manifest)['source_release_path']);val={row['sha256'] for row in source if row['split']=='validation' and any(obj['class']=='pothole' for obj in row['objects'])}
        refs=[]
        for case,filename in zip(cases,PROOFS):
            if case['sha256'] not in val:raise ValueError('canonical_case_not_positive_validation')
            path=manifest.parent/filename
            strict_proof(manifest,path,image_sha=case['sha256'],checkpoint_sha=checkpoint_sha,positive=True)
            report=read(path)
            if report.get('app_backend')!='onnxruntime-cpu' or report.get('app_backend_version')!='1.31.1' or any(report.get(key) is not True for key in ('metadata_valid','weight_hash_verified','preprocess_contract_verified','finite_output')) or any(report.get('detection_postprocess_parity',{}).get(mode,{}).get('passed') is not True for mode in ('ort','torch')):raise ValueError('actual_default_raw_and_detection_post_proof_required')
            refs.append(pointer(path))
        result[role]=refs
    return cases_plan,result


def comparison_plan(family,role):
    if role not in PARENTS:raise ValueError('unknown_comparison_role')
    policy=pinned(family['parent_policy']);cohort=policy['cohorts'][role]
    if sha256(cohort['path'])!=cohort['sha256']:raise ValueError('fixed_benchmark_metadata_changed')
    rows=read(cohort['path'])
    if len(rows)!=cohort['samples'] or any(row['split']!='test' or any(obj['class']!='pothole' for obj in row['objects']) for row in rows) or sum(bool(row['objects']) for row in rows)!=cohort['positive'] or len({row['sha256'] for row in rows})!=len(rows) or len({row['sample_id'] for row in rows})!=len(rows):raise ValueError('fixed_TEST_cohort_contract_changed')
    for row in rows:source_deadline(row['expires_at'],required=True)
    cases,proofs=proof_refs(family);selected=family['members'][family['selection']['selected_member']];base=family['members']['baseline']
    artifact=pinned(selected['manifest'])
    return {'format':'pothole-family-one-shot-comparison-v1','role':'US100_new_blind_source_proxy' if role=='fresh' else 'MWPD97_RDD48_historical_regression_only',
        'models':{'baseline':base['manifest']['path'],'candidate':selected['manifest']['path']},'baseline_manifest_sha256':base['manifest']['sha256'],
        'candidate_release_sha256':artifact['dataset_release'].removeprefix('manifest-sha256:'),'benchmark_manifest':cohort['path'],'benchmark_sha256':cohort['sha256'],
        'family_freeze':pointer(DEST+'/family-freeze.json'),'family_digest':family['family_digest'],'parent_policy':family['parent_policy'],
        'parent_benchmark_plan':policy['benchmark_parent_plans'][role],'positive_VAL_case_plan':cases,'strict_proofs':proofs,
        'evaluation_sources':{path:sha256(path) for path in EVAL_SOURCES},'TEST_used_for_selection':False,'automatic_retry_allowed':False}


def prepare_comparison_plans():
    family=validated_family();plans={role:comparison_plan(family,role) for role in PARENTS}
    for role in plans:
        if workspace_path(DEST+'/'+role+'-plan.json').exists():raise ValueError('comparison_plans_already_prepared')
    for role,value in plans.items():write_new(DEST+'/'+role+'-plan.json',value)
    return {'plans':2,'family_digest':family['family_digest'],'image_reads':0,'TEST_inferences':0}


def run(role):
    if role not in PARENTS:raise ValueError('unknown_comparison_role')
    family=validated_family();path=DEST+'/'+role+'-plan.json';plan=read(path)
    if plan!=comparison_plan(family,role):raise ValueError('comparison_family_parent_or_proofs_changed')
    output=workspace_path(DEST+'/'+role+'-comparison.json');attempt=workspace_path(DEST+'/'+role+'-attempt.json')
    if output.exists() or attempt.exists():raise ValueError('one_shot_already_started_no_automatic_retry')
    value={'status':'started','family_digest':family['family_digest'],'plan_sha256':sha256(path),'parent_policy_sha256':family['parent_policy']['sha256'],'selection_allowed':False}
    write_new(attempt,value)  # Exclusive durable marker remains even on a failed first inference.
    result=compare(path,output)
    write_new(DEST+'/'+role+'-completion.json',value|{'status':'completed','samples':result['samples'],'comparison_sha256':sha256(output)})
    return {'role':role,'samples':result['samples'],'family_digest':family['family_digest'],'TEST_used_for_selection':False}


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
        print(json.dumps({'status':'family_evaluation_refused','reason':reason,'success':False}))
        raise SystemExit(2)
