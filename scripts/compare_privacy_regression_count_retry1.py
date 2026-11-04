"""Explicit count-only retry of the frozen 600-row regression; no family reselection."""
from __future__ import annotations
import argparse,json,os,sys,time,math
from pathlib import Path
from datetime import datetime,timezone
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from scripts import compare_privacy_family_v10_parallel as original
from scripts.compare_detector_models import known_classes
from scripts.compare_privacy_parallel import assemble_results
family=original.family
read=original.read
relative=original.relative
ordered_frame=original.ordered_frame
truth_rows=original.truth_rows
model_bindings=original.model_bindings
launch=original.launch
validate_worker_output=original.validate_worker_output
claim_attempt=original.claim_attempt
DEST=family.DEST
OUT=DEST/'parallel-regression-count-retry1'
PLAN=DEST/'regression-count-retry1-plan.json'
ATTEMPT=DEST/'regression-count-retry1-attempt.json'
PUBLIC=ROOT/'docs/reports/PRIVACY-REGRESSION-COUNT-RETRY1-PROTOCOL.json'
BOUNDARY=ROOT/'.cache/privacy-regression-count-failure-boundary.json'
BOUNDARY_SHA='9dbfa6f9fade4d3ff9ff9b394296ddc7a33da275b48eed266f79827e641ad027'
FAILED_SHA='2b4d57ab2e6ab23954a5b11349938e52ea4bd24c8e484c2361d5ed0e0426d3d5'
ORIGINAL_PLAN_SHA='38938b525f8525adf30897a9832e4492278f28a0f415de1de9207a0045d6986d'
BENCHMARK_SHA='03dbb7a8e60b0316e4c06508bcb76175097f594a29d0a25d40a2129998ae4c8a'

def source_pins():
    return dict(original.source_pins(),**{relative(Path(original.__file__)):sha256(Path(original.__file__)),relative(Path(__file__)):sha256(Path(__file__))})

def corrected_plan(value):
    if value.get('expected_samples')!=1400 or value.get('benchmark_sha256')!=BENCHMARK_SHA:raise ValueError('exact_original_wrong_count_plan_required')
    return dict(value,expected_samples=600)

def inputs():
    execution,value,frozen,parent,plan=original.admit('regression')
    if sha256(parent)!=ORIGINAL_PLAN_SHA or sha256(DEST/'regression-attempt.json')!=FAILED_SHA or sha256(BOUNDARY)!=BOUNDARY_SHA:raise ValueError('original_failed_count_evidence_changed')
    failed=read(DEST/'regression-attempt.json');boundary=read(BOUNDARY)
    if failed.get('status')!='failed' or failed.get('error_type')!='ValueError' or failed.get('automatic_retry_allowed') is not False or (DEST/'parallel-regression-v10').exists():raise ValueError('original_prelaunch_failure_required')
    if boundary.get('short_circuit_dataset_validator_calls')!=0 or boundary.get('TEST_image_hash_or_decode_calls_in_failed_phase')!=0 or boundary.get('model_worker_launches_in_failed_phase')!=0 or boundary.get('worker_directory_exists') is not False:raise ValueError('exact_no_pixel_prelaunch_failure_required')
    fresh=read(DEST/'fresh-attempt.json');fresh_result=read(DEST/'fresh-comparison.json')
    fresh_plan_sha=sha256(DEST/'fresh-plan.json')
    if fresh.get('status')!='completed' or fresh.get('samples')!=200 or fresh.get('plan_sha256')!=fresh_plan_sha or fresh_result.get('samples')!=200 or fresh_result.get('plan_sha256')!=fresh_plan_sha:raise ValueError('preserved_completed_fresh200_required')
    benchmark=workspace_path(plan['benchmark_manifest'],exists=True)
    if sha256(benchmark)!=BENCHMARK_SHA:raise ValueError('original_regression_manifest_changed')
    rows=read(benchmark)
    if len(rows)!=600 or any(r['split']!='test' for r in rows) or len({r['sample_id'] for r in rows})!=600 or len({r['sha256'] for r in rows})!=600:raise ValueError('exact_original600_unique_rows_required')
    if {label:sum(r.get('evaluation_class')==label for r in rows) for label in ('face','license_plate')}!={'face':300,'license_plate':300}:raise ValueError('original300_face300_plate_required')
    model_bindings(corrected_plan(plan),rows);truth_rows(rows,['face','license_plate']) # Metadata only; no images/models.
    return execution,value,frozen,parent,plan

def protocol_value():
    execution,value,frozen,parent,plan=inputs()
    return {'format':'privacy-regression-count-only-retry1-v1','source_sha256':source_pins(),'worker_source_sha256':sha256(Path(__file__)),
        'original_plan_sha256':ORIGINAL_PLAN_SHA,'original_failed_attempt_sha256':FAILED_SHA,'failure_boundary_sha256':BOUNDARY_SHA,
        'corrected_plan_sha256':sha256(PLAN),'original_parallel_protocol_sha256':sha256(original.PROTOCOL),
        'family_freeze_sha256':sha256(DEST/'winner-freeze.json'),'fresh_attempt_sha256':sha256(DEST/'fresh-attempt.json'),'fresh_comparison_sha256':sha256(DEST/'fresh-comparison.json'),
        'benchmark_sha256':BENCHMARK_SHA,'old_expected_samples':1400,'expected_samples':600,'face_samples':300,'plate_samples':300,
        'changed_plan_fields':['expected_samples'],'original600_rows_unchanged':True,'fresh_reexecuted':False,'model_or_threshold_selection_allowed':False,
        'runtime':execution['worker_runtime'],'direct_interpreter':'Python3.11 direct child, not venv launcher','worker_timeout_seconds':execution['worker_timeout_seconds'],
        'source_expires_at':value['source_expires_at'],'automatic_retry_allowed':False,'TEST_images_during_preparation':0}

def prepare():
    _,_,_,_,plan=inputs()
    if any(p.exists() for p in (PLAN,PUBLIC,ATTEMPT,OUT)):raise ValueError('new_count_retry_namespace_required')
    atomic_json(PLAN,corrected_plan(plan));atomic_json(PUBLIC,protocol_value())
    print(json.dumps({'status':'count600_correction_prepared_no_inference','protocol_sha256':sha256(PUBLIC),'TEST_images':0}))

def check_protocol():
    value=read(PUBLIC)
    if value!=protocol_value():raise ValueError('count_retry_protocol_or_source_changed')
    return value

def admit(role='regression'):
    if role!='regression':raise ValueError('regression_only_fresh_preserved')
    original_execution,value,frozen,parent,plan=inputs()
    execution=check_protocol();corrected=read(PLAN)
    if corrected!=corrected_plan(plan):raise ValueError('only_original_expected_count_may_change')
    return execution,value,frozen,PLAN,corrected

def assert_closure(execution):
    if sha256(workspace_path(execution['parent_plan'],exists=True))!=execution['parent_plan_sha256'] or sha256(DEST/'winner-freeze.json')!=execution['family_freeze_sha256']:raise ValueError('parent_plan_or_winner_changed')
    if sha256(PUBLIC)!=execution['parallel_protocol_sha256'] or source_pins()!=execution['source_sha256'] or sha256(Path(__file__))!=execution['worker_source_sha256']:raise ValueError('count_retry_execution_source_changed')
    if sha256(workspace_path(execution['benchmark_manifest'],exists=True))!=execution['benchmark_sha256']:raise ValueError('benchmark_changed')
    for item in execution['model_closure'].values():
        for ref in item.values():
            if sha256(workspace_path(ref['path'],exists=True))!=ref['sha256']:raise ValueError('model_or_source_artifact_changed')

def assert_worker_attempt(execution):
    if type(execution.get('parent_pid')) is not int or os.getppid()!=execution['parent_pid']:raise ValueError('active_direct_parent_worker_only')
    if execution.get('comparison_role')!='regression':raise ValueError('regression_only_worker')
    path=workspace_path(execution['attempt_path'],exists=True)
    if path!=workspace_path(ATTEMPT) or sha256(path)!=execution['attempt_sha256']:raise ValueError('same_active_count_retry_attempt_required')
    value=read(path)
    if value.get('status')!='started' or value.get('writer')!='two_process_CPU' or value.get('plan_sha256')!=execution['parent_plan_sha256'] or value.get('parallel_protocol_sha256')!=execution['parallel_protocol_sha256'] or value.get('selection_allowed') is not False or value.get('automatic_retry_allowed') is not False:raise ValueError('active_parallel_attempt_required')

def worker(plan_path,name,output,expected_sha):
    if sys.version_info[:2]!=(3,11):raise ValueError('direct_Python311_required')
    from agent.models import OnnxModel
    plan_path=workspace_path(plan_path,exists=True);output=workspace_path(output)
    if plan_path!=workspace_path(OUT/'execution-plan.json') or output.parent!=plan_path.parent or output.name!=name+'.json' or output.exists():raise ValueError('fresh_private_worker_output_required')
    if sha256(plan_path)!=expected_sha:raise ValueError('worker_execution_plan_changed')
    execution=read(plan_path);assert_worker_attempt(execution);assert_closure(execution);admit(execution['comparison_role'])
    if name not in ('baseline','candidate'):raise ValueError('known_model_worker_required')
    manifest=workspace_path(execution['models'][name],exists=True);model=OnnxModel(manifest,cpu_threads=2,allow_spinning=True);cv2.setNumThreads(1)
    try:
        options=model.net.session.get_session_options()
        if model.backend!='onnxruntime-cpu' or model.backend_version!='1.31.1' or model.net.session.get_providers()!=['CPUExecutionProvider'] or options.intra_op_num_threads!=2 or options.inter_op_num_threads!=1 or cv2.getNumThreads()!=1:raise ValueError('exact_CPU131_intra2_inter1_CV1_required')
        artifact=dict(model.manifest);model.manifest=dict(artifact,confidence=.01)
        rows=sorted((r for r in read(workspace_path(execution['benchmark_manifest'],exists=True)) if r['split']=='test'),key=lambda r:r['sample_id'])
        predictions=[];timings=[];counts=[];order=[]
        for row in rows:
            if row.get('ignore_bboxes_xywh_px'):raise ValueError('same_serial_explicit_ignore_protocol_required')
            if datetime.fromisoformat(row['expires_at'].replace('Z','+00:00'))<=datetime.now(timezone.utc):raise ValueError('original_source_expired_before_pixel_read')
            image=cv2.imread(str(workspace_path(row['path'],exists=True)));known=known_classes(row,artifact['classes'])
            start=time.perf_counter();found=model.predict(image);timings.append(time.perf_counter()-start)
            selected=ordered_frame(found,row['sample_id'],known);predictions.extend(selected);counts.append(len(selected));order.append(row['sample_id'])
        assert_closure(execution)
        if sha256(plan_path)!=expected_sha:raise ValueError('execution_plan_changed_during_worker')
        atomic_json(output,{'synthetic_contract':False,'name':name,'execution_plan_sha256':expected_sha,
            'model_manifest_sha256':execution['execution_models'][name]['manifest_sha256'],'weights_sha256':artifact['sha256'],
            'benchmark_sha256':execution['benchmark_sha256'],'sample_order':order,'prediction_counts':counts,'predictions':predictions,'timings':timings,
            'images':len(rows),'backend':model.backend,'version':model.backend_version,'providers':model.net.session.get_providers(),'intra_threads':2,'inter_threads':1,'CV_threads':1})
    finally:model.close()

def run(authorized=False):
    role='regression'
    if authorized is not True:raise ValueError('explicit_root_final_comparison_authorization_required')
    if sys.version_info[:2]!=(3,11):raise ValueError('direct_Python311_required')
    execution_protocol,_,frozen,parent,plan=admit(role)
    output=workspace_path(OUT/'comparison.json')
    if output.exists():raise ValueError('final_comparison_already_exists')
    attempt=workspace_path(ATTEMPT);parent_sha=sha256(parent);protocol_sha=sha256(PUBLIC)
    if read(parent)!=plan or check_protocol()!=execution_protocol:raise ValueError('admission_snapshot_changed')
    claim_attempt(attempt,{'status':'started','plan_sha256':parent_sha,'parallel_protocol_sha256':protocol_sha,'writer':'two_process_CPU','selection_allowed':False,'automatic_retry_allowed':False})
    try:
        from agent.dataset import validate_research_release
        benchmark=workspace_path(plan['benchmark_manifest'],exists=True)
        rows=sorted((r for r in read(benchmark) if r['split']=='test'),key=lambda r:r['sample_id'])
        if len(rows)!=plan['expected_samples'] or not validate_research_release(rows)['valid']:raise ValueError('same_frozen_benchmark_and_count_required')
        metadata,closure,classes=model_bindings(plan,rows);truth=truth_rows(rows,classes)
        directory=workspace_path(OUT);directory.mkdir(exist_ok=False)
        atomic_json(directory/'source.json',{'expires_at':min(r['expires_at'] for r in rows),'source_expiry_extended':False,'raw_photos_copied':False})
        execution=dict(plan,parent_plan=relative(parent),parent_plan_sha256=parent_sha,family_freeze_sha256=sha256(family.DEST/'winner-freeze.json'),
                       parallel_protocol_sha256=protocol_sha,worker_source_sha256=execution_protocol['worker_source_sha256'],source_sha256=execution_protocol['source_sha256'],execution_models=metadata,model_closure=closure,
                       parent_pid=os.getpid(),comparison_role=role,attempt_path=relative(attempt),attempt_sha256=sha256(attempt))
        execution_path=directory/'execution-plan.json';atomic_json(execution_path,execution);execution_sha=sha256(execution_path)
        assert_closure(execution)
        admit(role)  # Close the original exact plan, winner and numeric gates too.
        commands={name:[sys.executable,str(Path(__file__)),'worker','--plan',str(execution_path),'--plan-sha',execution_sha,'--name',name,'--output',str(directory/(name+'.json'))] for name in ('baseline','candidate')}
        start=time.perf_counter();launch(commands,directory,execution_protocol['worker_timeout_seconds']);elapsed=time.perf_counter()-start
        assert_closure(execution)
        admit(role)
        if sha256(execution_path)!=execution_sha:raise ValueError('execution_snapshot_changed')
        found={name:read(directory/(name+'.json')) for name in metadata};order=[row['sample_id'] for row in rows]
        for name,value in found.items():validate_worker_output(value,name,execution,execution_sha,order)
        results=assemble_results(truth,{name:v['predictions'] for name,v in found.items()},classes,metadata,{name:v['timings'] for name,v in found.items()})
        report={'count_correction':{'original_expected':1400,'actual_same_manifest':600,'original_plan_sha256':ORIGINAL_PLAN_SHA,'original_failed_attempt_sha256':FAILED_SHA,'protocol_sha256':protocol_sha,'fresh_reexecuted':False,'cohort_or_model_selection_changed':False},'scope':plan['role'],'evaluated_at':datetime.now(timezone.utc).isoformat(),'benchmark_sha256':sha256(benchmark),'plan_sha256':parent_sha,'samples':len(rows),'results':results,
            'map50_change':results['candidate']['ap_metrics']['map50']-results['baseline']['ap_metrics']['map50'],
            'known_class_conditional':any(r.get('known_label_coverage') or r.get('evaluation_class') for r in rows),'test_used_for_selection':False,
            'independent_physical_identity_or_route_verified':False,'project_acceptance_passed':False,
            'limits':['Publisher-known classes only; unknown other classes excluded from negatives','Single-model CPU timing under shared host load, not field/full-pipeline FPS','Small source proxy groups do not prove physical independence',
                      'Two concurrent CPU2/CV1 model processes; latency includes contention, not isolated serial speed or FPS promotion']}
        if output.exists():raise ValueError('result_written_by_other_execution')
        atomic_json(output,report);atomic_json(output.with_name(output.stem+'-per-image.json'),{'truth':truth,'predictions':{name:v['predictions'] for name,v in found.items()}})
        atomic_json(directory/'execution-receipt.json',{'original_plan_sha256':ORIGINAL_PLAN_SHA,'original_failed_attempt_sha256':FAILED_SHA,'worker_source_sha256':sha256(Path(__file__)),'status':'completed','parent_plan_sha256':parent_sha,'execution_plan_sha256':execution_sha,'parallel_protocol_sha256':protocol_sha,
                    'wall_seconds':elapsed,'wall_scope':'worker launch-to-join only, excluding admission and final scoring/serialization','model_processes':2,'model_or_threshold_selection_allowed':False,'timing_is_shared_parallel_context':True,'GPU_operations':0})
        atomic_json(attempt,{'status':'completed','plan_sha256':parent_sha,'parallel_protocol_sha256':protocol_sha,'writer':'two_process_CPU','samples':len(rows),'selection_allowed':False,'automatic_retry_allowed':False})
        print(json.dumps({'status':'frozen_comparison_completed','role':role,'samples':len(rows),'wall_seconds':elapsed,'math_unchanged':True}))
    except BaseException as error:
        atomic_json(attempt,{'status':'failed','error_type':type(error).__name__,'plan_sha256':parent_sha,'parallel_protocol_sha256':protocol_sha,'writer':'two_process_CPU','selection_allowed':False,'automatic_retry_allowed':False});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','worker']);p.add_argument('--authorized',action='store_true');p.add_argument('--plan');p.add_argument('--plan-sha');p.add_argument('--name',choices=['baseline','candidate']);p.add_argument('--output');a=p.parse_args()
    try:
        if a.action=='prepare':prepare()
        elif a.action=='run':run(a.authorized)
        else:worker(a.plan,a.name,a.output,a.plan_sha)
    except Exception as e:print(json.dumps({'status':'count_retry_stopped','error_type':type(e).__name__,'reason':str(e) if isinstance(e,ValueError) and str(e).replace('_','').isalnum() else 'metadata_or_execution_error'}));raise SystemExit(1)
