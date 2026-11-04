"""Two owned CPU model processes; frozen family math and one-shot barrier.

Only scheduling changes. No model/data sharding, new thresholds, or automatic
retry. Protocol preparation and synthetic contract workers never open TEST data.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,math
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
from scripts import evaluate_privacy_family as family
from scripts.compare_detector_models import known_classes
from scripts.compare_privacy_parallel import assemble_results

PROTOCOL=ROOT/'docs/reports/PRIVACY-FAMILY-V8-PARALLEL-EXECUTION-PROTOCOL.json'
FAMILY_SHA='5777de565111135f3793d247a332d5bb4c5cae4635378f8cfacfcbbff3cb4f59'
FAMILY_SOURCE_SHA='95b5dc346e30f0ee8341afdcdc874754507ce4c937f55261bdcb71b15779ca10'
SOURCES=['agent/models.py','agent/evaluate.py','agent/pipeline.py','scripts/compare_detector_models.py',
         'scripts/compare_privacy_parallel.py','scripts/evaluate_privacy_family.py','scripts/evaluate_public_refinement.py']

def read(path):return json.loads(workspace_path(path,exists=True).read_text(encoding='utf-8'))
def relative(path):return path.relative_to(ROOT).as_posix()

def source_pins():return {name:sha256(ROOT/name) for name in SOURCES}

def protocol_value():
    value=family.protocol()
    if sha256(family.DEST/'protocol.json')!=FAMILY_SHA or sha256(ROOT/'scripts/evaluate_privacy_family.py')!=FAMILY_SOURCE_SHA:raise ValueError('original_family_source_and_protocol_required')
    result={'format':'privacy-family-v8-two-process-execution-v1','created_before_final_TEST_predictions':True,
        'original_family_protocol_sha256':FAMILY_SHA,'original_family_source_sha256':FAMILY_SOURCE_SHA,
        'worker_source_sha256':sha256(Path(__file__)),'source_sha256':source_pins(),
        'parent_benchmarks':value['benchmarks'],'original_trainer_plan_sha256':value['trainer_plan_sha256'],
        'execution':'exactly two independent owned CPU processes, one baseline and one candidate, each processing the entire same sorted cohort',
        'worker_runtime':{'engine':'onnxruntime-cpu','version':'1.31.1','providers':['CPUExecutionProvider'],'intra_threads':2,'inter_threads':1,'OpenCV_threads':1,'allow_spinning':True},
        'order':'sorted sample_id; publisher object order for truth; original OnnxModel per-image prediction order; no global prediction reorder or TopK',
        'math':'same OnnxModel.predict floor.01/NMS/preprocessing and imported canonical score for AP/operating/fixed.35',
        'admission':'original family protocol + frozen winner + exact original comparison_plan + numeric_gates before TEST labels/pixels',
        'shared_attempt':'same original {role}-attempt.json exclusive create, flush and fsync before TEST decode; serial and parallel mutually exclude',
        'worker_ownership':'direct child of active parent only; exact started-attempt SHA and parent plan required; stale/failed direct worker replay rejected',
        'closure':'parent plan/family freeze/worker/source/model/weights/provenance/release/benchmark SHA checked at start and finish',
        'failure':'stop only owned children; failed attempt and partial private outputs preserved; no automatic retry',
        'result_format':'same canonical comparison keys and per-model fields; parallel walltime recorded separately; additional limits disclose contention',
        'timing':'shared-host two-process contention; scheduling change only, not isolated model speed, FPS or field acceptance',
        'wall_clock_scope':'worker launch-to-join only; admission/hash checks/final metric assembly and serialization are excluded',
        'worker_timeout_seconds':7200,'TEST_predictions_during_preparation':0,'GPU_operations':0,'training_or_selection_changed':False,
        'source_expires_at':value['source_expires_at']}
    return result

def prepare():
    result=protocol_value()
    if PROTOCOL.exists():raise ValueError('parallel_protocol_already_frozen')
    atomic_json(PROTOCOL,result)
    print(json.dumps({'status':'parallel_protocol_prepared_no_inference','protocol_sha256':sha256(PROTOCOL),'worker_source_sha256':result['worker_source_sha256'],'TEST_predictions':0}))

def check_protocol():
    value=read(PROTOCOL)
    if value!=protocol_value():raise ValueError('immutable_execution_protocol_required')
    if value['original_family_protocol_sha256']!=FAMILY_SHA or sha256(family.DEST/'protocol.json')!=FAMILY_SHA or sha256(ROOT/'scripts/evaluate_privacy_family.py')!=FAMILY_SOURCE_SHA:raise ValueError('original_family_binding_changed')
    if value['worker_source_sha256']!=sha256(Path(__file__)) or value['source_sha256']!=source_pins():raise ValueError('frozen_worker_or_math_source_changed')
    return value

def admit(role):
    if role not in ('fresh','regression'):raise ValueError('known_comparison_role_required')
    execution=check_protocol();value=family.protocol();frozen=family.read(family.DEST/'winner-freeze.json')
    if frozen['protocol_sha256']!=FAMILY_SHA:raise ValueError('frozen_winner_protocol_mismatch')
    parent=family.DEST/(role+'-plan.json');plan=family.read(parent)
    if plan!=family.comparison_plan(value,frozen,role):raise ValueError('exact_original_comparison_plan_required')
    if plan['family_freeze_sha256']!=sha256(family.DEST/'winner-freeze.json'):raise ValueError('winner_freeze_changed')
    if execution['parent_benchmarks']!=value['benchmarks'] or plan['benchmark_sha256']!=execution['parent_benchmarks'][role]['sha256']:raise ValueError('opaque_benchmark_binding_changed')
    family.numeric_gates(frozen)  # Mandatory, before claim or TEST labels/pixels.
    return execution,value,frozen,parent,plan

def claim_attempt(path,value):
    path=workspace_path(path)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())

def ordered_frame(found,sample_id,known):
    return [dict(item,sample_id=sample_id) for item in found if item['class'] in known]

def truth_rows(rows,classes):
    result=[]
    for row in rows:
        if row.get('ignore_bboxes_xywh_px'):raise ValueError('same_serial_explicit_ignore_protocol_required')
        known=known_classes(row,classes)
        result.extend({'sample_id':row['sample_id'],'class':o['class'],'bbox_xywh_px':o['bbox_xywh_px']} for o in row['objects'] if o['class'] in known)
    return result

def model_bindings(plan,rows):
    hashes={r['sha256'] for r in rows};groups={r['leakage_group_id'] for r in rows};metadata={};closure={};classes=None
    for name,reference in plan['models'].items():
        path=workspace_path(reference,exists=True);info=read(path)
        if info.get('task')!='redaction' or info.get('research_only') is not True or info.get('production_approved') is not False:raise ValueError('same_serial_research_redaction_only')
        if classes is None:classes=info['classes']
        elif classes!=info['classes']:raise ValueError('same_serial_incomparable_classes')
        if classes!=['face','license_plate']:raise ValueError('privacy_two_class_contract_required')
        source=workspace_path(info['source_release_path'],exists=True);source_sha=sha256(source)
        if info['dataset_release']!='manifest-sha256:'+source_sha:raise ValueError('source_release_changed')
        development=[r for r in read(source) if r['split'] in ('train','validation')]
        if hashes&{r['sha256'] for r in development} or groups&{r['leakage_group_id'] for r in development}:raise ValueError('same_serial_holdout_development_leakage')
        if name=='baseline' and sha256(path)!=plan['baseline_manifest_sha256']:raise ValueError('baseline_changed')
        if name=='candidate' and source_sha!=plan['candidate_release_sha256']:raise ValueError('candidate_release_changed')
        confidence=info.get('confidence',.35)
        if type(confidence) not in (int,float) or not math.isfinite(confidence) or confidence<.01:raise ValueError('same_serial_confidence_floor_required')
        weights=workspace_path(info['weights'],exists=True)
        if sha256(weights)!=info['sha256']:raise ValueError('frozen_weights_changed')
        metadata[name]={'manifest_sha256':sha256(path),'weights_sha256':info['sha256'],'operating_confidence':confidence,
                        'architecture':info.get('architecture'),'input_size':info['input_size'],'backend':'onnxruntime-cpu','backend_version':'1.31.1'}
        paths={'manifest':path,'weights':weights,'source_release':source,'provenance':workspace_path(info['training_provenance'],exists=True),
               'verification_face':path.parent/'verification-final.json','verification_plate':path.parent/'verification-plate.json'}
        closure[name]={key:{'path':relative(value),'sha256':sha256(value)} for key,value in paths.items()}
    if set(metadata)!={'baseline','candidate'}:raise ValueError('exactly_one_baseline_and_candidate_required')
    return metadata,closure,classes

def assert_closure(execution):
    if sha256(workspace_path(execution['parent_plan'],exists=True))!=execution['parent_plan_sha256'] or sha256(family.DEST/'winner-freeze.json')!=execution['family_freeze_sha256']:raise ValueError('parent_plan_or_winner_changed')
    if sha256(PROTOCOL)!=execution['parallel_protocol_sha256'] or source_pins()!=execution['source_sha256'] or sha256(Path(__file__))!=execution['worker_source_sha256']:raise ValueError('execution_source_changed')
    if sha256(workspace_path(execution['benchmark_manifest'],exists=True))!=execution['benchmark_sha256']:raise ValueError('benchmark_changed')
    for item in execution['model_closure'].values():
        for reference in item.values():
            if sha256(workspace_path(reference['path'],exists=True))!=reference['sha256']:raise ValueError('model_or_source_artifact_changed')

def worker(plan_path,name,output,expected_sha):
    from agent.models import OnnxModel
    plan_path=workspace_path(plan_path,exists=True);output=workspace_path(output)
    if not plan_path.is_relative_to(family.DEST) or output.parent!=plan_path.parent or output.exists():raise ValueError('fresh_private_worker_output_required')
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

def assert_worker_attempt(execution):
    if type(execution.get('parent_pid')) is not int or os.getppid()!=execution['parent_pid']:raise ValueError('active_direct_parent_worker_only')
    role=execution.get('comparison_role')
    if role not in ('fresh','regression'):raise ValueError('known_worker_role_required')
    path=workspace_path(execution['attempt_path'],exists=True)
    if path!=workspace_path(family.DEST/(role+'-attempt.json')) or sha256(path)!=execution['attempt_sha256']:raise ValueError('same_active_one_shot_attempt_required')
    value=read(path)
    if value.get('status')!='started' or value.get('writer')!='two_process_CPU' or value.get('plan_sha256')!=execution['parent_plan_sha256'] or value.get('parallel_protocol_sha256')!=execution['parallel_protocol_sha256'] or value.get('selection_allowed') is not False or value.get('automatic_retry_allowed') is not False:raise ValueError('active_parallel_attempt_required')

def stop_owned(jobs):
    errors=[]
    for process in jobs.values():
        try:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        except ProcessLookupError:pass
        except Exception as error:errors.append(type(error).__name__)
    if errors:raise RuntimeError('owned_child_cleanup_failed')

def wait_owned(jobs,timeout=7200):
    started=time.monotonic()
    while True:
        statuses={name:process.poll() for name,process in jobs.items()}
        if any(code is not None and code!=0 for code in statuses.values()):raise ValueError('owned_model_worker_failed')
        if all(code==0 for code in statuses.values()):return
        if time.monotonic()-started>timeout:raise ValueError('owned_model_workers_timed_out')
        time.sleep(.1)

def launch(commands,folder,timeout):
    if set(commands)!={'baseline','candidate'}:raise ValueError('exactly_two_model_processes_required')
    jobs={};logs=[];env=os.environ.copy();cache=ROOT/'.cache'
    for name,relative_path in [('TEMP','tmp'),('TMP','tmp'),('TMPDIR','tmp'),('PYTHONPYCACHEPREFIX','pycache')]:
        path=workspace_path(cache/relative_path);path.mkdir(exist_ok=True);env[name]=str(path)
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    try:
        for name,command in commands.items():
            stream=(folder/(name+'.log')).open('x',encoding='utf-8');logs.append(stream)
            jobs[name]=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=stream)
        wait_owned(jobs,timeout)
    finally:
        try:stop_owned(jobs)
        finally:
            for stream in logs:stream.close()

def validate_worker_output(value,name,execution,execution_sha,expected_order):
    metadata=execution['execution_models'][name]
    if value.get('synthetic_contract') is not False or value.get('name')!=name or value.get('execution_plan_sha256')!=execution_sha or value.get('model_manifest_sha256')!=metadata['manifest_sha256'] or value.get('weights_sha256')!=metadata['weights_sha256'] or value.get('benchmark_sha256')!=execution['benchmark_sha256']:raise ValueError('worker_output_binding_failed')
    if value.get('backend')!='onnxruntime-cpu' or value.get('version')!='1.31.1' or value.get('providers')!=['CPUExecutionProvider'] or (value.get('intra_threads'),value.get('inter_threads'),value.get('CV_threads'))!=(2,1,1):raise ValueError('worker_runtime_profile_mismatch')
    if value.get('sample_order')!=expected_order or value.get('images')!=len(expected_order) or len(value.get('timings',[]))!=len(expected_order) or len(value.get('prediction_counts',[]))!=len(expected_order):raise ValueError('complete_same_sorted_worker_rows_required')
    offset=0
    for sid,count in zip(expected_order,value['prediction_counts']):
        if type(count) is not int or count<0:raise ValueError('valid_per_image_prediction_counts_required')
        if any(p['sample_id']!=sid for p in value['predictions'][offset:offset+count]):raise ValueError('per_image_prediction_order_corrupted')
        offset+=count
    if offset!=len(value['predictions']) or any(not math.isfinite(t) or t<0 for t in value['timings']):raise ValueError('complete_predictions_and_timings_required')

def run(role,authorized=False):
    if authorized is not True:raise ValueError('explicit_root_final_comparison_authorization_required')
    execution_protocol,_,frozen,parent,plan=admit(role)
    output=workspace_path(family.DEST/(role+'-comparison.json'))
    if output.exists():raise ValueError('final_comparison_already_exists')
    attempt=workspace_path(family.DEST/(role+'-attempt.json'));parent_sha=sha256(parent);protocol_sha=sha256(PROTOCOL)
    if read(parent)!=plan or check_protocol()!=execution_protocol:raise ValueError('admission_snapshot_changed')
    claim_attempt(attempt,{'status':'started','plan_sha256':parent_sha,'parallel_protocol_sha256':protocol_sha,'writer':'two_process_CPU','selection_allowed':False,'automatic_retry_allowed':False})
    try:
        from agent.dataset import validate_research_release
        benchmark=workspace_path(plan['benchmark_manifest'],exists=True)
        rows=sorted((r for r in read(benchmark) if r['split']=='test'),key=lambda r:r['sample_id'])
        if len(rows)!=plan['expected_samples'] or not validate_research_release(rows)['valid']:raise ValueError('same_frozen_benchmark_and_count_required')
        metadata,closure,classes=model_bindings(plan,rows);truth=truth_rows(rows,classes)
        directory=workspace_path(family.DEST/('parallel-'+role+'-v8'));directory.mkdir(exist_ok=False)
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
        report={'scope':plan['role'],'evaluated_at':datetime.now(timezone.utc).isoformat(),'benchmark_sha256':sha256(benchmark),'plan_sha256':parent_sha,'samples':len(rows),'results':results,
            'map50_change':results['candidate']['ap_metrics']['map50']-results['baseline']['ap_metrics']['map50'],
            'known_class_conditional':any(r.get('known_label_coverage') or r.get('evaluation_class') for r in rows),'test_used_for_selection':False,
            'independent_physical_identity_or_route_verified':False,'project_acceptance_passed':False,
            'limits':['Publisher-known classes only; unknown other classes excluded from negatives','Single-model CPU timing under shared host load, not field/full-pipeline FPS','Small source proxy groups do not prove physical independence',
                      'Two concurrent CPU2/CV1 model processes; latency includes contention, not isolated serial speed or FPS promotion']}
        if output.exists():raise ValueError('result_written_by_other_execution')
        atomic_json(output,report);atomic_json(output.with_name(output.stem+'-per-image.json'),{'truth':truth,'predictions':{name:v['predictions'] for name,v in found.items()}})
        atomic_json(directory/'execution-receipt.json',{'status':'completed','parent_plan_sha256':parent_sha,'execution_plan_sha256':execution_sha,'parallel_protocol_sha256':protocol_sha,
                    'wall_seconds':elapsed,'wall_scope':'worker launch-to-join only, excluding admission and final scoring/serialization','model_processes':2,'model_or_threshold_selection_allowed':False,'timing_is_shared_parallel_context':True,'GPU_operations':0})
        atomic_json(attempt,{'status':'completed','plan_sha256':parent_sha,'parallel_protocol_sha256':protocol_sha,'writer':'two_process_CPU','samples':len(rows),'selection_allowed':False})
        print(json.dumps({'status':'frozen_comparison_completed','role':role,'samples':len(rows),'wall_seconds':elapsed,'math_unchanged':True}))
    except BaseException as error:
        atomic_json(attempt,{'status':'failed','error_type':type(error).__name__,'plan_sha256':parent_sha,'parallel_protocol_sha256':protocol_sha,'writer':'two_process_CPU','selection_allowed':False,'automatic_retry_allowed':False});raise

def contract_worker(source,output):
    source=workspace_path(source,exists=True);output=workspace_path(output)
    if not source.is_relative_to(ROOT/'.cache') or not output.is_relative_to(ROOT/'.cache') or output.exists():raise ValueError('fresh_synthetic_cache_contract_only')
    value=read(source)
    if value.get('synthetic_contract') is not True:raise ValueError('explicit_synthetic_fixture_required')
    atomic_json(output,value)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run','worker','contract-worker']);parser.add_argument('--role',choices=['fresh','regression']);parser.add_argument('--authorized',action='store_true');parser.add_argument('--name',choices=['baseline','candidate']);parser.add_argument('--plan');parser.add_argument('--plan-sha');parser.add_argument('--output');args=parser.parse_args()
    try:
        if args.action=='prepare':prepare()
        elif args.action=='run':run(args.role,args.authorized)
        elif args.action=='worker':worker(args.plan,args.name,args.output,args.plan_sha)
        else:contract_worker(args.plan,args.output)
    except Exception as error:print(json.dumps({'status':'parallel_comparison_stopped','error_type':type(error).__name__}),file=sys.stderr);raise SystemExit(1)
