"""Privacy-only two-process CPU execution, unchanged canonical inference/score.

Each model preserves sorted sample order and per-image prediction order. This
changes scheduling/timing context, never models, thresholds, NMS or GT selection.
Lane evaluations already completed with the serial tool remain untouched.
"""
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
from agent.models import OnnxModel
from agent.dataset import validate_research_release
from scripts.compare_detector_models import known_classes,score
from scripts.compare_lane_models import preserve_freeze
from scripts.evaluate_public_refinement import gates,publish
DEST=ROOT/'data/training/refinement-evaluation'
PROTOCOL=ROOT/'docs/reports/PRIVACY-REFINEMENT-PARALLEL-PROTOCOL-V2.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def private_output(path):
    path=workspace_path(path)
    if not path.is_relative_to(DEST) or path.suffix!='.json' or path.exists():raise ValueError('fresh_private_refinement_JSON_required')
    return path
def protocol():
    return {'format':'privacy-parallel-evaluation-protocol-v2','created_before_privacy_final_predictions':True,
        'parent_protocol_sha256':'32b4c6b2c5a82997d3a985db953ea03c1fabceabc4224b1c32c29beb740c5c90',
        'parent_protocol_git_commit':'e032bfe','runtime_optimization_git_commit':'e201a8e',
        'revision_reason':'Pre-final-evaluation source pin update for bitwise-equivalent float32 preprocessing copy optimization; not a training/threshold/NMS/normalization arithmetic change',
        'optimization_evidence':{'bitwise_cases_per_environment':828,'environments':2,'applied_predict_setInput_exact_cases':828,'microbenchmark_report_sha256':sha256(ROOT/'docs/reports/PREPROCESSING-MICROBENCH.json')},
        'prior_attempt':{'status':'preflight_source_pin_mismatch','TEST_predictions':0,'model_workers_spawned':0,'original_protocol_and_failure_log_preserved':True},
        'parent_fresh_plan_sha256':sha256(ROOT/'docs/reports/PRIVACY-REFINEMENT-FRESH-PLAN.json'),'parent_regression_plan_sha256':sha256(ROOT/'docs/reports/PRIVACY-REFINEMENT-REGRESSION-PLAN.json'),
        'worker_source_sha256':sha256(Path(__file__)),'canonical_model_source_sha256':sha256(ROOT/'agent/models.py'),'canonical_score_source_sha256':sha256(ROOT/'scripts/compare_detector_models.py'),
        'execution':'two concurrent independent CPU model processes, each ORT intra2/inter1 and OpenCV2; no GPU',
        'order':'identical sorted sample_id test rows; identical OnnxModel prediction order within each row; never sort or interleave across models',
        'math':'unchanged OnnxModel.predict confidence_floor.01/NMS and exact original score function for AP/selected_confidence/fixed.35',
        'data':'fresh200 and oldTEST400 only, unchanged parent plans; all other historical tests remain leakage blacklist only',
        'timing':'shared parallel CPU contention; per-model median/p95 are not isolated serial model speedups; execution wall time separately recorded',
        'training_selection_model_threshold_changed':False,'old_lane_results_changed':False,'GPU_operations':0}

def assemble_results(truth,predictions,classes,metadata,timings):
    """Exactly the serial helper's metrics; timing is deliberately contextual."""
    return {name:{**metadata[name],'ap_metrics':score(truth,found,classes,.01),
        'operating_metrics':score(truth,found,classes,metadata[name]['operating_confidence']),
        'fixed_035_reference':score(truth,found,classes,.35),
        'latency_seconds':{'median':float(np.median(timings[name])),'p95':float(np.percentile(timings[name],95))}} for name,found in predictions.items()}

def worker(plan_path,name,output,plan_sha):
    plan_path=workspace_path(plan_path,exists=True)
    output=private_output(output)
    if not plan_path.is_relative_to(DEST):raise ValueError('private_execution_plan_required')
    if sha256(plan_path)!=plan_sha:raise ValueError('execution_plan_changed_before_worker')
    plan=read(plan_path);manifest=workspace_path(plan['models'][name],exists=True);info=read(manifest)
    model=OnnxModel(manifest);model.manifest=dict(model.manifest,confidence=.01);cv2.setNumThreads(2)
    if model.backend!='onnxruntime-cpu' or model.backend_version!='1.31.1':raise ValueError('canonical_CPU131_worker_required')
    bench=workspace_path(plan['benchmark_manifest'],exists=True)
    if sha256(bench)!=plan['benchmark_sha256']:raise ValueError('benchmark_changed_before_worker')
    corpus=read(bench);rows=sorted((r for r in corpus if r['split']=='test'),key=lambda r:r['sample_id'])
    predictions=[];timings=[]
    for row in rows:
        if row.get('ignore_bboxes_xywh_px'):raise ValueError('same_serial_ignore_region_protocol_required')
        known=known_classes(row,info['classes']);image=cv2.imread(str(workspace_path(row['path'],exists=True)))
        start=time.perf_counter();found=model.predict(image);timings.append(time.perf_counter()-start)
        predictions.extend(dict(item,sample_id=row['sample_id']) for item in found if item['class'] in known)
    if sha256(manifest)!=plan['execution_models'][name]['manifest_sha256'] or sha256(workspace_path(info['weights']))!=info['sha256']:raise ValueError('frozen_model_changed')
    if sha256(plan_path)!=plan_sha:raise ValueError('execution_plan_changed_during_worker')
    if sha256(bench)!=plan['benchmark_sha256']:raise ValueError('benchmark_changed_during_worker')
    atomic_json(output,{'synthetic_contract':False,'name':name,'model_manifest_sha256':sha256(manifest),'weights_sha256':info['sha256'],
        'benchmark_sha256':sha256(bench),'predictions':predictions,'timings':timings,'images':len(rows),'backend':model.backend,'version':model.backend_version})

def run(role):
    expected=protocol()
    if not PROTOCOL.exists() or read(PROTOCOL)!=expected:raise ValueError('frozen_parallel_protocol_required')
    parent=ROOT/f'docs/reports/PRIVACY-REFINEMENT-{role.upper()}-PLAN.json';parent_sha=sha256(parent);plan=read(parent);gates('privacy',plan)
    output=DEST/f'privacy-{role}.json'
    if output.exists():raise ValueError('final_comparison_already_exists_no_repeat')
    directory=DEST/f'parallel-privacy-{role}'
    if directory.exists():raise ValueError('fresh_execution_directory_required_preserve_failed_attempt')
    bench=workspace_path(plan['benchmark_manifest'],exists=True)
    if sha256(bench)!=plan['benchmark_sha256']:raise ValueError('benchmark_changed')
    corpus=read(bench);rows=sorted((r for r in corpus if r['split']=='test'),key=lambda r:r['sample_id'])
    if not rows or not validate_research_release(rows)['valid']:raise ValueError('invalid_same_frozen_test')
    hashes={r['sha256'] for r in rows};groups={r['leakage_group_id'] for r in rows};metadata={};classes=None
    for name,path in plan['models'].items():
        manifest=workspace_path(path,exists=True);info=read(manifest)
        if info.get('task')!='redaction' or info.get('research_only') is not True or info.get('production_approved') is not False:raise ValueError('research_redaction_only')
        if classes is None:classes=info['classes']
        elif classes!=info['classes']:raise ValueError('incomparable_classes')
        source=workspace_path(info['source_release_path'],exists=True)
        if 'manifest-sha256:'+sha256(source)!=info['dataset_release']:raise ValueError('source_changed')
        dev=[r for r in read(source) if r['split'] in ('train','validation')]
        if hashes&{r['sha256'] for r in dev} or groups&{r['leakage_group_id'] for r in dev}:raise ValueError('same_serial_holdout_development_leakage')
        if name=='baseline' and sha256(manifest)!=plan['baseline_manifest_sha256']:raise ValueError('baseline_changed')
        if name=='candidate' and info['dataset_release']!='manifest-sha256:'+plan['candidate_release_sha256']:raise ValueError('candidate_release_changed')
        confidence=info.get('confidence',.35)
        if confidence<.01:raise ValueError('operating_confidence_below_fixed_floor')
        metadata[name]={'manifest_sha256':sha256(manifest),'weights_sha256':info['sha256'],'operating_confidence':confidence,'architecture':info.get('architecture'),'input_size':info['input_size'],
            'backend':'onnxruntime-cpu','backend_version':'1.31.1','timing_context':'two parallel independent CPU2-thread model processes; shared-host, not isolated serial speed'}
    if set(metadata)!={'baseline','candidate'}:raise ValueError('exactly_one_baseline_candidate')
    truth=[]
    for row in rows:
        if row.get('ignore_bboxes_xywh_px'):raise ValueError('same_serial_ignore_region_protocol_required')
        known=known_classes(row,classes);truth.extend({'sample_id':row['sample_id'],'class':obj['class'],'bbox_xywh_px':obj['bbox_xywh_px']} for obj in row['objects'] if obj['class'] in known)
    directory.mkdir(parents=True);atomic_json(directory/'source.json',{'expires_at':min(r['expires_at'] for r in rows),'source_expiry_extended':False});execution=dict(plan,execution_models=metadata)
    execution_path=directory/'execution-plan.json';atomic_json(execution_path,execution);execution_sha=sha256(execution_path)
    preserve_freeze(output.with_name(output.stem+'-freeze.json'),{'plan_sha256':sha256(parent),'benchmark_sha256':sha256(bench),'models':metadata,'parallel_protocol_sha256':sha256(PROTOCOL),'model_or_threshold_selection_allowed':False})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');jobs={};logs=[];start=time.perf_counter()
    try:
        for name in metadata:
            log=(directory/f'{name}.log').open('w',encoding='utf-8');logs.append(log)
            jobs[name]=subprocess.Popen([sys.executable,str(Path(__file__)),'worker','--plan',str(execution_path),'--plan-sha',execution_sha,'--name',name,'--output',str(directory/f'{name}.json')],cwd=ROOT,env=env,stdout=log,stderr=log)
        for name,process in jobs.items():
            if process.wait()!=0:raise ValueError('parallel_model_worker_failed:'+name)
    finally:
        for process in jobs.values():
            if process.poll() is None:process.terminate();process.wait(timeout=30)
        for log in logs:log.close()
    elapsed=time.perf_counter()-start
    if sha256(execution_path)!=execution_sha or read(PROTOCOL)!=expected or sha256(parent)!=parent_sha or sha256(bench)!=plan['benchmark_sha256']:raise ValueError('execution_protocol_or_data_changed')
    found={name:read(directory/f'{name}.json') for name in metadata}
    for name,value in found.items():
        if value.get('synthetic_contract') is not False or value['model_manifest_sha256']!=metadata[name]['manifest_sha256'] or value['weights_sha256']!=metadata[name]['weights_sha256'] or value['benchmark_sha256']!=sha256(bench) or value['images']!=len(rows):raise ValueError('worker_output_binding_failed')
        if value['backend']!='onnxruntime-cpu' or value['version']!='1.31.1':raise ValueError('actual_CPU131_worker_required')
        path=workspace_path(plan['models'][name]);info=read(path)
        if sha256(path)!=metadata[name]['manifest_sha256'] or sha256(workspace_path(info['weights']))!=info['sha256']:raise ValueError('model_changed_during_comparison')
    predictions={name:value['predictions'] for name,value in found.items()};timings={name:value['timings'] for name,value in found.items()}
    results=assemble_results(truth,predictions,classes,metadata,timings)
    report={'scope':plan['role'],'evaluated_at':datetime.now(timezone.utc).isoformat(),'benchmark_sha256':sha256(bench),'plan_sha256':sha256(parent),'samples':len(rows),'results':results,
        'map50_change':results['candidate']['ap_metrics']['map50']-results['baseline']['ap_metrics']['map50'],'known_class_conditional':True,'test_used_for_selection':False,
        'independent_physical_identity_or_route_verified':False,'project_acceptance_passed':False,'parallel_protocol_sha256':sha256(PROTOCOL),'parallel_execution_wall_seconds':elapsed,
        'limits':['Same frozen test/model/score; scheduling only changed','Per-model latency includes concurrent CPU contention, not isolated serial benchmarking','Publisher known-class labels only; no full privacy acceptance']}
    atomic_json(output,report);atomic_json(output.with_name(output.stem+'-per-image.json'),{'truth':truth,'predictions':predictions});publish('privacy')
    print(json.dumps({'cohort':role,'images':len(rows),'parallel_execution_wall_seconds':elapsed,'math_unchanged':True}))

def contract_worker(source,output):
    source=workspace_path(source,exists=True);output=workspace_path(output)
    if not source.is_relative_to(ROOT/'.cache') or not output.is_relative_to(ROOT/'.cache'):raise ValueError('synthetic_contract_must_stay_under_cache')
    fixture=read(source);atomic_json(output,{'synthetic_contract':True,'predictions':fixture['predictions'],'timings':fixture['timings']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze-protocol','worker','run','contract-worker']);p.add_argument('--role',choices=['fresh','regression']);p.add_argument('--plan');p.add_argument('--plan-sha');p.add_argument('--name',choices=['baseline','candidate']);p.add_argument('--output');a=p.parse_args()
    if a.action=='freeze-protocol':preserve_freeze(PROTOCOL,protocol())
    elif a.action=='worker':worker(a.plan,a.name,a.output,a.plan_sha)
    elif a.action=='contract-worker':contract_worker(a.plan,a.output)
    else:run(a.role)
