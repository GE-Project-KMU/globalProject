"""Exact frozen-model CPU-only ABBA on nonpersonal synthetic16; opt-in execution."""
from __future__ import annotations
import argparse,copy,gc,hashlib,json,math,os,platform,shutil,statistics,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np,psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent import pipeline
from agent.paths import workspace_path,sha256,atomic_json
from agent.material_policy import source_deadline
from agent.inference_runner import create_models,InferenceRunner,profile_options
from scripts import gpu_model_set as g
PLAN=ROOT/'data/gpu-model-sets/cycle2-final-v7/plan.json'
PLAN_SHA='fda53437ca10b9fa7e0f87eb2b8160adcbde076c39c0fa2c80aa52104a5f443c'
BASE=ROOT/'data/gpu-model-sets/cycle2-final-v7'
PUBLIC=ROOT/'docs/reports/FINAL-CPU-PIPELINE-BENCHMARK-PROTOCOL.json'
SOURCES=(*g.RUNTIME_SOURCES,'agent/pipeline.py','agent/inference_runner.py','agent/evidence_writer.py','agent/material_policy.py','agent/governance.py','scripts/benchmark_final_cpu_pipeline.py')
FRAMES=16;WIDTH=1280;HEIGHT=720;SEED=20261112
RAW_BUDGET=512*1024**2;RSS_BUDGET=8*1024**3;OUTPUT_BUDGET=1024**3;RESERVE=512*1024**2

def source_pins():return {n:sha256(ROOT/n) for n in SOURCES}
def environment():return {'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,'system':platform.system(),'machine':platform.machine()}
def fresh(path,value):
    path=workspace_path(path)
    if path.exists():raise ValueError('new_CPU_benchmark_output_required')
    atomic_json(path,value)
def scope(path):
    path=workspace_path(path)
    if path==BASE or not path.is_relative_to(BASE):raise ValueError('final_CPU_benchmark_child_required')
    return path

def models_and_cap():
    if sha256(PLAN)!=PLAN_SHA:raise ValueError('exact_final_modelset_required')
    plan=g.validate_plan(g.read(PLAN));caps=[];bindings={}
    for task,entry in plan['models'].items():
        manifest=g.pinned(entry['manifest']);provenance=workspace_path(manifest['training_provenance'],exists=True)
        from scripts.research_retention import evidence_expiries
        values=evidence_expiries(g.read(provenance))
        if not values:raise ValueError('source_provenance_expiry_required')
        caps.extend(values)
        bindings[task]={'manifest':entry['manifest'],'weights_sha256':entry['weights_sha256'],'cpu_verifications':entry['cpu_verifications'],'provenance':g.pointer(provenance)}
    cap=min(caps);source_deadline(cap.isoformat(),required=True)
    return plan,bindings,cap.isoformat()

def policy():
    _,bindings,cap=models_and_cap()
    return {'format':'final-frozen-models-synthetic16-CPU-ABBA-v1','model_set':g.pointer(PLAN),'models':bindings,
        'source_sha256':source_pins(),'environment':environment(),'frames_per_timed_run':FRAMES,'input_size':[WIDTH,HEIGHT],
        'OpenCV_threads':1,'seed':SEED,'order':['default','fast','fast','default'],'profiles':{n:profile_options(n) for n in ('default','fast')},
        'io_profile':'overlap','model_sessions_per_run':3,'maximum_concurrent_profiles':1,'warmups_per_run':5,
        'raw_post_proof':'Untimed synthetic16 default then fast, same three sessions before timed work; raw arrays RAM-only, input blob SHA and post exact',
        'raw_tolerance':{'rtol':.001,'atol':.0001},'timed_predict_calls_each_model':FRAMES,'quality_bad_does_not_permit_model_skip':True,'raw_reference_budget':RAW_BUDGET,'main_rss_budget':RSS_BUDGET,'output_budget':OUTPUT_BUDGET,
        'model_source_expires_at':cap,'synthetic_expiry_rule':'min(actual nonpersonal generation UTC+2h, original model source expiry)',
        'source_capture_is_physical':False,'GPU_operations':0,'TEST_or_user_images_used':False,
        'scope':'FP32 CPU pipeline preprocessing/inference/post/quality/tracking/redaction/PNG/JSON/finalflush; load/warm/proof/teardown separate; no capture/decode/GNSS/API',
        'independent_accuracy_proven':False,'ten_fps_field_acceptance':False,'ten_minute_sustained_test':False,'automatic_retry_allowed':False}

def prepare(execution,output):
    execution=scope(execution);output=scope(output)
    if execution.is_relative_to(output) or any(p.exists() for p in (execution,output,PUBLIC)):raise ValueError('new_separate_CPU_execution_required')
    value=policy();fresh(PUBLIC,value)
    fresh(execution,{'format':'final-CPU-execution-v1','protocol':g.pointer(PUBLIC),'policy':value,'output':output.relative_to(ROOT).as_posix(),'run_count':1})
    print(json.dumps({'execution_sha256':sha256(execution),'protocol_sha256':sha256(PUBLIC),'GPU_operations':0,'models_created':0,'images_read':0}))

def generated_frames():
    rng=np.random.default_rng(SEED)
    return [rng.integers(0,256,size=(HEIGHT,WIDTH,3),dtype=np.uint8) for _ in range(FRAMES)]

def normalize(result):
    ids={};items=[]
    for row in result['observations']:
        defects=[]
        for d in row['potholes']:
            item=copy.deepcopy(d)
            if 'event_id' in item:item['event_id']=ids.setdefault(item['event_id'],len(ids))
            defects.append(item)
        items.append({k:row[k] for k in ('frame_id','pts_seconds','quality','lane')}|{'potholes':defects})
    return items

def exact_post(a,b):
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact_post(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(exact_post(x,y) for x,y in zip(a,b))
    return a==b

class Capture:
    def __init__(self,net):self.net=net;self.raw=None;self.blob_sha=None
    def __getattr__(self,name):return getattr(self.net,name)
    def setInput(self,blob):self.blob_sha=hashlib.sha256(np.ascontiguousarray(blob).tobytes()).hexdigest();return self.net.setInput(blob)
    def forward(self):
        value=self.net.forward()
        if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('finite_FP32_raw_required')
        self.raw=np.array(value,copy=True);return value

def proof(models,profile,frames,reference,deadline):
    proxies={k:Capture(m.net) for k,m in models.items()};original={k:m.net for k,m in models.items()};stored=[];total=0;maxerr=0.
    try:
        for k,m in models.items():m.net=proxies[k]
        with InferenceRunner(models,profile) as runner:
            for index,frame in enumerate(frames):
                source_deadline(deadline,required=True);guard()
                found=runner.predict(frame);current={k:{'raw':p.raw,'blob_sha':p.blob_sha} for k,p in proxies.items()}
                if reference is None:
                    total+=sum(v['raw'].nbytes for v in current.values())
                    if total>RAW_BUDGET:raise ValueError('raw_RAM_reference_budget_exceeded')
                    stored.append((current,copy.deepcopy(found)))
                else:
                    raw,post=reference[index]
                    for task,actual in current.items():
                        expected=raw[task]
                        if actual['blob_sha']!=expected['blob_sha'] or actual['raw'].shape!=expected['raw'].shape or not np.allclose(actual['raw'],expected['raw'],rtol=.001,atol=.0001):raise ValueError('CPU_profile_default_raw_tolerance_failed')
                        maxerr=max(maxerr,float(np.max(np.abs(actual['raw']-expected['raw']))))
                    if not exact_post(found,post):raise ValueError('CPU_profile_postprocess_exact_failed')
    finally:
        for k,m in models.items():m.net=original[k]
    return stored if reference is None else None,{'cases':len(frames),'tasks':len(models),'passed':True,'max_abs_error':maxerr,'raw_arrays_written':0,'comparison':'default_reference_capture' if reference is None else 'fast_vs_default'}

def guard():
    rss=psutil.Process().memory_info().rss
    if rss>RSS_BUDGET:raise ValueError('CPU_benchmark_RSS_budget_exceeded')
    return rss

def close_owned(models,closed,primary=None):
    errors=[]
    for model in models.values():
        if id(model) in closed:continue
        try:model.close();closed.add(id(model))
        except BaseException as e:errors.append(e)
    if errors:
        if primary is not None:
            for e in errors:primary.add_note('CPU model cleanup failed: '+type(e).__name__)
        else:raise errors[0]

def execute(frames,output,parents,deadline,*,loader=create_models,require_real=True):
    if len(frames)!=FRAMES or any(f.shape!=(HEIGHT,WIDTH,3) or f.dtype!=np.uint8 for f in frames):raise ValueError('exact_synthetic16_required')
    reference=None;runs=[];all_hashes=[];all_semantics=[];proofs=[]
    for index,profile in enumerate(('default','fast','fast','default')):
        source_deadline(deadline,required=True);models={};closed=set();primary=None;teardown=[];cleanup_errors=[]
        try:
            rss_before=guard();start=time.perf_counter();models=loader(parents,profile,inference_backend='cpu');load=time.perf_counter()-start
            if set(models)!=set(g.TASKS) or len({id(m) for m in models.values()})!=3:raise ValueError('exact_three_independent_CPU_sessions_required')
            if require_real:
                if cv2.getNumThreads()!=1:raise ValueError('CPU_OpenCV_one_thread_required')
                expected=profile_options(profile)
                for m in models.values():
                    opts=m.net.session.get_session_options()
                    if m.backend!='onnxruntime-cpu' or m.backend_version!='1.31.1' or m.net.session.get_providers()!=['CPUExecutionProvider'] or opts.intra_op_num_threads!=expected['cpu_threads'] or opts.inter_op_num_threads!=1 or m.net.runtime_options!={'cpu_intra_op_threads':expected['cpu_threads'],'cpu_inter_op_threads':1,'allow_spinning':expected['allow_spinning']}:raise ValueError('exact_CPU_ORT1311_profile_required')
                del m,opts
            start=time.perf_counter()
            with InferenceRunner(models,profile) as runner:
                for frame in frames[:5]:
                    source_deadline(deadline,required=True);runner.predict(frame)
            warm=time.perf_counter()-start;proof_seconds=0.
            if index<2:
                start=time.perf_counter();value,report=proof(models,profile,frames,reference,deadline);proof_seconds=time.perf_counter()-start;proofs.append(report)
                if index==0:reference=value
                else:reference=None
            predict_times={task:[] for task in models};phases={k:[] for k in ('quality','redaction','tracking','PNG_publish','JSON')}
            def timer(values,fn):
                def timed(*a,**k):
                    tick=time.perf_counter()
                    try:return fn(*a,**k)
                    finally:values.append(time.perf_counter()-tick)
                return timed
            for task,model in models.items():model.predict=timer(predict_times[task],model.predict)
            del model
            def final_close(values):
                start=time.perf_counter()
                try:close_owned(values,closed)
                except BaseException as error:cleanup_errors.append(error)
                finally:
                    values.clear();gc.collect();teardown.append(time.perf_counter()-start)
            def replay(*a,**k):
                for i,frame in enumerate(frames):
                    source_deadline(deadline,required=True);guard()
                    yield {'frame_id':i,'frame':frame,'pts_seconds':i/12}
            start=time.perf_counter()
            with patch.object(pipeline,'replay',replay),patch.object(pipeline,'create_models',lambda *a,**k:models),patch.object(pipeline,'close_models',final_close),patch.object(pipeline,'frame_quality',timer(phases['quality'],pipeline.frame_quality)),patch.object(pipeline,'redact',timer(phases['redaction'],pipeline.redact)),patch.object(pipeline.Tracker,'update',timer(phases['tracking'],pipeline.Tracker.update)),patch.object(pipeline.EvidenceWriter,'_write',timer(phases['PNG_publish'],pipeline.EvidenceWriter._write)),patch.object(pipeline,'atomic_json',timer(phases['JSON'],pipeline.atomic_json)):
                result=pipeline.analyze_video(output/'nonpersonal-memory',output/f'run-{index}',pothole_manifest=parents['pothole'],lane_manifest=parents['lane'],redaction_manifest=parents['redaction'],cpu_profile=profile,inference_backend='cpu',io_profile='overlap',source_expires_at=deadline,source_captured_at=None)
            wall=time.perf_counter()-start;rss_after=guard()
            if cleanup_errors:raise cleanup_errors[0]
            if any(len(calls)!=FRAMES for calls in predict_times.values()):raise ValueError('all_three_models_must_predict_every_frame')
            if result['analyzed_frames']!=FRAMES or [r['frame_id'] for r in result['observations']]!=list(range(FRAMES)):raise ValueError('CPU_frame_count_or_order_changed')
            hashes=[]
            for row in result['observations']:
                if sha256(row['image']['path'])!=row['image']['sha256']:raise ValueError('CPU_saved_PNG_changed')
                hashes.append(row['image']['sha256'])
            all_hashes.append(hashes);all_semantics.append(normalize(result))
            if all_hashes[-1]!=all_hashes[0] or all_semantics[-1]!=all_semantics[0]:raise ValueError('CPU_PNG_quality_tracking_lane_exact_failed')
            elapsed=wall-sum(teardown)
            if not math.isfinite(elapsed) or elapsed<=0:raise ValueError('positive_CPU_processing_wall_required')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>OUTPUT_BUDGET:raise ValueError('CPU_output_budget_exceeded')
            row={'profile':profile,'frames':FRAMES,'model_sessions':3,'model_load_seconds':load,'warmup_calls':5,'warmup_seconds':warm,'untimed_raw_post_proof_seconds':proof_seconds,'pipeline_wall_including_teardown':wall,'teardown_seconds':sum(teardown),'processing_seconds':elapsed,'processing_fps':FRAMES/elapsed,'rss_before':rss_before,'rss_after':rss_after,'io_profile':'overlap','actual_timed_model_predict_calls':{k:len(v) for k,v in predict_times.items()},'model_predict_median_ms':{k:statistics.median(v)*1000 for k,v in predict_times.items()},'stage_total_ms':{k:sum(v)*1000 for k,v in phases.items()},'stage_times_overlap_and_cannot_be_summed':True}
            runs.append(row);atomic_json(output/f'run-{index}-timing.json',row)
        except BaseException as e:
            primary=e
            for cleanup in cleanup_errors:e.add_note('CPU pipeline cleanup also failed: '+type(cleanup).__name__)
            raise
        finally:
            close_owned(models,closed,primary);models.clear();gc.collect()
    return {'runs':runs,'median_processing_fps':{p:statistics.median(r['processing_fps'] for r in runs if r['profile']==p) for p in ('default','fast')},'raw_post_proof':proofs,'PNG_quality_tracking_lane_exact':True,'raw_arrays_written':0}

def run(execution_path,execution_sha,approved=False):
    if approved is not True:raise ValueError('explicit_isolated_CPU_slot_required')
    execution_path=scope(execution_path)
    if sha256(execution_path)!=execution_sha:raise ValueError('CPU_execution_digest_changed')
    execution=g.read(execution_path);value=policy()
    if execution.get('protocol')!=g.pointer(PUBLIC) or g.read(PUBLIC)!=value or execution.get('policy')!=value or execution.get('run_count')!=1:raise ValueError('CPU_protocol_source_or_model_changed')
    output=scope(execution['output']);attempt=output.with_name(output.name+'-attempt.json')
    if output.exists():raise ValueError('new_CPU_run_required')
    if shutil.disk_usage(ROOT).free<OUTPUT_BUDGET+RESERVE:raise ValueError('CPU_output_reserve_required')
    created=datetime.now(timezone.utc);deadline=min(created+timedelta(hours=2),datetime.fromisoformat(value['model_source_expires_at'])).isoformat()
    with attempt.open('x',encoding='utf-8') as stream:json.dump({'status':'started','execution_sha256':execution_sha,'synthetic_created_at':created.isoformat(),'source_expires_at':deadline,'automatic_retry_allowed':False},stream);stream.flush();os.fsync(stream.fileno())
    output.mkdir();old_cv=cv2.getNumThreads();cv2.setNumThreads(1)
    try:
        plan,_,_=models_and_cap();parents={k:v['manifest']['path'] for k,v in plan['models'].items()}
        result=execute(generated_frames(),output,parents,deadline)
        if policy()!=value or sha256(execution_path)!=execution_sha:raise ValueError('CPU_source_or_model_changed_during_execution')
        result.update(format='final-CPU-synthetic16-ABBA-results-v1',model_set_sha256=PLAN_SHA,execution_sha256=execution_sha,source_sha256=source_pins(),GPU_operations=0,source_expires_at=deadline,synthetic_created_at=created.isoformat(),new_source_generation_is_nonpersonal=True,accuracy_proof_scope='synthetic16 only; not independent VAL accuracy',ten_fps_field_acceptance=False,ten_minute_sustained_test=False,decode_or_capture_measured=False,automatic_retry_allowed=False)
        atomic_json(output/'aggregate.json',result);atomic_json(attempt,{'status':'completed','execution_sha256':execution_sha,'automatic_retry_allowed':False})
    except BaseException as e:atomic_json(attempt,{'status':'failed','error_type':type(e).__name__,'execution_sha256':execution_sha,'automatic_retry_allowed':False});raise
    finally:cv2.setNumThreads(old_cv)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='action',required=True)
    a=commands.add_parser('prepare');a.add_argument('--execution-plan',required=True);a.add_argument('--output',required=True)
    a=commands.add_parser('run');a.add_argument('--execution-plan',required=True);a.add_argument('--execution-sha256',required=True);a.add_argument('--approved-exclusive-cpu-slot',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.execution_plan,a.output)
    else:run(a.execution_plan,a.execution_sha256,a.approved_exclusive_cpu_slot)
