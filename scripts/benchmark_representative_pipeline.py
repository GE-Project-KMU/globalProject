"""Approved-slot-only VAL32 GPU IO ABBA; metadata freeze is image/GPU free."""
from __future__ import annotations
import argparse,hashlib,io,json,math,os,platform,shutil,statistics,sys,time
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import PIL
from PIL import Image
from agent import pipeline
from agent.inference_runner import create_models,InferenceRunner
from agent.material_policy import source_deadline
from agent.paths import workspace_path,sha256,atomic_json
from agent.governance import blocked_identities
from scripts import gpu_model_set as g
from scripts.prepare_representative_performance import select_representative,GROUPS
from scripts.research_retention import evidence_expiries
from scripts.probe_gpu_pipeline_soak import process_rss

PERF_ROOT=ROOT/'data/gpu-model-sets/cycle2-final-v7'
REGISTRY=ROOT/'data/gpu-runtime-profiles'
SOURCES=('scripts/benchmark_representative_pipeline.py','scripts/prepare_representative_performance.py',
    'scripts/gpu_model_set.py','agent/models.py','agent/inference_runner.py','agent/pipeline.py',
    'agent/evidence_writer.py','agent/gpu_worker_client.py','scripts/gpu_inference_worker.py','agent/material_policy.py','agent/paths.py','agent/governance.py','scripts/research_retention.py','scripts/probe_gpu_pipeline_soak.py')
MAX_SOURCE_BYTES=32*1024**2;MAX_SOURCE_PIXELS=24_000_000;MAX_PREPARED_BYTES=128*1024**2
OUTPUT_BUDGET=1024**3;DISK_RESERVE=512*1024**2
MAX_MAIN_RSS_BYTES=2*1024**3
WIDTH,HEIGHT=1280,720


def environment():
    return {'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),
        'opencv':cv2.__version__,'numpy':np.__version__,'pillow':PIL.__version__}


def source_pins():return {name:sha256(ROOT/name) for name in SOURCES}


def admitted_host():
    if platform.system()!=g.GPU_PROFILE['system'] or platform.machine()!=g.GPU_PROFILE['machine']:raise ValueError('admitted_gpu_host_required_before_image_read')


def guarded_rss():
    value=process_rss(os.getpid())
    if value>MAX_MAIN_RSS_BYTES:raise ValueError('main_process_rss_budget_exceeded')
    return value


def scoped_output(value):
    path=workspace_path(value)
    if path==PERF_ROOT.resolve() or not path.is_relative_to(PERF_ROOT.resolve()):raise ValueError('exact_new_performance_output_required')
    return path


def validated_representative(path):
    value=g.read(path)
    if value.get('format')!='representative-public-val32-performance-plan-v1':raise ValueError('representative_plan_format_mismatch')
    profile_ref=value['registered_profile']
    if not workspace_path(profile_ref['path'],exists=True).is_relative_to(REGISTRY.resolve()):raise ValueError('registered_local_gpu_profile_required')
    descriptor=g.pinned(profile_ref);plan=g.validate_descriptor(descriptor,True)
    if not workspace_path(descriptor['plan']['path'],exists=True).is_relative_to(PERF_ROOT.resolve()):raise ValueError('new_cycle_model_set_required')
    if descriptor['plan']!=value['inference_plan'] or descriptor['native']!=value['native_proof']:raise ValueError('representative_native_binding_mismatch')
    releases={task:g.pinned(entry['validation_manifest']) for task,entry in plan['models'].items()}
    canonical={task:{g.pinned(ref)['image_sha256'] for ref in entry['cpu_verifications']} for task,entry in plan['models'].items()}
    expected=select_representative(plan,releases,canonical)
    if value.get('cases')!=expected or value.get('case_set_sha256')!=g.canonical(expected):raise ValueError('representative_frozen_cases_changed')
    if value.get('groups')!={group:8 for group in GROUPS} or value.get('prepared_input_size')!=[WIDTH,HEIGHT]:raise ValueError('representative_group_or_size_changed')
    if any(value.get(key) is not False for key in ('source_captured_at_provided','original_capture_date_known','test_or_user_media_used','accuracy_dataset_or_labels_changed')) or value.get('precision')!='FP32 unchanged' or value.get('local_material_state')!='unreviewed':raise ValueError('representative_source_contract_changed')
    deadline=min(datetime.fromisoformat(row['expires_at'].replace('Z','+00:00')) for row in expected)
    if source_deadline(value['source_expires_at'],required=True)!=deadline:raise ValueError('representative_source_deadline_changed')
    return value,descriptor,plan


def model_source_cap(plan):
    proofs={};dates=[];blocked=blocked_identities()
    for task,entry in plan['models'].items():
        manifest=g.pinned(entry['manifest']);ref=g.pointer(manifest['training_provenance'])
        provenance=g.pinned(ref)
        if not isinstance(provenance,dict) or not isinstance(provenance.get('assets'),list) or not provenance['assets'] or any(not isinstance(asset,dict) for asset in provenance['assets']):raise ValueError('nonempty_model_source_assets_required')
        if manifest.get('data_revoked') or manifest.get('needs_revalidation') or provenance.get('data_revoked') or provenance.get('needs_revalidation') or any(
            asset.get('revoked_at') or set(asset.get('identity_hashes',[]))&blocked for asset in provenance.get('assets',[])):
            raise ValueError('model_source_revoked_or_needs_revalidation')
        values=evidence_expiries(provenance)
        if not values:raise ValueError('model_original_source_expiry_required')
        dates.extend(values);proofs[task]=ref
    cap=min(dates);source_deadline(cap.isoformat(),required=True)
    return proofs,cap


def freeze_execution(representative_path,execution_path,output):
    representative_ref=g.pointer(representative_path)
    value,descriptor,plan=validated_representative(representative_path)
    provenance,model_cap=model_source_cap(plan)
    effective_cap=min(model_cap,source_deadline(value['source_expires_at'],required=True)).isoformat()
    output=scoped_output(output);execution=scoped_output(execution_path)
    if output.exists() or execution.exists() or execution.is_relative_to(output):raise ValueError('fresh_separate_execution_and_output_required')
    result={'format':'representative-pipeline-execution-binding-v1','representative_plan':representative_ref,
        'registered_profile':value['registered_profile'],'source_sha256':source_pins(),'environment':environment(),
        'output':output.relative_to(ROOT).as_posix(),'order':['default','overlap','overlap','default'],'frames_each':32,
        'warmup_calls':5,'maximum_prepared_bytes':MAX_PREPARED_BYTES,'main_rss_budget_bytes':MAX_MAIN_RSS_BYTES,'output_budget_bytes':OUTPUT_BUDGET,
        'source_expires_at':effective_cap,'model_source_provenance':provenance,'source_captured_at':None,'timeline':'synthetic index/12',
        'images_read':0,'gpu_sessions_started':0,'run_count_per_binding':1}
    validated_representative(representative_path)
    if result['source_sha256']!=source_pins() or representative_ref['sha256']!=sha256(representative_path):raise ValueError('execution_sources_or_plan_changed_during_freeze')
    g.write(execution,result)
    return {'execution_plan_sha256':sha256(execution),'images_read':0,'gpu_sessions_started':0}


def validate_execution(path):
    value=g.read(path)
    if value.get('format')!='representative-pipeline-execution-binding-v1' or value.get('source_sha256')!=source_pins() or value.get('environment')!=environment():raise ValueError('execution_source_or_environment_changed')
    representative=g.pinned(value['representative_plan'])
    validated=validated_representative(value['representative_plan']['path'])
    provenance,model_cap=model_source_cap(validated[2])
    effective_cap=min(model_cap,source_deadline(representative['source_expires_at'],required=True)).isoformat()
    if representative!=validated[0] or value['registered_profile']!=representative['registered_profile']:raise ValueError('execution_profile_changed')
    if value.get('order')!=['default','overlap','overlap','default'] or value.get('frames_each')!=32 or value.get('warmup_calls')!=5 or value.get('maximum_prepared_bytes')!=MAX_PREPARED_BYTES or value.get('output_budget_bytes')!=OUTPUT_BUDGET or value.get('source_captured_at') is not None or value.get('source_expires_at')!=effective_cap or value.get('model_source_provenance')!=provenance or value.get('run_count_per_binding')!=1:raise ValueError('execution_scope_changed')
    if value.get('main_rss_budget_bytes')!=MAX_MAIN_RSS_BYTES:raise ValueError('execution_memory_budget_changed')
    scoped_output(value['output']);return value,*validated


def prepare_frames(cases,deadline):
    source_deadline(deadline,required=True)
    if len(cases)!=32 or 32*WIDTH*HEIGHT*3>MAX_PREPARED_BYTES:raise ValueError('fixed_prepared_frame_budget_required')
    frames=[];details=[];started=time.perf_counter()
    for row in cases:
        source_deadline(row['expires_at'],required=True);path=g.public_case_path(row['path'])
        if not path.is_file() or not 0<path.stat().st_size<=MAX_SOURCE_BYTES:raise ValueError('bounded_public_image_required')
        tick=time.perf_counter()
        with path.open('rb') as stream:data=stream.read(MAX_SOURCE_BYTES+1)
        if len(data)>MAX_SOURCE_BYTES or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('public_source_bytes_changed')
        read_done=time.perf_counter()
        with Image.open(io.BytesIO(data)) as header:
            if header.width<=0 or header.height<=0 or header.width*header.height>MAX_SOURCE_PIXELS:raise ValueError('public_image_pixel_budget_exceeded')
        image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or image.shape[0]*image.shape[1]>MAX_SOURCE_PIXELS:raise ValueError('public_image_decode_failed')
        decoded=time.perf_counter();scale=min(WIDTH/image.shape[1],HEIGHT/image.shape[0])
        resized=cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))))
        frame=np.full((HEIGHT,WIDTH,3),114,np.uint8);frame[:resized.shape[0],:resized.shape[1]]=resized
        source_deadline(deadline,required=True);source_deadline(row['expires_at'],required=True)
        frames.append(frame);details.append({'image_sha256':row['sha256'],'expires_at':row['expires_at'],
            'read_seconds':read_done-tick,'decode_seconds':decoded-read_done,'letterbox_seconds':time.perf_counter()-decoded})
    if sum(frame.nbytes for frame in frames)>MAX_PREPARED_BYTES:raise ValueError('prepared_frame_memory_budget_exceeded')
    return frames,details,time.perf_counter()-started


def close_owned(models,primary=None):
    errors=[]
    for model in models.values():
        try:model.close()
        except BaseException as error:errors.append(error)
    if errors:
        if primary is not None:
            for error in errors:primary.add_note('Model cleanup also failed: '+type(error).__name__)
        else:
            for error in errors[1:]:errors[0].add_note('Other model cleanup failed: '+type(error).__name__)
            raise errors[0]


def semantics(result):
    return [{key:row[key] for key in ('frame_id','pts_seconds','quality','lane')}|
        {'potholes':[{key:value for key,value in defect.items() if key!='event_id'} for defect in row['potholes']]} for row in result['observations']]


def execute_abba(frames,output,parents,profile,deadline,*,model_loader=create_models,require_cuda=True):
    """Testable real pipeline harness. CLI always requires CUDA and native loaders."""
    if len(frames)!=32 or any(frame.shape!=(HEIGHT,WIDTH,3) or frame.dtype!=np.uint8 for frame in frames):raise ValueError('fixed_prepared_frames_required')
    runs=[];hashes=[];observations=[]
    for index,io_profile in enumerate(('default','overlap','overlap','default')):
        source_deadline(deadline,required=True);models={};primary=None;cleanup_errors=[];teardown=[]
        try:
            rss_before=guarded_rss() if require_cuda else None
            tick=time.perf_counter();models=model_loader(parents,'fast',inference_backend='cuda',gpu_profile=profile);load=time.perf_counter()-tick
            if require_cuda and (set(models)!=set(parents) or any(getattr(model,'backend',None)!='onnxruntime-cuda-worker' or
                model.net.session.get_providers()!=g.GPU_PROFILE['providers'] or model.net.process.poll() is not None for model in models.values())):raise ValueError('actual_native_cuda_models_required')
            stage={task:[] for task in models}
            for task,model in models.items():
                predict=model.predict
                def timed(image,predict=predict,task=task):
                    t=time.perf_counter()
                    try:return predict(image)
                    finally:stage[task].append(time.perf_counter()-t)
                model.predict=timed
            tick=time.perf_counter()
            with InferenceRunner(models,'fast') as runner:
                for warm in frames[:5]:runner.predict(warm)
            warmup=time.perf_counter()-tick
            for values in stage.values():values.clear()
            if require_cuda:guarded_rss()
            def final_close(values):
                t=time.perf_counter()
                try:close_owned(values)
                except BaseException as error:cleanup_errors.append(error)
                finally:teardown.append(time.perf_counter()-t)
            def replay(*args,**kwargs):
                for frame_id,frame in enumerate(frames):
                    source_deadline(deadline,required=True)
                    yield {'frame_id':frame_id,'frame':frame,'pts_seconds':frame_id/12}
            tick=time.perf_counter()
            with patch.object(pipeline,'create_models',lambda *a,**k:models),patch.object(pipeline,'replay',replay),patch.object(pipeline,'close_models',final_close):
                result=pipeline.analyze_video(output/'prepared-public-memory',output/f'run-{index}',pothole_manifest=parents['pothole'],
                    lane_manifest=parents['lane'],redaction_manifest=parents['redaction'],cpu_profile='fast',inference_backend='cuda',
                    gpu_profile=profile,io_profile=io_profile,source_expires_at=deadline,source_captured_at=None)
            wall=time.perf_counter()-tick
            rss_after=guarded_rss() if require_cuda else None
            if cleanup_errors:raise cleanup_errors[0]
            if result['analyzed_frames']!=32 or [row['frame_id'] for row in result['observations']]!=list(range(32)):raise ValueError('saved_observation_count_or_order_changed')
            for row in result['observations']:
                if sha256(row['image']['path'])!=row['image']['sha256']:raise ValueError('saved_png_hash_changed')
            hashes.append([row['image']['sha256'] for row in result['observations']]);observations.append(semantics(result))
            processing=wall-sum(teardown)
            if not math.isfinite(processing) or processing<=0:raise ValueError('invalid_processing_wall')
            runs.append({'io_profile':io_profile,'frames':32,'model_load_seconds':load,'five_warmup_seconds':warmup,
                'pipeline_wall_including_teardown_seconds':wall,'worker_teardown_seconds':sum(teardown),
                'warmed_processing_seconds_including_png_json_flush':processing,'warmed_processing_fps':32/processing,
                'model_predict_median_ms':{task:statistics.median(values)*1000 for task,values in stage.items()},
                'main_rss_bytes_before':rss_before,'main_rss_bytes_after':rss_after,'main_rss_budget_bytes':MAX_MAIN_RSS_BYTES,
                'actual_native_cuda_required':require_cuda})
            print(json.dumps({'completed_run':index+1,'io_profile':io_profile,'warmed_processing_fps':32/processing}),flush=True)
        except BaseException as error:
            primary=error
            for cleanup in cleanup_errors:primary.add_note('Model teardown failed: '+type(cleanup).__name__)
            raise
        finally:close_owned(models,primary)
    return {'runs':runs,'opaque_png_bytes_same':all(value==hashes[0] for value in hashes),
        'observation_order_quality_tracking_lane_same':all(value==observations[0] for value in observations)}


def run(execution_path,*,execution_sha256=None,approved=False):
    # This check must precede metadata validation and every source image read.
    if approved is not True:raise ValueError('explicit_exclusive_gpu_slot_approval_required')
    if not isinstance(execution_sha256,str) or len(execution_sha256)!=64 or any(c not in '0123456789abcdef' for c in execution_sha256):raise ValueError('pinned_execution_sha256_required')
    if sha256(execution_path)!=execution_sha256:raise ValueError('execution_binding_digest_changed')
    binding,representative,descriptor,plan=validate_execution(execution_path)
    admitted_host()
    output=scoped_output(binding['output'])
    if output.exists():raise ValueError('execution_binding_already_used')
    if shutil.disk_usage(ROOT).free<OUTPUT_BUDGET+DISK_RESERVE:raise ValueError('output_budget_reserve_unavailable')
    output.mkdir(parents=True)
    old_profiling=os.environ.pop('ROAD_GPU_PROFILE_DIRECTORY',None)
    try:
        frames,preparation,seconds=prepare_frames(representative['cases'],binding['source_expires_at'])
        parents={task:entry['manifest']['path'] for task,entry in plan['models'].items()}
        result=execute_abba(frames,output,parents,binding['registered_profile']['path'],binding['source_expires_at'])
        validate_execution(execution_path)
        if sha256(execution_path)!=execution_sha256:raise ValueError('execution_binding_changed_during_run')
        size=sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
        if size>OUTPUT_BUDGET:raise ValueError('output_budget_exceeded')
        aggregate={'format':'representative-val32-io-abba-v1',**result,'passed':result['opaque_png_bytes_same'] and result['observation_order_quality_tracking_lane_same'],
            'execution_binding_sha256':sha256(execution_path),'source_sha256':binding['source_sha256'],'environment':binding['environment'],
            'registered_profile_sha256':binding['registered_profile']['sha256'],'case_set_sha256':representative['case_set_sha256'],
            'preparation_read_decode_letterbox_seconds':seconds,'preparation_excluded_from_warmed_fps':True,
            'prepared_frame_bytes':sum(frame.nbytes for frame in frames),'prepared_frame_budget_bytes':MAX_PREPARED_BYTES,
            'output_bytes':size,'output_budget_bytes':OUTPUT_BUDGET,'source_expires_at':binding['source_expires_at'],
            'source_captured_at':None,'replay_timeline':'synthetic frame_index/12','local_material_state':'unreviewed',
            'test_or_user_images_read':0,'public_validation_images_read':32,'source_or_GT_modified':False,'new_precision':False,
            'real_capture_or_video_decode_in_warmed_scope':False,'ten_minute_sustained_test_completed':False,
            'post_run_png_hash_audit_excluded_from_warmed_fps':True,
            'continuous_real_road_tracking_or_field_fps_claim':False,'profiling_enabled':False}
        atomic_json(output/'private-preparation.json',preparation);atomic_json(output/'aggregate.json',aggregate)
        if aggregate['passed'] is not True:raise ValueError('exact_pipeline_output_mismatch')
        return aggregate
    except BaseException as error:
        atomic_json(output/'failure.json',{'status':'failed','error_type':type(error).__name__,'success':False,'partial_private_outputs_preserved':True})
        raise
    finally:
        if old_profiling is not None:os.environ['ROAD_GPU_PROFILE_DIRECTORY']=old_profiling


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='action',required=True)
    freeze=commands.add_parser('freeze');freeze.add_argument('--representative-plan',required=True);freeze.add_argument('--execution-plan',required=True);freeze.add_argument('--output',required=True)
    execute=commands.add_parser('run');execute.add_argument('--execution-plan',required=True);execute.add_argument('--execution-sha256',required=True);execute.add_argument('--approved-exclusive-gpu-slot',action='store_true');args=parser.parse_args()
    if args.action=='freeze':print(json.dumps(freeze_execution(args.representative_plan,args.execution_plan,args.output)))
    else:print(json.dumps(run(args.execution_plan,execution_sha256=args.execution_sha256,approved=args.approved_exclusive_gpu_slot)))


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,TypeError) as error:
        reason=str(error) if isinstance(error,ValueError) and all(character in 'abcdefghijklmnopqrstuvwxyz_0123456789' for character in str(error)) else 'metadata_or_filesystem_error'
        print(json.dumps({'status':'representative_benchmark_refused','reason':reason,'success':False}))
        raise SystemExit(2)
