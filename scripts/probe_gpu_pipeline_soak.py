"""Explicitly approved 600s GPU software soak with real decode/PNG/JSON, synthetic input only.

This is preparation code until a fresh exact model-set has passed admission and
an exclusive GPU slot has been approved. It is not a field/camera accuracy test.
"""
from __future__ import annotations
import argparse,ctypes,json,math,os,shutil,subprocess,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from agent import pipeline
from agent.inference_runner import create_models,close_models,InferenceRunner
from agent.paths import workspace_path,sha256,atomic_json
from agent.material_policy import source_deadline
from scripts.gpu_model_set import read,validate_descriptor,pointer

WIDTH,HEIGHT=1280,720
DURATION=600.
RATE=12.
BUDGET=40*1024**3
RESERVE=5*1024**3


def settings(duration,rate,budget,reserve):
    if isinstance(duration,bool) or duration!=600:raise ValueError('full_600_second_soak_required')
    if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 10<rate<=20:
        raise ValueError('explicit_synthetic_input_rate_above_ten_required')
    if type(budget) is not int or type(reserve) is not int or budget<=0 or reserve<1024**3:raise ValueError('positive_output_budget_and_reserve_required')
    frames=math.ceil(duration*rate)+2
    # Conservative cap for uncompressed RGB PNG plus JSON/metadata. No image
    # dropping/ring overwrite is used to make the disk budget appear smaller.
    bound=frames*(WIDTH*HEIGHT*3+2*1024**2+16*1024)+256*1024**2
    if bound>budget:raise ValueError('budget_below_worst_case_frame_bound')
    return {'duration_seconds':float(duration),'synthetic_input_rate':float(rate),'maximum_frames':frames,
            'output_budget_bytes':budget,'free_disk_reserve_bytes':reserve,'worst_case_output_bound_bytes':bound}


def process_rss(pid):
    if os.name!='nt':raise ValueError('this_admitted_gpu_probe_requires_windows')
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x1000|0x10,False,pid)
    if not handle:raise OSError('owned_process_memory_open_failed')
    try:
        value=Counters();value.cb=ctypes.sizeof(value)
        if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(value),value.cb):raise OSError('owned_process_memory_query_failed')
        return int(value.WorkingSetSize)
    finally:kernel.CloseHandle(handle)


def gpu_memory():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
            cwd=ROOT,capture_output=True,text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        parts=result.stdout.strip().splitlines()
        if result.returncode or len(parts)!=1:raise ValueError('gpu_query_unavailable')
        memory,utilization=[int(value.strip()) for value in parts[0].split(',')]
        return {'device_used_MiB':memory,'device_utilization_pct':utilization,'scope':'whole_device_not_per_process'}
    except (OSError,ValueError,subprocess.TimeoutExpired):return {'available':False}


def source_expiries(value):
    found=[]
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('expires_at','source_expires_at','collection_expires_at','mask_expires_at','license_expires_at') and item:
                found.append(source_deadline(item,required=True))
            elif isinstance(item,(list,dict)):found.extend(source_expiries(item))
    elif isinstance(value,list):
        for item in value:found.extend(source_expiries(item))
    return found


def generated_video(path):
    path=workspace_path(path)
    if path.exists():raise ValueError('new_generated_video_required')
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),RATE,(WIDTH,HEIGHT))
    if not writer.isOpened():raise RuntimeError('synthetic_video_encoder_unavailable')
    rng=np.random.default_rng(20261105)
    image=rng.integers(0,256,(HEIGHT,WIDTH,3),np.uint8)
    try:
        for index in range(16):writer.write(np.roll(image,index*3,axis=1))
    finally:writer.release()
    if not path.exists() or not path.stat().st_size:raise RuntimeError('synthetic_video_missing')


def decoded_samples(path,*,duration,rate,maximum_frames,observe,clock=time.monotonic,sleep=time.sleep):
    capture=cv2.VideoCapture(str(workspace_path(path,exists=True)))
    if not capture.isOpened():raise RuntimeError('synthetic_decoder_unavailable')
    started=clock();index=0
    try:
        while clock()-started<duration:
            if index>=maximum_frames:raise RuntimeError('frame_bound_exceeded')
            due=started+index/rate
            while clock()<due:sleep(min(.05,due-clock()))
            if clock()-started>=duration:break
            decode_start=clock();ok,frame=capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES,0);ok,frame=capture.read()
            if not ok or frame.shape!=(HEIGHT,WIDTH,3) or frame.dtype!=np.uint8:raise RuntimeError('synthetic_decode_contract_failed')
            elapsed=clock()-started;observe(index,elapsed,clock()-decode_start)
            yield {'frame_id':index,'frame':frame,'pts_seconds':index/rate}
            index+=1
    finally:capture.release()


def memory_summary(samples):
    kept=[row for row in samples if row['elapsed_seconds']>=30]
    if not kept:raise ValueError('post_warmup_memory_samples_required')
    roles=['main']+sorted(kept[0]['worker_rss_bytes'])
    values={role:[row['main_rss_bytes'] if role=='main' else row['worker_rss_bytes'][role] for row in kept] for role in roles}
    return {'warmup_excluded_seconds':30,'samples':len(kept),
        'host_rss_bytes':{role:{'first':rows[0],'last':rows[-1],'min':min(rows),'max':max(rows),'last_minus_first':rows[-1]-rows[0]} for role,rows in values.items()},
        'gpu_device_peak_MiB':max(row['gpu']['device_used_MiB'] for row in kept),
        'gpu_scope':'whole_device_not_per_worker','absence_of_memory_leaks_proven':False}


def run(profile_path,output,*,duration=DURATION,rate=RATE,budget=BUDGET,reserve=RESERVE,approved=False):
    if approved is not True:raise ValueError('explicit_exclusive_gpu_slot_approval_required')
    config=settings(duration,rate,budget,reserve)
    config.update(host_rss_budget_bytes=12*1024**3,gpu_device_budget_MiB=10*1024)
    profile_sha=sha256(profile_path)
    descriptor=read(profile_path);plan=validate_descriptor(descriptor,True)
    output=workspace_path(output)
    if output.exists():raise ValueError('new_soak_output_required')
    if shutil.disk_usage(ROOT).free<budget+reserve:raise ValueError('insufficient_scoped_output_budget_reserve')
    parents={task:entry['manifest']['path'] for task,entry in plan['models'].items()}
    expiries=[]
    for entry in plan['models'].values():
        manifest=read(entry['manifest']['path']);expiries.extend(source_expiries(read(manifest['training_provenance'])))
    if not expiries:raise ValueError('original_model_source_expiries_required')
    original_cap=min(expiries)
    if original_cap<=datetime.now(timezone.utc)+timedelta(seconds=duration+300):raise ValueError('model_source_expires_before_soak_can_complete')
    output.mkdir(parents=True);source=output/'generated-nonpersonal.mp4';generated_video(source)
    captured=datetime.now(timezone.utc);deadline=min(original_cap,captured+timedelta(hours=2)).isoformat()
    sources=['agent/models.py','agent/inference_runner.py','agent/pipeline.py','agent/evidence_writer.py',
             'agent/gpu_worker_client.py','scripts/gpu_inference_worker.py','scripts/probe_gpu_pipeline_soak.py']
    pins={name:sha256(ROOT/name) for name in sources};video_hash=sha256(source)
    atomic_json(output/'plan.json',{'format':'explicit-gpu-synthetic-soak-plan-v1',**config,'registered_profile':pointer(profile_path),
        'input_size':[WIDTH,HEIGHT],'source_sha256':pins,'generated_video_sha256':video_hash,'generated_source_expires_at':deadline,
        'io_profile':'overlap','cpu_profile':'fast','real_video_decode':True,'synthetic_video_replayed':True,
        'physical_camera_or_GNSS_evidence':False,'new_model_or_precision_admission_bypassed':False})
    models={};memory=[];decoded=[];progress=-30.;started=None;workers={}
    try:
        load_started=time.monotonic()
        models=create_models(parents,'fast',inference_backend='cuda',gpu_profile=profile_path)
        load_seconds=time.monotonic()-load_started
        workers={task:model.net.process for task,model in models.items()}
        cap=cv2.VideoCapture(str(source));ok,warm=cap.read();cap.release()
        if not ok:raise RuntimeError('warmup_decode_failed')
        warm_started=time.monotonic()
        with InferenceRunner(models,'fast') as runner:
            for _ in range(5):runner.predict(warm)
        warm_seconds=time.monotonic()-warm_started
        def observe(index,elapsed,decode_seconds):
            nonlocal progress
            decoded.append(decode_seconds)
            source_deadline(deadline,required=True)
            if elapsed-progress>=5:
                row={'elapsed_seconds':elapsed,'main_rss_bytes':process_rss(os.getpid()),
                    'worker_rss_bytes':{task:process_rss(proc.pid) for task,proc in workers.items()},'gpu':gpu_memory()}
                if 'device_used_MiB' not in row['gpu']:raise RuntimeError('gpu_memory_budget_cannot_be_observed')
                if row['main_rss_bytes']+sum(row['worker_rss_bytes'].values())>config['host_rss_budget_bytes']:raise RuntimeError('host_rss_budget_exceeded')
                if row['gpu']['device_used_MiB']>config['gpu_device_budget_MiB']:raise RuntimeError('gpu_device_memory_budget_exceeded')
                memory.append(row)
                progress=elapsed
                if shutil.disk_usage(ROOT).free<reserve:raise RuntimeError('free_disk_reserve_reached')
                if int(elapsed)%30<5:print(json.dumps({'elapsed_seconds':round(elapsed,1),'frames_decoded':index+1}),flush=True)
        def replay(path,**kwargs):
            if workspace_path(path)!=source:raise ValueError('generated_source_only')
            return decoded_samples(path,duration=duration,rate=rate,maximum_frames=config['maximum_frames'],observe=observe)
        started=time.monotonic()
        with patch.object(pipeline,'replay',replay),patch.object(pipeline,'create_models',lambda *a,**k:models):
            result=pipeline.analyze_video(source,output/'analysis',pothole_manifest=parents['pothole'],lane_manifest=parents['lane'],
                redaction_manifest=parents['redaction'],cpu_profile='fast',inference_backend='cuda',gpu_profile=profile_path,
                io_profile='overlap',source_expires_at=deadline,source_captured_at=captured.isoformat())
        elapsed=time.monotonic()-started
        if elapsed<duration:raise AssertionError('full_duration_not_reached')
        if result['analyzed_frames']!=len(decoded) or [row['frame_id'] for row in result['observations']]!=list(range(len(decoded))):raise AssertionError('decoded_observation_count_or_order_mismatch')
        for row in result['observations']:
            if sha256(row['image']['path'])!=row['image']['sha256']:raise AssertionError('saved_evidence_sha_mismatch')
        if sha256(source)!=video_hash or pins!={name:sha256(ROOT/name) for name in sources} or sha256(profile_path)!=profile_sha:raise AssertionError('soak_input_profile_or_source_changed')
        validate_descriptor(read(profile_path),True)
        size=sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
        if size>budget:raise AssertionError('output_budget_exceeded')
        if any(proc.poll() is None for proc in workers.values()):raise AssertionError('owned_gpu_worker_not_closed')
        aggregate={'format':'explicit-gpu-synthetic-soak-results-v1','status':'completed','planned_seconds':duration,'processing_seconds_including_decode_png_json_final_flush':elapsed,
            'inferred_saved_frames':len(decoded),'processing_fps':len(decoded)/elapsed,'ten_fps_in_this_synthetic_scope':len(decoded)/elapsed>=10,
            'source_sha256':pins,'registered_profile_sha256':profile_sha,'output_bytes':size,'output_budget_bytes':budget,'error_count':0,
            'worker_processes_closed':True,'original_source_expiries_extended':False,'io_profile':'overlap','input_size':[WIDTH,HEIGHT],
            'post_warmup_memory':memory_summary(memory),'host_rss_budget_bytes':config['host_rss_budget_bytes'],'gpu_device_budget_MiB':config['gpu_device_budget_MiB'],
            'input_paced_at_fps':rate,'uncapped_peak_throughput_claim':False,'synthetic_clip_replayed':True,'real_video_decode_measured':True,
            'model_load_and_five_warmups_excluded':True,'model_load_seconds':load_seconds,'five_warmup_seconds':warm_seconds,
            'worker_teardown_included_in_processing_wall':True,'final_evidence_sha_audit_excluded':True,'user_or_test_media_read':0,
            'field_camera_GNSS_depth_API_test':False,'model_or_privacy_accuracy_measured':False,'production_or_field_acceptance':False}
        atomic_json(output/'memory-samples.json',memory);atomic_json(output/'aggregate.json',aggregate)
        return aggregate
    except BaseException as error:
        atomic_json(output/'failure.json',{'status':'failed','error_type':type(error).__name__,'partial_outputs_preserved':True,'ten_fps_acceptance':False})
        raise
    finally:close_models(models)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--approved-exclusive-gpu-slot',action='store_true');args=parser.parse_args()
    print(json.dumps(run(args.profile,args.output,approved=args.approved_exclusive_gpu_slot),indent=2))
