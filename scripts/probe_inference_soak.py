"""Ten-minute CPU software soak on generated geometry; no field/media evidence."""
from __future__ import annotations
import argparse,gc,json,math,os,statistics,sys,threading,time,weakref
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
import cv2
import numpy as np
from agent.paths import atomic_json,sha256,workspace_path
from agent.models import OnnxModel
from agent import pipeline

MODELS={'pothole':'data/training/pothole-nano-rdd2020-v4/model.json',
        'lane':'data/training/lane-a2d2-supported-ort131-v3/lane.json',
        'redaction':'data/training/redaction-inference-options-v4/option-tiled/model.json'}
OUTPUT='data/training/hardware-free-soak-v6'
DURATION=600.;WARMUP=30.;BUDGET=250*1024*1024

class Deadline:
    def __init__(self,seconds,clock=time.monotonic):
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:raise ValueError('positive_finite_duration_required')
        self.clock=clock;self.started=clock();self.end=self.started+seconds
    def elapsed(self):return max(0.,self.clock()-self.started)
    def active(self):return self.clock()<self.end

def memory_bytes():
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi',use_last_error=True);psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):raise OSError('own_process_memory_query_failed')
        return int(counters.WorkingSetSize),'current_working_set'
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024),'process_high_water_mark_not_current_RSS'

def memory_summary(samples,warmup):
    kept=[row for row in samples if row['elapsed_seconds']>=warmup]
    if not kept:raise ValueError('post_warmup_memory_samples_required')
    if any(not math.isfinite(row['elapsed_seconds']) or type(row['memory_bytes']) is not int or row['memory_bytes']<0 for row in kept):raise ValueError('finite_memory_samples_required')
    values=[row['memory_bytes'] for row in kept]
    return {'warmup_excluded_seconds':warmup,'samples':len(kept),'first_bytes':values[0],'last_bytes':values[-1],
            'min_bytes':min(values),'max_bytes':max(values),'median_bytes':statistics.median(values),'last_minus_first_bytes':values[-1]-values[0],
            'sample_span_seconds':kept[-1]['elapsed_seconds']-kept[0]['elapsed_seconds'],'not_a_proof_of_absent_memory_leaks':True}

def assert_cpu(model):
    if model.backend!='onnxruntime-cpu' or model.net.session.get_providers()!=['CPUExecutionProvider']:raise ValueError('CPU_only_session_required')
    options=model.net.runtime_options
    if options!={'cpu_intra_op_threads':2,'cpu_inter_op_threads':1,'allow_spinning':True}:raise ValueError('default_ORT_two_thread_profile_required')

def original_expiries(value):
    found=[]
    if isinstance(value,dict):
        for key,item in value.items():
            if key in {'expires_at','source_expires_at','collection_expires_at','mask_expires_at','license_expires_at'} and item:
                from datetime import datetime
                moment=datetime.fromisoformat(item.replace('Z','+00:00'))
                if moment.tzinfo is None:raise ValueError('aware_source_expiry_required')
                found.append(moment)
            elif isinstance(item,(dict,list)):found.extend(original_expiries(item))
    elif isinstance(value,list):
        for item in value:found.extend(original_expiries(item))
    return found

def run():
    output=workspace_path(OUTPUT)
    if output!=ROOT/OUTPUT or os.path.lexists(output):raise ValueError('fresh_fixed_private_output_required')
    cv2.setNumThreads(2)
    model_pins={};expiries=[]
    for task,relative in MODELS.items():
        path=workspace_path(relative,exists=True);manifest=json.loads(path.read_text(encoding='utf-8'))
        if manifest['task']!=task or manifest.get('production_approved') is not False:raise ValueError('old_research_manifest_required')
        weights=workspace_path(manifest['weights'],exists=True)
        digest=sha256(weights)
        if digest!=manifest['sha256']:raise ValueError('existing_weight_checksum_mismatch')
        provenance=workspace_path(manifest['training_provenance'],exists=True)
        expiries.extend(original_expiries(json.loads(provenance.read_text(encoding='utf-8'))))
        model_pins[task]={'manifest':relative,'manifest_sha256':sha256(path),'weights':manifest['weights'],'weights_sha256':digest}
    if not expiries:raise ValueError('original_model_source_expiry_required')
    expires=min(expiries).isoformat()
    output.mkdir(parents=True)
    atomic_json(output/'plan.json',{'format':'hardware-free-soak-plan-v1','duration_seconds':DURATION,'warmup_seconds':WARMUP,'output_budget_bytes':BUDGET,
        'models':model_pins,'input_origin':'programmatically_generated_nonpersonal_geometry','input_size':[64,48],'cpu_profile':'default','inference_backend':'cpu','GPU_used':False,'expires_at':expires,'script_sha256':sha256(Path(__file__))})
    image=np.full((48,64,3),105,np.uint8)
    image[::2,::2]=(70,110,135);image[:,22:28]=220
    fixture=output/'generated-geometry.png'
    if not cv2.imwrite(str(fixture),image):raise RuntimeError('fixture_write_failed')
    sessions=[];closed=[];refs=[];memory=[];deadline=None;last_progress=-30.;started_threads={t.ident for t in threading.enumerate()}
    class ObservedModel(OnnxModel):
        def __init__(self,path):
            super().__init__(path,backend='onnxruntime',cpu_threads=2,allow_spinning=True)
            assert_cpu(self);refs.append(weakref.ref(self))
            sessions.append({'task':self.manifest['task'],'providers':self.net.session.get_providers(),'runtime_options':dict(self.net.runtime_options)})
        def close(self):
            closed.append(self.manifest['task']);super().close()
    def generated_samples(source,*,sample_fps):
        nonlocal deadline,last_progress
        if workspace_path(source)!=fixture:raise ValueError('generated_fixture_only')
        deadline=Deadline(DURATION);index=0
        while deadline.active():
            elapsed=deadline.elapsed();rss,kind=memory_bytes()
            memory.append({'elapsed_seconds':elapsed,'memory_bytes':rss})
            if elapsed-last_progress>=30:
                size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
                if size>BUDGET-16*1024*1024:raise RuntimeError('private_output_budget_exceeded')
                atomic_json(output/'progress.json',{'elapsed_seconds':elapsed,'frames_completed':index,'memory_bytes':rss,'memory_kind':kind,'GPU_used':False})
                print(json.dumps({'elapsed_seconds':round(elapsed,1),'frames_completed':index,'memory_MiB':round(rss/2**20,1)}),flush=True)
                last_progress=elapsed
            # Deterministic synthetic motion, never a timestamp from a camera.
            frame=np.roll(image,index%4,axis=0).copy()
            yield {'frame_id':index,'frame':frame,'pts_seconds':float(index)/10}
            index+=1
        rss,kind=memory_bytes();memory.append({'elapsed_seconds':deadline.elapsed(),'memory_bytes':rss})
    try:
        with patch.object(pipeline,'OnnxModel',ObservedModel),patch.object(pipeline,'replay',generated_samples):
            result=pipeline.analyze_video(fixture,output/'analysis',pothole_manifest=MODELS['pothole'],lane_manifest=MODELS['lane'],redaction_manifest=MODELS['redaction'],cpu_profile='default',inference_backend='cpu',source_expires_at=expires)
        gc.collect()
        if deadline is None or deadline.elapsed()<DURATION:raise AssertionError('full_soak_duration_not_reached')
        if sorted(closed)!=sorted(MODELS) or any(ref() is not None for ref in refs):raise AssertionError('owned_model_objects_not_released')
        if {t.ident for t in threading.enumerate()}-started_threads:raise AssertionError('owned_python_threads_not_cleaned')
        images=sorted((output/'analysis').glob('*.redacted.png'))
        if len(images)!=result['analyzed_frames'] or not images:raise AssertionError('saved_output_count_mismatch')
        for row in result['observations']:
            if sha256(row['image']['path'])!=row['image']['sha256']:raise AssertionError('saved_image_hash_mismatch')
            row['expires_at']=expires
        result.update(synthetic_origin=True,input_origin='programmatically_generated_nonpersonal_geometry',mp4_decoded_by_this_probe=False,expires_at=expires)
        atomic_json(output/'analysis/analysis.json',result)
        for task,pin in model_pins.items():
            if sha256(pin['manifest'])!=pin['manifest_sha256'] or sha256(pin['weights'])!=pin['weights_sha256']:raise AssertionError('old_model_changed_during_soak')
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size>BUDGET:raise AssertionError('private_output_budget_exceeded')
        stats=memory_summary(memory,WARMUP)
        aggregate={'format':'hardware-free-inference-soak-v1','status':'completed','planned_seconds':DURATION,'actual_seconds_including_final_JSON':deadline.elapsed(),
            'deadline_semantics':'Stop before next frame after 600 seconds; an in-flight CPU inference and final JSON may finish afterward.',
            'inferred_frames':result['analyzed_frames'],'saved_PNGs':len(images),'output_bytes_before_final_summary':size,'output_budget_bytes':BUDGET,'error_count':0,
            'memory_kind':memory_bytes()[1],'post_warmup_memory':stats,'sessions':sessions,'runtime':result['runtime_details'],'GPU_used':False,'cv2_threads':cv2.getNumThreads(),
            'model_objects_closed_and_released':True,'owned_python_threads_cleaned':True,'subprocesses_created':0,'old_models_unchanged':True,
            'model_weight_sha256':{task:p['weights_sha256'] for task,p in model_pins.items()},'source_expires_at':expires,
            'synthetic_input':True,'input_size':[64,48],'user_recordings_read':False,'public_photographs_read':False,'external_uploads':0,
            'mp4_decoding_test':False,'camera_acquisition_test':False,'field_accuracy_measured':False,'privacy_recall_measured':False,
            'software_synthetic_soak_FPS':result['analyzed_frames']/deadline.elapsed(),
            'limits':['Old research models only, not newly trained candidate verification.','Generated geometry and synthetic timestamps are not camera/GNSS evidence.','No 60-minute camera, 10FPS target or real-road accuracy claim.','CPU throughput includes concurrent unrelated training load; not an isolated benchmark.','Raw source photos/video are never read; generated local PNGs are not shared approvals.']}
        atomic_json(output/'memory-samples.json',memory);atomic_json(output/'summary.json',aggregate)
        return aggregate
    except BaseException as error:
        atomic_json(output/'failure.json',{'status':'failed','error_type':type(error).__name__,'GPU_used':False,'raw_or_user_media_read':False,'files_deleted':0})
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    print(json.dumps(run()),flush=True)

if __name__=='__main__':main()
