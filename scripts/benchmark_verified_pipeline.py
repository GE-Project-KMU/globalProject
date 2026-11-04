"""Paired warmed CPU/CUDA 2D pipeline timing with explicit inclusion scopes."""
import argparse,json,sys,time,os
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from agent import pipeline
from agent.inference_runner import create_models,InferenceRunner,close_models
from scripts.benchmark_inference_cpu import MANIFESTS,Capture,synthetic_frames,scoped,sha


def run(output,gpu_profile,native_proof,model_set=None):
    output=scoped(output)
    if output.exists():raise ValueError('new_verified_pipeline_output_required')
    proof_path=scoped(native_proof);proof=json.loads(proof_path.read_text())
    manifests=MANIFESTS
    if model_set is None:
        receipt_path=proof_path.parent/'receipt.json';receipt=json.loads(receipt_path.read_text())
        if (proof.get('passed') is not True or proof.get('gpu_node_events_observed') is not True or
            sha(receipt_path)!=proof.get('receipt_sha256') or receipt.get('gpu_profile_sha256')!=sha(gpu_profile)):
            raise ValueError('passing_native_gpu_profile_proof_required')
        if any(sha(ROOT/name)!=digest for name,digest in receipt['source_sha256'].items()):raise ValueError('native_gpu_proof_source_changed')
    else:
        from scripts.gpu_model_set import read,validate_descriptor
        registered=read(gpu_profile);plan=validate_descriptor(registered,True)
        if sha(model_set)!=registered['plan']['sha256'] or sha(native_proof)!=registered['native']['sha256']:raise ValueError('model_set_benchmark_binding_mismatch')
        if any(sha(ROOT/name)!=digest for name,digest in proof['source_sha256'].items()):raise ValueError('native_gpu_proof_source_changed')
        manifests={task:entry['manifest']['path'] for task,entry in plan['models'].items()}
    output.mkdir(parents=True)
    old_profile=os.environ.pop('ROAD_GPU_PROFILE_DIRECTORY',None)
    original={name:getattr(pipeline,name) for name in ('replay','create_models','frame_quality','redact','atomic_json','close_models')}
    original_write=pipeline.cv2.imwrite;original_track=pipeline.Tracker.update
    frame=synthetic_frames()[1];runs=[];hashes=[]
    synthetic_at=datetime.now(timezone.utc);synthetic_expires=(synthetic_at+timedelta(hours=1)).isoformat()
    source_files=['agent/models.py','agent/inference_runner.py','agent/pipeline.py','agent/gpu_worker_client.py','scripts/gpu_inference_worker.py']
    source_sha={name:sha(ROOT/name) for name in source_files}
    try:
        # Primary CPU-default/CUDA pair is ABBA; CPU-fast is supplemental CC.
        order=[('cpu','default'),('cuda','fast'),('cuda','fast'),('cpu','default')]
        if model_set is None:order+=[('cpu','fast'),('cpu','fast')]
        for index,(backend,profile) in enumerate(order):
            start=time.perf_counter();models=create_models(manifests,profile,inference_backend=backend,gpu_profile=gpu_profile if backend=='cuda' else None)
            load=time.perf_counter()-start
            stage={task:[] for task in models};phase={name:[] for name in ('quality','tracking','redaction','png_storage','json_storage','worker_teardown')}
            for task,model in models.items():
                model.net=Capture(model.net);predict=model.predict
                def wrapped(image,task=task,model=model,predict=predict):
                    tick=time.perf_counter();value=predict(image);done=time.perf_counter();net=model.net
                    stage[task].append({'preprocess_and_input_checks':net.forward_at-tick,
                        'forward_transport_and_checks':net.forward_done-net.forward_at,'postprocess':done-net.forward_done,'total':done-tick})
                    return value
                model.predict=wrapped
            start=time.perf_counter()
            with InferenceRunner(models,profile) as runner:
                for _ in range(5):runner.predict(frame)
            warm=time.perf_counter()-start
            for values in stage.values():values.clear()
            def timer(name,fn):
                def wrapped(*a,**k):
                    tick=time.perf_counter()
                    try:return fn(*a,**k)
                    finally:phase[name].append(time.perf_counter()-tick)
                return wrapped
            pipeline.frame_quality=timer('quality',original['frame_quality']);pipeline.redact=timer('redaction',original['redact'])
            pipeline.Tracker.update=timer('tracking',original_track);pipeline.cv2.imwrite=timer('png_storage',original_write)
            pipeline.atomic_json=timer('json_storage',original['atomic_json']);pipeline.close_models=timer('worker_teardown',original['close_models'])
            pipeline.replay=lambda *a,**k:iter([{'frame':frame,'frame_id':i,'pts_seconds':i/10} for i in range(16)])
            pipeline.create_models=lambda *a,**k:models
            tick=time.perf_counter()
            try:
                result=pipeline.analyze_video(output/'synthetic-memory-source',output/f'run-{index}',
                    pothole_manifest=manifests['pothole'],lane_manifest=manifests['lane'],redaction_manifest=manifests['redaction'],cpu_profile=profile,
                    inference_backend=backend,gpu_profile=gpu_profile if backend=='cuda' else None,
                    source_expires_at=synthetic_expires,source_captured_at=synthetic_at.isoformat())
            finally:close_models(models)
            wall=time.perf_counter()-tick;teardown=sum(phase['worker_teardown']);processing=wall-teardown
            hashes.append([row['image']['sha256'] for row in result['observations']])
            runs.append({'backend':backend,'profile':profile,'frames':16,'model_load_seconds':load,'warmup_calls':5,'warmup_seconds':warm,
                'pipeline_wall_including_teardown_seconds':wall,'worker_teardown_seconds':teardown,
                'warmed_processing_seconds_including_png_json':processing,'warmed_processing_fps':16/processing,
                'model_stage_median_ms':{task:{key:float(np.median([v[key] for v in values])*1000) for key in values[0]} for task,values in stage.items()},
                'other_stage_total_ms':{key:sum(values)*1000 for key,values in phase.items()},'runtime':result['runtime_details']})
            print(json.dumps({'completed_pipeline_run':index+1,'backend':backend,'warmed_processing_fps':16/processing}),flush=True)
        png_identity=all(value==hashes[0] for value in hashes)
        assert source_sha=={name:sha(ROOT/name) for name in source_files},'pipeline_source_changed'
        aggregate={'format':'verified-native-cuda-cpu-pipeline-benchmark-v1','source_sha256':source_sha,'runs':runs,
            'opaque_masked_png_bytes_same':png_identity,'native_strict_proof_sha256':sha(proof_path),
            'weights_or_thresholds_changed':False,'user_or_test_images_read':0,
            'input':'Fixed nonpersonal synthetic1280x720; no video decode/capture/clock/GNSS/calibration/API.',
            'warmed_processing_includes':['application preprocessing','GPU shared-memory transport when selected','model inference','postprocess','image quality','pothole tracking','opaque redaction','PNG writes','JSON write'],
            'separately_measured_and_excluded_from_warmed_fps':['session/process/model load','five warmup calls','worker teardown'],
            'decode_measured':False,'ten_minute_sustained_test_completed':False,'ten_fps_field_acceptance':False,
            'whole_system_deployment_throughput_claim':False,'profiling_enabled':False}
        (output/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n');print(json.dumps(aggregate,indent=2))
    finally:
        for name,value in original.items():setattr(pipeline,name,value)
        pipeline.cv2.imwrite=original_write;pipeline.Tracker.update=original_track
        if old_profile is not None:os.environ['ROAD_GPU_PROFILE_DIRECTORY']=old_profile


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');p.add_argument('--gpu-profile',default='data/gpu-runtime-profiles/ort1202-v1.json');p.add_argument('--native-proof',default='data/gpu-inference-probe/native-v1/aggregate.json');p.add_argument('--model-set');a=p.parse_args();run(a.output,a.gpu_profile,a.native_proof,a.model_set)
