"""Warmed real-model 2D pipeline benchmark on nonpersonal synthetic frames."""
import argparse,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from agent import pipeline
from agent.inference_runner import create_models,InferenceRunner
from scripts.benchmark_inference_cpu import MANIFESTS,synthetic_frames,scoped,sha


def run(output):
    output=scoped(output)
    if output.exists():raise ValueError('new_pipeline_benchmark_output_required')
    output.mkdir(parents=True)
    frame=synthetic_frames()[1]
    models={profile:create_models(MANIFESTS,profile) for profile in ('default','fast')}
    for profile,values in models.items():
        with InferenceRunner(values,profile) as runner:
            for _ in range(3):runner.predict(frame)
    original_replay=pipeline.replay;original_create=pipeline.create_models
    results={};hashes={};order=['default','fast','fast','default']
    try:
        for index,profile in enumerate(order):
            pipeline.replay=lambda *a,**k:iter([{'frame':frame,'frame_id':i,'pts_seconds':i/10} for i in range(8)])
            pipeline.create_models=lambda *a,**k:models[profile]
            started=time.perf_counter()
            result=pipeline.analyze_video(output/'synthetic-memory-source',output/f'run-{index}',
                pothole_manifest=MANIFESTS['pothole'],lane_manifest=MANIFESTS['lane'],redaction_manifest=MANIFESTS['redaction'],cpu_profile=profile)
            elapsed=time.perf_counter()-started
            results.setdefault(profile,[]).append({'elapsed_seconds':elapsed,'frames':8,'processing_fps':8/elapsed})
            hashes.setdefault(profile,[]).append([row['image']['sha256'] for row in result['observations']])
    finally:pipeline.replay=original_replay;pipeline.create_models=original_create
    assert all(value==hashes['default'][0] for values in hashes.values() for value in values),'pipeline_image_output_changed'
    aggregate={'format':'warmed-synthetic-2d-pipeline-cpu-v1','order':order,'frames_per_run':8,'warmup_calls_per_profile':3,
        'results':results,'median_processing_fps':{key:float(np.median([row['processing_fps'] for row in values])) for key,values in results.items()},
        'masked_image_bytes_same':True,'models_and_thresholds_changed':False,'gpu_operations':0,'user_or_test_images_read':0,
        'source_sha256':{name:sha(ROOT/name) for name in ('agent/models.py','agent/pipeline.py','agent/inference_runner.py')},
        'scope':'Actual warmed models, quality/tracking/opaque masking/PNG and JSON storage; synthetic in-memory source. Model startup/capture/decode/GNSS/calibration/API excluded.',
        'concurrent_gpu_training_on_host_possible':True,'ten_fps_field_acceptance':False}
    (output/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n');print(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');a=p.parse_args();run(a.output)
