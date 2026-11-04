"""Explicit native shared-memory CUDA validation on pinned public VAL only."""
import argparse,json,hashlib,sys,os,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from agent.inference_runner import create_models,close_models
from scripts.benchmark_inference_cpu import MANIFESTS,Capture,measure,parity,scoped,sha


def run(output,profile):
    output=scoped(output)
    if output.exists():raise ValueError('new_native_gpu_verification_output_required')
    output.mkdir(parents=True)
    source_plan=ROOT/'data/cpu-inference-benchmark/public-val-v1/private-plan.json'
    cases=json.loads(source_plan.read_text())['cases']
    source_files=['agent/models.py','agent/gpu_worker_client.py','scripts/gpu_inference_worker.py']
    source_sha={name:sha(ROOT/name) for name in source_files}
    receipt={'format':'native-shared-memory-cuda-public-val-v1','source_sha256':source_sha,
        'gpu_profile_sha256':sha(profile),'reference_case_plan_sha256':sha(source_plan),
        'case_counts':{task:len(rows) for task,rows in cases.items()},'raw_tolerance':{'rtol':.001,'atol':.0001},
        'model_or_threshold_selection_performed':False,'test_or_user_images_read':0}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2))
    old_profile=os.environ.get('ROAD_GPU_PROFILE_DIRECTORY');os.environ['ROAD_GPU_PROFILE_DIRECTORY']=str(output/'operator-profiles')
    baseline={};candidate={};checks={};metadata={}
    try:
        baseline=create_models(MANIFESTS)
        candidate=create_models(MANIFESTS,'fast',inference_backend='cuda',gpu_profile=profile)
        for models in (baseline,candidate):
            for model in models.values():model.net=Capture(model.net)
        for task,rows in cases.items():
            values=[]
            for index,row in enumerate(rows):
                if datetime.fromisoformat(row['expires_at'])<=datetime.now(timezone.utc):raise ValueError('source_expired')
                raw_bytes=scoped(row['path']).read_bytes()
                if hashlib.sha256(raw_bytes).hexdigest()!=row['sha256']:raise ValueError('public_validation_source_changed')
                image=cv2.imdecode(np.frombuffer(raw_bytes,np.uint8),cv2.IMREAD_COLOR)
                if image is None:raise ValueError('image_decode_failed')
                old,_=measure(baseline[task],image);raw=baseline[task].net.last.copy()
                new,_=measure(candidate[task],image)
                if not np.array_equal(baseline[task].net.blob,candidate[task].net.blob):raise ValueError('application_preprocessing_blob_changed')
                values.append({'image_sha256':row['sha256'],**parity(task,raw,old,candidate[task].net.last,new)})
                if (index+1)%20==0:print(json.dumps({'native_gpu_task':task,'completed':index+1,'total':len(rows)}),flush=True)
            checks[task]=values
            metadata[task]={'backend':candidate[task].backend,'version':candidate[task].backend_version,
                'runtime_options':candidate[task].net.runtime_options}
            (output/(task+'-private-checks.json')).write_text(json.dumps(values,indent=2))
    finally:
        close_models(candidate);close_models(baseline)
        if old_profile is None:os.environ.pop('ROAD_GPU_PROFILE_DIRECTORY',None)
        else:os.environ['ROAD_GPU_PROFILE_DIRECTORY']=old_profile
    placements={}
    for file in (output/'operator-profiles').glob('*.json'):
        for event in json.loads(file.read_text()):
            provider=event.get('args',{}).get('provider')
            if provider:placements[provider]=placements.get(provider,0)+1
    assert source_sha=={name:sha(ROOT/name) for name in source_files},'native_gpu_source_changed'
    result={'format':receipt['format'],'passed':all(value['passed'] for rows in checks.values() for value in rows),
        'results':{task:{'cases':len(rows),'raw_failures':sum(not r['raw']['passed'] for r in rows),
            'post_failures':sum(not r['post']['passed'] for r in rows),'max_raw_abs_error':max(r['raw']['max_abs_error'] for r in rows)} for task,rows in checks.items()},
        'actual_runtime':metadata,'profiled_node_events_by_provider':placements,'gpu_node_events_observed':placements.get('CUDAExecutionProvider',0)>0,
        'all_nodes_claimed_gpu':False,'application_preprocess_and_postprocess_unchanged':True,
        'receipt_sha256':sha(output/'receipt.json'),'test_or_user_images_read':0,'accuracy_measured':False,
        'ten_fps_acceptance':False,'timings_are_not_benchmarked_here':True}
    (output/'aggregate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');p.add_argument('--gpu-profile',default='data/gpu-runtime-profiles/ort1202-v1.json');a=p.parse_args();run(a.output,a.gpu_profile)
