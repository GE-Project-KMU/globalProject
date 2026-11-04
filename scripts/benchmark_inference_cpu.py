"""Controlled fixed-weight CPU profiling; synthetic inputs, no user recording I/O."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from agent.models import OnnxModel,OrtNetwork,enforce_runtime_profile
from scripts.verify_trained_models import compare,compare_detections

MANIFESTS={'pothole':'data/training/pothole-nano-rdd2020-v4/model.json',
    'lane':'data/training/lane-a2d2-supported-ort131-v3/lane.json',
    'redaction':'data/training/redaction-inference-options-v4/option-tiled/model.json'}
SETTINGS={'baseline':(2,True),'two_no_spin':(2,False),'four_no_spin':(4,False),'eight_no_spin':(8,False)}


def scoped(value):
    path=Path(value);path=(path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('outside_workspace')
    return path


def sha(path):return hashlib.sha256(scoped(path).read_bytes()).hexdigest()


def synthetic_frames():
    """No source imagery or target GT; raw/postprocess equivalence only."""
    rng=np.random.default_rng(20261105)
    return [np.full((720,1280,3),114,np.uint8),
        rng.integers(0,256,size=(720,1280,3),dtype=np.uint8),
        np.stack(np.meshgrid(np.arange(1920,dtype=np.uint16)%256,np.arange(1080,dtype=np.uint16)%256),axis=-1).repeat(2,axis=2)[:,:,:3].astype(np.uint8)]


class ConfiguredNetwork(OrtNetwork):
    def __init__(self,weights,threads,spinning):
        import onnxruntime as ort
        if type(threads) is not int or not 1<=threads<=8 or type(spinning) is not bool:raise ValueError('invalid_cpu_settings')
        self.backend_version=self.version=ort.__version__
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1;options.log_severity_level=3
        options.add_session_config_entry('session.intra_op.allow_spinning','1' if spinning else '0')
        options.add_session_config_entry('session.inter_op.allow_spinning','1' if spinning else '0')
        self.session=ort.InferenceSession(scoped(weights).read_bytes(),sess_options=options,providers=['CPUExecutionProvider'])
        self.input_name=self.session.get_inputs()[0].name;self.output_name=self.session.get_outputs()[0].name;self.blob=None


class Capture:
    def __init__(self,net):self.net=net;self.last=None
    def __getattr__(self,name):return getattr(self.net,name)
    def setInput(self,value):self.input_at=time.perf_counter();self.net.setInput(value)
    def forward(self):
        self.forward_at=time.perf_counter();self.last=self.net.forward();self.forward_done=time.perf_counter();return self.last


def measure(model,frame):
    started=time.perf_counter();prediction=model.predict(frame);done=time.perf_counter();net=model.net
    return prediction,{'total':done-started,'preprocess_and_input_checks':net.forward_at-started,
        'ort_and_finite_checks':net.forward_done-net.forward_at,'postprocess':done-net.forward_done}


def parity(task,reference_raw,reference_prediction,raw,prediction):
    numeric=compare(reference_raw,raw,.001,.0001)
    if task=='lane':post={'passed':bool(np.array_equal(reference_prediction,prediction)),
        'pixel_agreement':float((reference_prediction==prediction).mean())}
    else:post=compare_detections(reference_prediction,prediction)
    return {'raw':numeric,'post':post,'passed':numeric['passed'] and post['passed']}


def timings(values):
    return {key:{'median_ms':float(np.median([v[key] for v in values])*1000),
        'p95_ms':float(np.percentile([v[key] for v in values],95)*1000)} for key in values[0]}


def build(manifests,setting):
    models={}
    for task,path in manifests.items():
        model=OnnxModel(path)
        if setting!='baseline':
            model.net=ConfiguredNetwork(model.manifest['weights'],*SETTINGS[setting])
            enforce_runtime_profile(model.manifest,model.net)
        model.net=Capture(model.net);models[task]=model
    return models


def run(output):
    output=scoped(output)
    if output.exists():raise ValueError('new_benchmark_directory_required')
    output.mkdir(parents=True)
    snapshots={task:{'manifest_sha256':sha(path),'weights_sha256':json.loads(scoped(path).read_text())['sha256']} for task,path in MANIFESTS.items()}
    source_sha=sha(ROOT/'agent/models.py')
    plan={'format':'controlled-cpu-inference-benchmark-v1','model_sha256':snapshots,'adapter_source_sha256':source_sha,
        'benchmark_source_sha256':sha(Path(__file__)),'synthetic_seed':20261105,
        'parity_cases':3,'warmup_calls':3,'measured_calls':9,'settings':{k:{'intra_op_threads':v[0],'allow_spinning':v[1]} for k,v in SETTINGS.items()},
        'raw_tolerance':{'rtol':.001,'atol':.0001},'test_or_user_images_read':0,'gpu_operations':0,
        'accuracy_evaluated':False,'input_size_or_threshold_changed':False,'pipeline_scope':'Three predict calls only; excludes capture, tracking, quality, redaction, storage, transport.'}
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    frames=synthetic_frames();reference={};results={}
    for setting in SETTINGS:
        models=build(MANIFESTS,setting);checks={};profile={}
        for task,model in models.items():
            checks[task]=[]
            for index,frame in enumerate(frames):
                prediction,_=measure(model,frame);raw=model.net.last.copy()
                if setting=='baseline':reference[task,index]=(raw,prediction)
                checks[task].append(parity(task,*reference[task,index],raw,prediction))
            for _ in range(3):measure(model,frames[1])
            samples=[measure(model,frames[1])[1] for _ in range(9)]
            profile[task]=timings(samples)
        # All three models use distinct sessions; never call one mutable net twice concurrently.
        serial=[];parallel=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for iteration in range(10):
                started=time.perf_counter();predictions={task:models[task].predict(frames[1]) for task in models};serial_elapsed=time.perf_counter()-started
                started=time.perf_counter();futures={task:pool.submit(model.predict,frames[1]) for task,model in models.items()}
                concurrent={task:future.result() for task,future in futures.items()};parallel_elapsed=time.perf_counter()-started
                for task in models:
                    if task=='lane':assert np.array_equal(predictions[task],concurrent[task])
                    else:assert compare_detections(predictions[task],concurrent[task])['passed']
                if iteration>=3:serial.append(serial_elapsed);parallel.append(parallel_elapsed)
        results[setting]={'strict_parity':checks,'all_cases_passed':all(c['passed'] for values in checks.values() for c in values),
            'stage_profile':profile,'three_model_predict':{'serial_median_ms':float(np.median(serial)*1000),
                'parallel_median_ms':float(np.median(parallel)*1000),'serial_p95_ms':float(np.percentile(serial,95)*1000),
                'parallel_p95_ms':float(np.percentile(parallel,95)*1000)},'parallel_post_matches_serial':True}
        print(json.dumps({'completed_cpu_setting':setting,'strict_passed':results[setting]['all_cases_passed'],
            'three_model_predict':results[setting]['three_model_predict']}),flush=True)
        del model,models;gc.collect()
    assert sha(ROOT/'agent/models.py')==source_sha,'adapter_changed_during_benchmark'
    for task,path in MANIFESTS.items():
        assert sha(path)==snapshots[task]['manifest_sha256']
        manifest=json.loads(scoped(path).read_text());assert sha(manifest['weights'])==snapshots[task]['weights_sha256']
    result={'plan':plan,'results':results,'source_and_model_hashes_unchanged':True,
        'ten_fps_field_acceptance':False,'synthetic_latency_only':True}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');a=p.parse_args();run(a.output)
