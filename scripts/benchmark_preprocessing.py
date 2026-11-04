"""Synthetic uint8 preprocessing microbenchmark. No model/weights/network/GPU IO."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics,time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MEAN=[.485,.456,.406];STD=[.229,.224,.225]

def canvas(frame,size,*,lane=False):
    width,height=size;source_h,source_w=frame.shape[:2]
    scale=min(width/source_w,height/source_h)
    resized=cv2.resize(frame,(max(1,round(source_w*scale)),max(1,round(source_h*scale))))
    image=np.full((height,width,3),114,np.uint8);image[:resized.shape[0],:resized.shape[1]]=resized
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB) if lane else image

def finalize(blob):
    # Same finite scan + contiguous marshalling as OrtNetwork.setInput.
    if not np.isfinite(blob).all():raise ValueError('finite_input_required')
    return np.ascontiguousarray(blob)

def baseline(frame,size,task,mean=MEAN,std=STD):
    image=canvas(frame,size,lane=task=='lane')
    if task=='lane':
        image=image.astype(np.float32)/255
        mean=np.array(mean,np.float32);std=np.array(std,np.float32)
        if np.any(std<=0):raise ValueError('invalid_normalization_std')
        blob=((image-mean)/std).transpose(2,0,1)[None].astype(np.float32)
    else:blob=image.transpose(2,0,1)[None].astype(np.float32)
    return finalize(blob)

def candidate(frame,size,task,mean=MEAN,std=STD):
    image=canvas(frame,size,lane=task=='lane')
    if task=='lane':
        image=image.astype(np.float32)
        # Keep division/subtraction/division in the SAME order; never multiply
        # by reciprocal, fuse, change mean precision or relax tolerance.
        image/=255
        mean=np.array(mean,np.float32);std=np.array(std,np.float32)
        if np.any(std<=0):raise ValueError('invalid_normalization_std')
        image-=mean;image/=std
    blob=np.ascontiguousarray(image.transpose(2,0,1)[None],dtype=np.float32)
    return finalize(blob)

def exact(a,b):
    return a.shape==b.shape and a.dtype==b.dtype==np.float32 and a.tobytes(order='C')==b.tobytes(order='C')

def patterns(height,width):
    y,x=np.indices((height,width))
    yield np.stack(((x+y)%256,(x*3+y*5)%256,(x*7+y*11)%256),axis=-1).astype(np.uint8)
    yield np.repeat((((x+y)%2)*255).astype(np.uint8)[...,None],3,axis=2)
    yield np.random.default_rng(20261105).integers(0,256,(height,width,3),dtype=np.uint8)
    yield np.zeros((height,width,3),np.uint8)
    yield np.full((height,width,3),255,np.uint8)

def paired(frame,size,task,blocks=8):
    if type(blocks) is not int or blocks<2:raise ValueError('multiple_paired_blocks_required')
    for _ in range(3):baseline(frame,size,task);candidate(frame,size,task)
    values={'baseline':[],'candidate':[]};deltas=[]
    for _ in range(blocks):
        batch={'baseline':[],'candidate':[]}
        for name in ('baseline','candidate','candidate','baseline'):
            tick=time.perf_counter_ns();result=(baseline if name=='baseline' else candidate)(frame,size,task);elapsed=(time.perf_counter_ns()-tick)/1e6
            batch[name].append(elapsed);values[name].append(elapsed)
            # Materialize output through setInput equivalent. Do not time a
            # subsequent bytes copy or an actual model forward.
            if not result.flags.c_contiguous:raise AssertionError('contiguous_input_required')
            del result
        deltas.append(statistics.mean(batch['baseline'])-statistics.mean(batch['candidate']))
    old=statistics.median(values['baseline']);new=statistics.median(values['candidate'])
    return {'order_per_block':['baseline','candidate','candidate','baseline'],'blocks':blocks,'warmup_calls_each':3,
        'baseline_median_ms':old,'candidate_median_ms':new,'median_saved_ms':old-new,'speedup_ratio':old/new,
        'baseline_min_max_ms':[min(values['baseline']),max(values['baseline'])],'candidate_min_max_ms':[min(values['candidate']),max(values['candidate'])],
        'paired_block_mean_saved_ms':deltas,'candidate_faster_blocks':sum(d>0 for d in deltas),'bitwise_exact':True}

def run(output):
    output=Path(output)
    if not output.is_absolute():output=ROOT/output
    output=output.resolve()
    if output.parent!=ROOT/'docs/reports' or output.exists():raise ValueError('new_public_aggregate_path_required')
    cv2.setNumThreads(2)
    configurations=[('pothole',[640,640]),('redaction',[1920,1920]),('lane',[1024,1024])]
    shapes=[(48,64),(49,65),(720,1280),(1080,1920)]
    checks=0;results=[]
    # Every uint8 scalar is checked before timing, with no tolerance.
    for task,size in configurations:
        for value in range(256):
            frame=np.full((7,9,3),value,np.uint8)
            if not exact(baseline(frame,[32,32],task),candidate(frame,[32,32],task)):raise AssertionError('uint8_scalar_bitwise_mismatch')
            checks+=1
        for height,width in shapes:
            for frame in patterns(height,width):
                if not exact(baseline(frame,size,task),candidate(frame,size,task)):raise AssertionError('full_pattern_bitwise_mismatch')
                checks+=1
            frame=next(patterns(height,width))
            row={'task':task,'source_size':[width,height],'target_size':size,**paired(frame,size,task)}
            results.append(row)
            print(json.dumps({'completed_cases':len(results),'task':task,'target_size':size,'saved_ms':row['median_saved_ms'],'speedup':row['speedup_ratio']}),flush=True)
    report={'format':'synthetic-preprocessing-microbenchmark-v1','source_sha256':hashlib.sha256((ROOT/'agent/models.py').read_bytes()).hexdigest(),
        'benchmark_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'bitwise_equivalence_checks':checks,'all_bitwise_exact':True,
        'numpy_version':np.__version__,'opencv_version':cv2.__version__,'opencv_threads':cv2.getNumThreads(),'results':results,
        'weights_read':0,'model_manifests_read':0,'public_or_user_images_read':0,'raw_network_inferences':0,'GPU_operations':0,'shared_runtime_edited':False,
        'candidate_scope':'Detector direct contiguous float32 cast; lane same-order inplace divide/subtract/divide then contiguous NCHW. Includes unchanged setInput finite check and ascontiguous marshalling.',
        'limits':['Synthetic uint8 CPU microbenchmark only; no inference, CUDA transfer, network postprocess, PNG or JSON storage timing.','Concurrent unrelated GPU training can contend for CPU/memory bandwidth; ABBA paired blocks reduce but do not remove host load effects.','Bitwise equivalence is measured for tested patterns, scalar values and shapes on the recorded NumPy/OpenCV versions.','No changed runtime or end-to-end FPS improvement is claimed.']}
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='docs/reports/PREPROCESSING-MICROBENCH.json');args=parser.parse_args();run(args.output)
