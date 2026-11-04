"""Explicit diagnostic only: app-preprocessed public VAL blobs vs CUDA FP32."""
import argparse,json,os,sys,time,subprocess,threading,queue
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def worker():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    dll=ROOT/'.venv-training/Lib/site-packages/torch/lib'
    handle=os.add_dll_directory(str(dll))
    sys.path.append(str(ROOT/'.venv-training/Lib/site-packages'))
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import numpy as np,onnxruntime as ort
    session=None;task=None
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request['mode']=='stop':break
            if task!=request['task']:
                session=None
                options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.log_severity_level=3
                providers=[('CUDAExecutionProvider',{'use_tf32':0,'cudnn_conv_algo_search':'HEURISTIC','gpu_mem_limit':2*1024**3}), 'CPUExecutionProvider']
                weights=(ROOT/request['weights']).resolve();assert weights.is_relative_to(ROOT)
                session=ort.InferenceSession(weights.read_bytes(),sess_options=options,providers=providers)
                if session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('cuda_provider_not_active')
                if session.get_provider_options()['CUDAExecutionProvider'].get('use_tf32')!='0':raise RuntimeError('tf32_not_disabled')
                task=request['task']
            blob=(ROOT/request['blob']).resolve();target=(ROOT/request['output']).resolve()
            assert blob.is_relative_to(ROOT) and target.is_relative_to(ROOT)
            value=np.load(blob,allow_pickle=False)
            started=time.perf_counter();raw=session.run(None,{session.get_inputs()[0].name:value})[0];elapsed=time.perf_counter()-started
            if not np.isfinite(raw).all():raise RuntimeError('nonfinite_cuda_output')
            np.save(target,raw,allow_pickle=False)
            print(json.dumps({'status':'ok','ort':ort.__version__,'providers':session.get_providers(),
                'provider_options':session.get_provider_options(),'forward_seconds':elapsed,
                'torch_cuda_libraries':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}),flush=True)
        except Exception as error:
            print(json.dumps({'status':'failed','error_type':type(error).__name__,'reason':str(error)[:200]}),flush=True)


def run(output):
    import numpy as np,cv2,hashlib
    from scripts.benchmark_inference_cpu import MANIFESTS,build,measure,parity,scoped,sha
    from scripts.verify_trained_models import postprocess_detection
    from scripts.setup_performance import environment
    output=scoped(output)
    if output.exists():raise ValueError('new_gpu_probe_output_required')
    output.mkdir(parents=True)
    source=json.loads((ROOT/'data/cpu-inference-benchmark/public-val-v1/private-plan.json').read_text())
    # CPU-only profile is never rewritten or bypassed for supported application load.
    plan={'format':'cuda-diagnostic-public-val-v1','parent_cpu_plan_sha256':sha(ROOT/'data/cpu-inference-benchmark/public-val-v1/private-plan.json'),
        'source_sha256':sha(Path(__file__)),'adapter_sha256':sha(ROOT/'agent/models.py'),
        'cpu_reference':'unchanged guarded ORT1.31.1 application','gpu_candidate':'separate ORT1.20.2 process, CUDA FP32 use_tf32=0',
        'deployment_supported':False,'cpu_manifest_profile_modified':False,'test_or_user_images_read':0,'cases':source['cases']}
    (output/'private-plan.json').write_text(json.dumps(plan,indent=2))
    models=build(MANIFESTS,'baseline')
    logs=(output/'worker.log').open('w',encoding='utf-8')
    process=subprocess.Popen([str(ROOT/'.venv-performance/python.exe'),str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=environment(),
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=logs,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    messages=queue.Queue()
    def read():
        for line in process.stdout:
            try:messages.put(json.loads(line))
            except ValueError:pass
    threading.Thread(target=read,daemon=True).start()
    checks={};metadata={};times={}
    blob=output/'blob.npy';rawfile=output/'raw.npy'
    try:
        for task,rows in source['cases'].items():
            checks[task]=[];times[task]=[]
            model=models[task]
            for index,row in enumerate(rows):
                path=scoped(row['path']);data=path.read_bytes()
                if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('public_source_changed')
                image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                prediction,_=measure(model,image);reference=model.net.last.copy()
                np.save(blob,model.net.blob,allow_pickle=False)
                request={'mode':'run','task':task,'weights':model.manifest['weights'],
                    'blob':blob.relative_to(ROOT).as_posix(),'output':rawfile.relative_to(ROOT).as_posix()}
                process.stdin.write(json.dumps(request)+'\n');process.stdin.flush()
                reply=messages.get(timeout=60)
                if reply['status']!='ok':raise RuntimeError('cuda_worker_failed:'+json.dumps(reply))
                raw=np.load(rawfile,allow_pickle=False)
                if task=='lane':
                    # Reuse the unchanged app's postprocess over this diagnostic raw.
                    from agent.models import OnnxModel
                    class FixedRaw:
                        def setInput(self,value):pass
                        def forward(self):return raw
                    adapter=OnnxModel.__new__(OnnxModel);adapter.manifest=model.manifest;adapter.net=FixedRaw()
                    candidate=adapter.predict(image)
                else:candidate=postprocess_detection(model.manifest,raw,image)
                checks[task].append({'image_sha256':row['sha256'],**parity(task,reference,prediction,raw,candidate)})
                metadata[task]=reply;times[task].append(reply['forward_seconds'])
                blob.unlink();rawfile.unlink()
                if (index+1)%20==0:print(json.dumps({'gpu_diagnostic_task':task,'completed':index+1,'total':len(rows)}),flush=True)
        result={'format':plan['format'],'all_cases_passed':all(c['passed'] for rows in checks.values() for c in rows),
            'results':{task:{'cases':len(rows),'raw_failures':sum(not r['raw']['passed'] for r in rows),
                'post_failures':sum(not r['post']['passed'] for r in rows),'max_raw_abs_error':max(r['raw']['max_abs_error'] for r in rows)} for task,rows in checks.items()},
            'metadata':metadata,'network_only_seconds':{task:{'median':float(np.median(values[3:])),'p95':float(np.percentile(values[3:],95))} for task,values in times.items()},
            'application_preprocessed_blob_unchanged':True,'cpu_profile_modified':False,'deployment_supported':False,'test_or_user_images_read':0}
        (output/'private-checks.json').write_text(json.dumps(checks,indent=2))
        (output/'aggregate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    finally:
        if process.poll() is None:
            try:process.stdin.write('{"mode":"stop"}\n');process.stdin.flush();process.wait(timeout=10)
            except Exception:process.kill();process.wait(timeout=5)
        logs.close();blob.unlink(missing_ok=True);rawfile.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',nargs='?');p.add_argument('--worker',action='store_true');a=p.parse_args()
    worker() if a.worker else run(a.output)
