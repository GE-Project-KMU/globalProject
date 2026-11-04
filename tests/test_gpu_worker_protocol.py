"""CPU-only protocol doubles; the explicitly enabled GPU test is separate."""
import hashlib,json,os,subprocess,sys,uuid
from pathlib import Path
from multiprocessing import shared_memory
import numpy as np
import pytest
from agent import gpu_worker_client as gpu


@pytest.fixture
def folder():
    path=gpu.ROOT/'.cache/gpu-protocol-tests'/uuid.uuid4().hex;path.mkdir(parents=True)
    return path


def ready(shape=[1,3,16,16]):
    return {'status':'ready','backend':'onnxruntime-cuda-worker','version':'1.20.2',
        'providers':['CUDAExecutionProvider','CPUExecutionProvider'],'use_tf32':'0','device_capability':[7,5],
        'input_type':'tensor(float)','output_type':'tensor(float)','input_shape':shape,'output_shape':shape,
        'provider_options':{'CUDAExecutionProvider':{'use_tf32':'0','gpu_mem_limit':'2147483648','cudnn_conv_algo_search':'HEURISTIC'}},'python':'3.11.9','cuda_runtime':'12.6','cudnn':90701,'device':'SYNTHETIC PROTOCOL DOUBLE'}


def test_tensor_bounds_and_exact_handshake():
    assert gpu.shape_bytes([1,3,1920,1920])==44236800
    for shape in ([True,3,1,1],[1,3,4096,4096],[0,3,1,1],[-1],['1'],[]):
        with pytest.raises(ValueError):gpu.shape_bytes(shape)
    assert gpu.validate_handshake(ready(),[1,3,16,16])['version']=='1.20.2'
    for changes in ({'providers':['CPUExecutionProvider']},{'use_tf32':'1'},{'version':'1.31.1'},
                    {'device_capability':[8,6]},{'input_type':'tensor(float16)'},{'input_shape':[1,3,32,32]}):
        with pytest.raises(ValueError):gpu.validate_handshake({**ready(),**changes},[1,3,16,16])


def evidence(folder):
    def save(name,value):
        path=folder/name;path.write_text(json.dumps(value));return {'path':path.relative_to(gpu.ROOT).as_posix(),'sha256':gpu.sha256(path)}
    base={'task':'pothole','format':'onnx','classes':['pothole'],'input_size':[640,640],'confidence':.1,
        'weights':'synthetic.onnx','sha256':'a'*64,'research_only':True,'production_approved':False}
    original=save('base.json',base)
    cpu=save('cpu.json',{'model_manifest_sha256':{'pothole':original['sha256']}})
    plan=save('gpu-plan.json',{'parent_cpu_plan_sha256':cpu['sha256']})
    result=save('proof.json',{'all_cases_passed':True,'application_preprocessed_blob_unchanged':True,'cpu_profile_modified':False,
        'results':{'pothole':{'cases':20,'raw_failures':0,'post_failures':0}},
        'metadata':{'pothole':{'ort':'1.20.2','providers':gpu.GPU_PROFILE['providers'],
            'provider_options':{'CUDAExecutionProvider':{'use_tf32':'0'}}}}})
    return {**base,'required_runtime_profile':gpu.GPU_PROFILE,'gpu_compatibility':{'base_manifest':original,'cpu_plan':cpu,'gpu_plan':plan,'aggregate':result}}


def test_child_cannot_change_threshold_or_reuse_changed_evidence(folder):
    child=evidence(folder);assert gpu.validate_gpu_manifest(child)
    with pytest.raises(ValueError,match='changed_model_contract'):gpu.validate_gpu_manifest({**child,'confidence':.35})
    path=gpu.workspace_path(child['gpu_compatibility']['aggregate']['path'])
    path.write_text('{}')
    with pytest.raises(ValueError,match='evidence_hash'):gpu.validate_gpu_manifest(child)


def test_insufficient_or_failed_proof_cannot_create_supported_child(folder):
    child=evidence(folder);entry=child['gpu_compatibility']['aggregate'];path=gpu.workspace_path(entry['path'])
    value=json.loads(path.read_text());value['results']['pothole']['raw_failures']=1
    path.write_text(json.dumps(value));entry['sha256']=gpu.sha256(path)
    with pytest.raises(ValueError,match='strict_validation'):gpu.validate_gpu_manifest(child)


def test_worker_rejects_weight_path_outside_its_workspace_before_gpu_import(folder,monkeypatch):
    from scripts import gpu_inference_worker as worker
    scope=folder/'restricted';scope.mkdir()
    monkeypatch.setattr(worker,'ROOT',scope)
    with pytest.raises(ValueError,match='outside_workspace'):worker.run(folder/'not-in-restricted.onnx','0'*64)


def fake_process(folder,mode):
    path=folder/'fake-worker.py'
    script="""import sys,json,time
import numpy as np
from multiprocessing import shared_memory
print(json.dumps(READY),flush=True)
buffers=[];arrays=[]
try:
 for line in sys.stdin:
  q=json.loads(line)
  if q['mode']=='stop':break
  if q['mode']=='bind':
   for key in ('input','output'):
    m=shared_memory.SharedMemory(name=q[key]);buffers.append(m);arrays.append(np.ndarray((1,3,16,16),np.float32,buffer=m.buf))
   print(json.dumps({'status':'bound','request_id':q['request_id']}),flush=True)
  elif q['mode']=='run':
   if MODE=='crash':sys.exit(3)
   if MODE=='timeout':time.sleep(1)
   np.copyto(arrays[1],arrays[0]+1);print(json.dumps({'status':'completed','request_id':q['request_id']}),flush=True)
finally:
 arrays=[]
 for m in buffers:m.close()
"""
    path.write_text('READY='+repr(ready())+'\nMODE='+repr(mode)+'\n'+script)
    def launch(command,**kwargs):return subprocess.Popen([sys.executable,str(path)],**kwargs)
    return launch


@pytest.mark.parametrize('mode',['normal','crash','timeout'])
def test_persistent_shared_memory_roundtrip_and_failure_cleanup(folder,monkeypatch,mode):
    monkeypatch.setattr(gpu,'validate_gpu_manifest',lambda value:True)
    monkeypatch.setattr(gpu.platform,'system',lambda:'Windows');monkeypatch.setattr(gpu.platform,'machine',lambda:'AMD64')
    fake_python=folder/'fake-python.exe';fake_python.write_bytes(b'synthetic')
    original=gpu.workspace_path
    monkeypatch.setattr(gpu,'workspace_path',lambda value,**kwargs:fake_python if str(value)=='.venv-performance/python.exe' else original(value,**kwargs))
    weights=folder/'fake.onnx';weights.write_bytes(b'SYNTHETIC-NO-ONNX')
    net=gpu.CudaWorkerNetwork(weights,{'sha256':gpu.sha256(weights),'input_size':[16,16]},timeout=5,process_factory=fake_process(folder,mode))
    names=[memory.name for memory in net.buffers]
    value=np.arange(768,dtype=np.float32).reshape(1,3,16,16)
    with pytest.raises(ValueError,match='invalid_gpu_input_blob'):net.setInput(value.astype(np.float64))
    with pytest.raises(ValueError,match='invalid_gpu_input_blob'):net.setInput(value.reshape(1,3,32,8))
    net.setInput(value)
    if mode=='normal':
        assert np.array_equal(net.forward(),value+1)
        net.setInput(value*2);assert np.array_equal(net.forward(),value*2+1)
    else:
        net.timeout=.05
        with pytest.raises(Exception,match='gpu_inference_failed'):net.forward()
    net.close();assert net.closed and net.process.poll() is not None and not net.buffers
    for name in names:
        with pytest.raises(FileNotFoundError):shared_memory.SharedMemory(name=name)


@pytest.mark.skipif(os.getenv('ROAD_RUN_VERIFIED_CUDA')!='1',reason='Requires approved exclusive CUDA slot and local pinned model assets; CPU protocol tests are not GPU support proof.')
def test_actual_opt_in_cuda_profile_smoke():
    from agent.inference_runner import create_models,InferenceRunner,close_models
    from scripts.benchmark_inference_cpu import MANIFESTS,synthetic_frames
    models=create_models(MANIFESTS,'fast',inference_backend='cuda',gpu_profile='data/gpu-runtime-profiles/ort1202-v1.json')
    try:
        with InferenceRunner(models,'fast') as runner:
            result=runner.predict(synthetic_frames()[0])
        assert set(result)=={'pothole','lane','redaction'}
        assert all(model.backend=='onnxruntime-cuda-worker' for model in models.values())
    finally:close_models(models)
