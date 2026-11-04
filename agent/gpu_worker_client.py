"""Opt-in, evidence-bound Windows CUDA worker; blobs stay in shared memory."""
from __future__ import annotations
import hashlib,json,os,platform,queue,subprocess,sys,threading,uuid
from pathlib import Path
from multiprocessing import shared_memory
import numpy as np
from .paths import ROOT,workspace_path,sha256

GPU_PROFILE={'engine':'onnxruntime-cuda-worker','version':'1.20.2','system':'Windows','machine':'AMD64',
             'providers':['CUDAExecutionProvider','CPUExecutionProvider'],'use_tf32':'0','device_capability':[7,5]}
MAX_BYTES=64*1024*1024


def evidence_document(item):
    if not isinstance(item,dict) or not isinstance(item.get('path'),str):raise ValueError('gpu_evidence_pointer_required')
    path=workspace_path(item['path'],exists=True)
    if not path.is_file() or path.stat().st_size>4*1024*1024 or sha256(path)!=item.get('sha256'):raise ValueError('gpu_evidence_hash_mismatch')
    return json.loads(path.read_text(encoding='utf-8'))


def validate_proof_binding(evidence,task,base):
    binding=evidence_document(evidence.get('binding'))
    if binding.get('format')!='native-gpu-proof-model-binding-v1':raise ValueError('gpu_model_binding_required')
    for name in ('aggregate','gpu_plan','cpu_plan'):
        if binding.get(name+'_sha256')!=evidence[name]['sha256']:raise ValueError('gpu_bound_proof_mismatch')
    native=evidence_document(binding.get('native_aggregate'))
    receipt=evidence_document(binding.get('native_receipt'))
    executed=evidence_document(binding.get('executed_profile'))
    if (native.get('passed') is not True or native.get('gpu_node_events_observed') is not True or
        native.get('application_preprocess_and_postprocess_unchanged') is not True or
        native.get('receipt_sha256')!=binding['native_receipt']['sha256'] or
        receipt.get('gpu_profile_sha256')!=binding['executed_profile']['sha256']):raise ValueError('gpu_native_receipt_binding_mismatch')
    entry=executed.get('models',{}).get(task,{})
    identities=binding.get('validated_models',{}).get(task,{})
    if (entry.get('base_manifest_sha256')!=evidence['base_manifest']['sha256'] or
        identities.get('base_manifest_sha256')!=evidence['base_manifest']['sha256'] or
        identities.get('weights_sha256')!=base.get('sha256') or
        identities.get('executed_child_sha256')!=entry.get('manifest_sha256')):raise ValueError('gpu_native_model_identity_mismatch')
    child=evidence_document({'path':entry.get('manifest'),'sha256':entry.get('manifest_sha256')})
    expected={k:v for k,v in base.items() if k!='required_runtime_profile'}
    if {k:v for k,v in child.items() if k not in ('required_runtime_profile','gpu_compatibility')}!=expected:raise ValueError('gpu_executed_model_contract_mismatch')
    result=native.get('results',{}).get(task,{})
    minimum={'lane':100,'pothole':20,'redaction':40}[task]
    if (type(result.get('cases')) is not int or result['cases']<minimum or
        type(result.get('raw_failures')) is not int or result['raw_failures']!=0 or
        type(result.get('post_failures')) is not int or result['post_failures']!=0):raise ValueError('gpu_native_strict_proof_required')


def shape_bytes(shape):
    if not isinstance(shape,(list,tuple)) or not shape or len(shape)>4 or any(type(v) is not int or v<1 or v>1_000_000 for v in shape):raise ValueError('invalid_gpu_tensor_shape')
    size=4
    for value in shape:size*=value
    if size>MAX_BYTES:raise ValueError('gpu_tensor_size_limit')
    return size


def validate_gpu_manifest(manifest):
    if manifest.get('required_runtime_profile')!=GPU_PROFILE:raise ValueError('verified_gpu_child_profile_required')
    evidence=manifest.get('gpu_compatibility')
    if not isinstance(evidence,dict):raise ValueError('gpu_evidence_required')
    if evidence.get('format')=='frozen-model-set-diagnostic-v1':
        from scripts.gpu_model_set import validate_staged_child
        return validate_staged_child(manifest)
    documents={}
    for name in ('base_manifest','aggregate','gpu_plan','cpu_plan'):
        documents[name]=evidence_document(evidence.get(name))
    base=documents['base_manifest'];task=base.get('task')
    expected={k:v for k,v in base.items() if k!='required_runtime_profile'}
    actual={k:v for k,v in manifest.items() if k not in ('required_runtime_profile','gpu_compatibility')}
    if expected!=actual:raise ValueError('gpu_child_changed_model_contract')
    proof=documents['aggregate'];plan=documents['gpu_plan'];cpu=documents['cpu_plan']
    minimum={'lane':100,'pothole':20,'redaction':40}
    result=proof.get('results',{}).get(task,{})
    metadata=proof.get('metadata',{}).get(task,{})
    if (task not in minimum or proof.get('all_cases_passed') is not True or
        proof.get('application_preprocessed_blob_unchanged') is not True or proof.get('cpu_profile_modified') is not False or
        type(result.get('cases')) is not int or result.get('cases',0)<minimum[task] or
        type(result.get('raw_failures')) is not int or type(result.get('post_failures')) is not int or
        result.get('raw_failures')!=0 or result.get('post_failures')!=0):raise ValueError('gpu_strict_validation_required')
    if metadata.get('ort')!='1.20.2' or metadata.get('providers')!=GPU_PROFILE['providers'] or metadata.get('provider_options',{}).get('CUDAExecutionProvider',{}).get('use_tf32')!='0':raise ValueError('gpu_proof_runtime_mismatch')
    if (plan.get('parent_cpu_plan_sha256')!=evidence['cpu_plan']['sha256'] or
        cpu.get('model_manifest_sha256',{}).get(task)!=evidence['base_manifest']['sha256']):raise ValueError('gpu_proof_model_binding_mismatch')
    if manifest.get('production_approved') is not False or manifest.get('research_only') is not True:raise ValueError('gpu_research_only_required')
    validate_proof_binding(evidence,task,base)
    shape_bytes([1,3,manifest['input_size'][1],manifest['input_size'][0]])
    return evidence


def validate_handshake(value,input_shape):
    if not isinstance(value,dict) or value.get('status')!='ready':raise ValueError('gpu_worker_handshake_failed')
    expected={'backend':'onnxruntime-cuda-worker','version':'1.20.2','providers':GPU_PROFILE['providers'],
              'use_tf32':'0','device_capability':[7,5],'input_type':'tensor(float)','output_type':'tensor(float)',
              'python':'3.11.9','cuda_runtime':'12.6','cudnn':90701,'device':'NVIDIA GeForce RTX 2080 Ti'}
    if any(value.get(key)!=wanted for key,wanted in expected.items()) or value.get('input_shape')!=input_shape:raise ValueError('gpu_worker_runtime_mismatch')
    actual=value.get('provider_options',{}).get('CUDAExecutionProvider',{})
    if any(actual.get(key)!=wanted for key,wanted in {'use_tf32':'0','gpu_mem_limit':'2147483648','cudnn_conv_algo_search':'HEURISTIC'}.items()):raise ValueError('gpu_worker_provider_options_mismatch')
    shape_bytes(value.get('output_shape'));return value


def gpu_bundle_manifests(manifests,profile_path):
    path=workspace_path(profile_path,exists=True)
    if not path.is_relative_to((ROOT/'data/gpu-runtime-profiles').resolve()):raise ValueError('registered_local_gpu_profile_required')
    profile=json.loads(path.read_text(encoding='utf-8'))
    if profile.get('format')=='registered-cuda-model-set-v1':
        from scripts.gpu_model_set import validate_descriptor
        plan=validate_descriptor(profile,True)
        if set(profile['models'])!=set(manifests):raise ValueError('gpu_bundle_model_set_mismatch')
        for task,base_path in manifests.items():
            if sha256(base_path)!=plan['models'][task]['manifest']['sha256']:raise ValueError('gpu_new_model_requires_revalidation')
        return {task:workspace_path(item['manifest'],exists=True) for task,item in profile['models'].items()}
    if profile.get('format')!='verified-local-cuda-bundle-v1' or set(profile.get('models',{}))!=set(manifests):raise ValueError('gpu_bundle_model_set_mismatch')
    result={}
    for task,base_path in manifests.items():
        entry=profile['models'][task];child=workspace_path(entry['manifest'],exists=True)
        if sha256(base_path)!=entry['base_manifest_sha256'] or sha256(child)!=entry['manifest_sha256']:raise ValueError('gpu_bundle_manifest_changed')
        value=json.loads(child.read_text(encoding='utf-8'))
        if value.get('task')!=task:raise ValueError('gpu_bundle_task_mismatch')
        validate_gpu_manifest(value);result[task]=child
    return result


class RemoteSession:
    def __init__(self,providers):self.providers=list(providers)
    def get_providers(self):return list(self.providers)


class CudaWorkerNetwork:
    backend='onnxruntime-cuda-worker';backend_version=version='1.20.2'
    def __init__(self,weights,manifest,*,timeout=60,process_factory=subprocess.Popen):
        self.process=None;self.buffers=[];self.closed=False;self.blob=None;self.lock=threading.RLock();self.timeout=timeout;self.request_id=0
        validate_gpu_manifest(manifest)
        if platform.system()!='Windows' or platform.machine()!='AMD64':raise ValueError('gpu_verified_host_mismatch')
        python=workspace_path('.venv-performance/python.exe',exists=True)
        weights=workspace_path(weights,exists=True)
        if sha256(weights)!=manifest['sha256']:raise ValueError('gpu_weight_digest_mismatch')
        self.input_shape=[1,3,manifest['input_size'][1],manifest['input_size'][0]]
        shape_bytes(self.input_shape)
        from scripts.setup_performance import environment
        env=environment();env['CUDA_VISIBLE_DEVICES']='0'
        command=[str(python),str(ROOT/'scripts/gpu_inference_worker.py'),'--weights',str(weights),'--sha256',manifest['sha256']]
        try:
            self.process=process_factory(command,cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.messages=queue.Queue()
            def read():
                try:
                    for line in self.process.stdout:
                        if len(line)>65536:self.messages.put({'status':'protocol_error'});break
                        try:self.messages.put(json.loads(line))
                        except ValueError:self.messages.put({'status':'protocol_error'});break
                finally:self.messages.put({'status':'worker_exited'})
            self.reader=threading.Thread(target=read,daemon=True);self.reader.start()
            ready=validate_handshake(self._receive(),self.input_shape);self.output_shape=ready['output_shape']
            self.runtime_options={'use_tf32':False,'provider_options':ready['provider_options'],'worker_python':ready['python'],
                'device':ready['device'],'device_capability':ready['device_capability'],'node_placement_verified':False,
                'cuda_runtime':ready['cuda_runtime'],'cudnn':ready['cudnn'],
                'transport':'persistent-process-shared-memory','cpu_fallback_only_permitted':False}
            self.session=RemoteSession(ready['providers'])
            self.buffers=[shared_memory.SharedMemory(name='roadwatch_gpu_'+uuid.uuid4().hex,create=True,size=shape_bytes(shape)) for shape in (self.input_shape,self.output_shape)]
            self.input=np.ndarray(self.input_shape,dtype=np.float32,buffer=self.buffers[0].buf)
            self.output=np.ndarray(self.output_shape,dtype=np.float32,buffer=self.buffers[1].buf)
            reply=self._request({'mode':'bind','input':self.buffers[0].name,'output':self.buffers[1].name})
            if reply.get('status')!='bound':raise ValueError('gpu_buffer_bind_failed')
        except BaseException:
            self.close();raise
    def _receive(self):
        try:return self.messages.get(timeout=self.timeout)
        except queue.Empty:raise RuntimeError('gpu_worker_timeout')
    def _request(self,value):
        if self.closed or self.process.poll() is not None:raise RuntimeError('gpu_worker_exited')
        self.request_id+=1
        self.process.stdin.write(json.dumps({**value,'request_id':self.request_id})+'\n');self.process.stdin.flush()
        response=self._receive()
        if response.get('request_id')!=self.request_id:raise RuntimeError('gpu_worker_reply_out_of_order')
        return response
    def setInput(self,value):
        self.blob=None
        if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or list(value.shape)!=self.input_shape or not np.isfinite(value).all():raise ValueError('invalid_gpu_input_blob')
        self.blob=np.ascontiguousarray(value)
    def forward(self):
        from .models import ModelNotReady
        with self.lock:
            try:
                if self.blob is None:raise RuntimeError('gpu_input_required')
                np.copyto(self.input,self.blob)
                reply=self._request({'mode':'run'})
                if reply.get('status')!='completed':raise RuntimeError('gpu_worker_forward_failed')
                result=self.output.copy()
                if not np.isfinite(result).all():raise RuntimeError('nonfinite_gpu_output')
                return result
            except Exception as error:
                self.close();raise ModelNotReady(type(error).__name__+':gpu_inference_failed') from error
    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True
            if self.process:
                if self.process.poll() is None:
                    try:self.process.stdin.write('{"mode":"stop"}\n');self.process.stdin.flush();self.process.wait(timeout=2)
                    except Exception:self.process.kill();self.process.wait(timeout=5)
                for stream in (self.process.stdin,self.process.stdout):
                    if stream:
                        try:stream.close()
                        except Exception:pass
            self.input=None;self.output=None
            for memory in self.buffers:
                try:memory.close();memory.unlink()
                except FileNotFoundError:pass
            self.buffers=[]
    def __del__(self):
        try:self.close()
        except Exception:pass
