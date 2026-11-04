"""Explicit new-artifact CUDA admission: freeze -> diagnose -> stage -> native -> register.

No phase inherits GPU support from a different weight/manifest/case-set identity.
All data references and individual case identities stay in ignored local plans.
"""
from __future__ import annotations
import argparse,hashlib,json,os,queue,subprocess,sys,threading,time,uuid
import platform
from datetime import datetime,timezone
from pathlib import Path
from multiprocessing import shared_memory
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256
from agent.gpu_worker_client import GPU_PROFILE,MAX_BYTES,shape_bytes,validate_handshake,CudaWorkerNetwork,RemoteSession
TASKS=('pothole','lane','redaction');COUNTS={'pothole':20,'lane':100,'redaction':40}
PUBLIC_ROOTS=(ROOT/'data/public',ROOT/'data/research')
REGISTRY_ROOT=ROOT/'data/gpu-runtime-profiles'
RUNTIME_SOURCES=('agent/models.py','agent/gpu_worker_client.py','scripts/gpu_inference_worker.py',
    'scripts/gpu_model_set.py','scripts/verify_trained_models.py','scripts/benchmark_inference_cpu.py')


def public_case_path(value):
    path=workspace_path(value)
    if any(path.is_relative_to(root.resolve()) for root in PUBLIC_ROOTS):return path
    training=(ROOT/'data/training').resolve()
    if path.is_relative_to(training) and path.relative_to(training).parts[0].startswith('research-'):return path
    raise ValueError('public_research_case_namespace_required')


def read(path):
    path=workspace_path(path,exists=True)
    if not path.is_file() or path.stat().st_size>40_000_000:raise ValueError('bounded_json_required')
    return json.loads(path.read_text(encoding='utf-8'))


def profile_events(path,limit=128*1024*1024):
    path=workspace_path(path,exists=True)
    if not path.is_file() or path.stat().st_size>limit:raise ValueError('bounded_operator_profile_required')
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,list):raise ValueError('operator_profile_array_required')
    return value


def pointer(path):
    path=workspace_path(path,exists=True)
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path)}


def pinned(item):
    if not isinstance(item,dict) or not isinstance(item.get('path'),str):raise ValueError('pinned_pointer_required')
    path=workspace_path(item['path'],exists=True)
    if sha256(path)!=item.get('sha256'):raise ValueError('pinned_file_changed')
    return read(path)


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=workspace_path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def new_directory(path):
    path=workspace_path(path)
    if path.exists():raise ValueError('new_phase_output_required')
    path.mkdir(parents=True);return path


def proof_passed(proof,manifest,manifest_sha):
    from scripts.verify_trained_models import finalize_verification
    derived=finalize_verification(dict(proof),manifest['task'])
    if (proof.get('manifest_sha256')!=manifest_sha or proof.get('weight_sha256')!=manifest['sha256'] or
        proof.get('task')!=manifest['task'] or proof.get('synthetic_input') is not False or
        proof.get('torch_checkpoint_compared') is not True or
        any(proof.get(key) is not True for key in ('artifact_contract_verified','numerical_parity_passed','postprocess_parity_passed'))):
        raise ValueError('exact_positive_public_cpu_canonical_pass_required')
    if (derived['artifact_contract_verified'] is not True or proof.get('app_backend')!='onnxruntime-cpu' or
        proof.get('app_backend_version')!='1.31.1' or any(proof.get(k) is not True for k in
        ('metadata_valid','weight_hash_verified','finite_output','preprocess_contract_verified'))):raise ValueError('full_guarded_cpu131_canonical_required')
    if manifest['task']!='lane' and any(proof.get('detection_postprocess_parity',{}).get(mode,{}).get('passed') is not True for mode in ('ort','torch')):
        raise ValueError('cpu_detection_postprocess_reference_required')
    if not isinstance(proof.get('image_sha256'),str):raise ValueError('canonical_image_identity_required')
    if type(proof.get('adapter_result_count')) is not int or proof['adapter_result_count']<=0:raise ValueError('positive_cpu_canonical_output_required')
    if any(proof[key].get('rtol')!=.001 or proof[key].get('atol')!=.0001 for key in ('app_repeatability','app_vs_ort','app_vs_torch','ort_vs_torch')):raise ValueError('cpu_canonical_default_tolerance_required')
    shape_bytes([1,3,manifest['input_size'][1],manifest['input_size'][0]])
    shape_bytes(proof.get('output_shape'))


def select_cases(task,rows,canonical_images,explicit=None):
    pool=sorted((row for row in rows if row.get('split')=='validation'),key=lambda row:(row.get('source_id',''),row['sample_id']))
    if explicit is not None:
        if not isinstance(explicit,list) or len(explicit)!=COUNTS[task] or len(set(explicit))!=COUNTS[task]:raise ValueError('explicit_case_count_or_duplicates')
        lookup={row['sha256']:row for row in pool}
        if any(value not in lookup for value in explicit):raise ValueError('explicit_case_not_validation')
        selected=[lookup[value] for value in explicit]
        if not canonical_images<=set(explicit):raise ValueError('explicit_cases_omit_canonical')
        if task=='lane' and (len(pool)!=100 or set(explicit)!=set(lookup)):raise ValueError('exact_lane_validation100_required')
        if task=='pothole' and any(not any(obj.get('class')=='pothole' for obj in row.get('objects',[])) for row in selected):raise ValueError('positive_pothole_cases_required')
        if task=='redaction' and any(sum(row.get('evaluation_class')==label and any(obj.get('class')==label for obj in row.get('objects',[])) for row in selected)!=20 for label in ('face','license_plate')):raise ValueError('balanced_privacy_cases_required')
        if task=='lane' and any(row['sha256'] in canonical_images and row.get('positive_pixels',0)<=0 for row in selected):raise ValueError('positive_canonical_lane_ground_truth_required')
        return selected
    if task=='lane':
        if len(pool)!=100:raise ValueError('exact_lane_validation100_required')
        if any(row['sha256'] in canonical_images and row.get('positive_pixels',0)<=0 for row in pool):raise ValueError('positive_canonical_lane_ground_truth_required')
        selected=pool
    else:
        pool=[row for row in pool if row.get('objects')]
        groups={'pothole':20} if task=='pothole' else {'face':20,'license_plate':20}
        selected=[]
        for label,count in groups.items():
            available=[row for row in pool if (task=='pothole' or row.get('evaluation_class')==label)
                and any(item.get('class')==label for item in row['objects'])]
            required=[row for row in available if row['sha256'] in canonical_images]
            if len(required)>count:raise ValueError('too_many_canonical_cases')
            chosen=required+[row for row in available if row not in required][:count-len(required)]
            if len(chosen)!=count:raise ValueError('insufficient_positive_validation_cases')
            selected+=chosen
    if not canonical_images<={row['sha256'] for row in selected}:raise ValueError('canonical_case_not_in_frozen_validation')
    if len({row['sha256'] for row in selected})!=COUNTS[task]:raise ValueError('duplicate_validation_case')
    return selected


def validate_plan(plan):
    if plan.get('format')!='frozen-cuda-model-set-v1' or plan.get('platform')!=GPU_PROFILE or set(plan.get('models',{}))!=set(TASKS):raise ValueError('frozen_model_set_format_mismatch')
    pinned(plan['input_specification'])
    if plan.get('planning_source_sha256')!={name:sha256(ROOT/name) for name in RUNTIME_SOURCES}:raise ValueError('frozen_model_set_source_changed')
    if plan.get('raw_tolerance')!={'rtol':.001,'atol':.0001}:raise ValueError('default_gpu_tolerance_required')
    for task,entry in plan['models'].items():
        manifest=pinned(entry['manifest'])
        if manifest.get('task')!=task or manifest.get('production_approved') is not False or manifest.get('research_only') is not True:raise ValueError('research_model_task_required')
        if sha256(manifest['weights'])!=entry['weights_sha256'] or manifest['sha256']!=entry['weights_sha256']:raise ValueError('frozen_model_weight_changed')
        canonical_images=set()
        for proof_ref in entry['cpu_verifications']:
            proof=pinned(proof_ref);proof_passed(proof,manifest,entry['manifest']['sha256']);canonical_images.add(proof['image_sha256'])
        if not canonical_images:raise ValueError('cpu_canonical_required_before_gpu')
        rows=pinned(entry['validation_manifest']);selected=select_cases(task,rows,canonical_images,entry.get('case_sha256s'))
        if manifest.get('dataset_release')!='manifest-sha256:'+sha256(manifest['source_release_path']):raise ValueError('model_source_release_hash_mismatch')
        source_rows=read(manifest['source_release_path'])
        if not isinstance(source_rows,list):raise ValueError('full_source_release_required_for_case_binding')
        source_validation={}
        for row in source_rows:
            if row.get('split')=='validation':source_validation.setdefault(row['sha256'],[]).append(row)
        expected=[{key:row[key] for key in ('path','sha256','expires_at','sample_id')}|
            ({'evaluation_class':row['evaluation_class']} if task=='redaction' else {}) for row in selected]
        if entry['cases']!=expected or entry['case_set_sha256']!=canonical(expected):raise ValueError('frozen_validation_case_set_changed')
        for row in expected:
            public_case_path(row['path'])
            expiry=datetime.fromisoformat(row['expires_at'].replace('Z','+00:00'))
            if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc):raise ValueError('public_validation_source_expired')
            parents=source_validation.get(row['sha256'],[])
            if not parents or expiry>min(datetime.fromisoformat(parent['expires_at'].replace('Z','+00:00')) for parent in parents):raise ValueError('validation_case_not_bound_to_parent_release_or_expiry')
    return plan


class FrozenDiagnosticNetwork(CudaWorkerNetwork):
    """Shared transport for a CPU-approved frozen trial, never an app profile."""
    backend='cuda-frozen-diagnostic-unapproved'
    def __init__(self,plan_path,plan_sha,task):
        import numpy as np
        from scripts.setup_performance import environment
        if sha256(plan_path)!=plan_sha:raise ValueError('diagnostic_plan_changed')
        plan=validate_plan(read(plan_path));manifest=pinned(plan['models'][task]['manifest'])
        if platform.system()!='Windows' or platform.machine()!='AMD64':raise ValueError('diagnostic_host_mismatch')
        self.process=None;self.buffers=[];self.closed=False;self.blob=None;self.lock=threading.RLock();self.timeout=60;self.request_id=0
        self.input_shape=[1,3,manifest['input_size'][1],manifest['input_size'][0]];shape_bytes(self.input_shape)
        python=workspace_path('.venv-performance/python.exe',exists=True)
        command=[str(python),str(ROOT/'scripts/gpu_inference_worker.py'),'--weights',str(workspace_path(manifest['weights'],exists=True)),'--sha256',manifest['sha256']]
        try:
            env=environment();env['CUDA_VISIBLE_DEVICES']='0'
            self.process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.messages=queue.Queue()
            def consume():
                try:
                    for line in self.process.stdout:
                        if len(line)>65536:self.messages.put({'status':'protocol_error'});break
                        try:self.messages.put(json.loads(line))
                        except ValueError:self.messages.put({'status':'protocol_error'});break
                finally:self.messages.put({'status':'worker_exited'})
            self.reader=threading.Thread(target=consume,daemon=True);self.reader.start()
            ready=validate_handshake(self._receive(),self.input_shape);self.output_shape=ready['output_shape']
            self.runtime_options={key:ready[key] for key in ('version','providers','provider_options','device','device_capability','python','cuda_runtime','cudnn')}
            self.runtime_options['deployment_supported']=False;self.runtime_options['use_tf32']=False
            self.session=RemoteSession(ready['providers'])
            self.buffers=[shared_memory.SharedMemory(name='roadwatch_gpu_'+uuid.uuid4().hex,create=True,size=shape_bytes(shape)) for shape in (self.input_shape,self.output_shape)]
            self.input=np.ndarray(self.input_shape,np.float32,buffer=self.buffers[0].buf);self.output=np.ndarray(self.output_shape,np.float32,buffer=self.buffers[1].buf)
            if self._request({'mode':'bind','input':self.buffers[0].name,'output':self.buffers[1].name}).get('status')!='bound':raise ValueError('diagnostic_bind_failed')
        except BaseException:self.close();raise


def postprocess(manifest,raw,image):
    from agent.models import OnnxModel
    class FixedRaw:
        def setInput(self,value):pass
        def forward(self):return raw
    model=OnnxModel.__new__(OnnxModel);model.manifest=manifest;model.net=FixedRaw()
    return model.predict(image)


def run_validation(plan_path,output,staging_path=None):
    import cv2,numpy as np
    from agent.models import OnnxModel
    from agent.inference_runner import close_models
    from scripts.benchmark_inference_cpu import Capture,measure,parity
    plan=validate_plan(read(plan_path));plan_sha=sha256(plan_path)
    descriptor=read(staging_path) if staging_path else None
    if descriptor:
        validate_descriptor(descriptor)
        if descriptor['plan']['sha256']!=plan_sha:raise ValueError('staged_plan_mismatch')
    output=new_directory(output);phase='native' if descriptor else 'diagnostic'
    before={name:sha256(ROOT/name) for name in RUNTIME_SOURCES}
    old_profiling=os.environ.get('ROAD_GPU_PROFILE_DIRECTORY');os.environ['ROAD_GPU_PROFILE_DIRECTORY']=str(output/'operator-profiles')
    baseline={};candidate={};details={};actual={}
    try:
        # All original CPU manifests/provenance must load before any CUDA trial.
        for task,entry in plan['models'].items():
            baseline[task]=OnnxModel(entry['manifest']['path']);baseline[task].net=Capture(baseline[task].net)
        for task,entry in plan['models'].items():
            os.environ['ROAD_GPU_PROFILE_DIRECTORY']=str(output/'operator-profiles'/task)
            if descriptor:
                candidate[task]=OnnxModel(descriptor['models'][task]['manifest'],backend='onnxruntime-cuda');candidate[task].net=Capture(candidate[task].net)
            else:candidate[task]=FrozenDiagnosticNetwork(plan_path,plan_sha,task)
            checks=[]
            for index,row in enumerate(entry['cases']):
                if datetime.fromisoformat(row['expires_at'])<=datetime.now(timezone.utc):raise ValueError('validation_source_expired')
                data=workspace_path(row['path'],exists=True).read_bytes()
                if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('public_validation_image_changed')
                image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                if image is None:raise ValueError('public_validation_decode_failed')
                reference,_=measure(baseline[task],image);raw=baseline[task].net.last.copy()
                if descriptor:
                    prediction,_=measure(candidate[task],image);gpu_raw=candidate[task].net.last
                    if not np.array_equal(baseline[task].net.blob,candidate[task].net.blob):raise ValueError('application_blob_changed')
                else:
                    candidate[task].setInput(baseline[task].net.blob);gpu_raw=candidate[task].forward()
                    prediction=postprocess(baseline[task].manifest,gpu_raw,image)
                checks.append({'image_sha256':row['sha256'],'expires_at':row['expires_at'],**parity(task,raw,reference,gpu_raw,prediction)})
                if (index+1)%20==0:print(json.dumps({'phase':phase,'task':task,'completed':index+1,'total':len(entry['cases'])}),flush=True)
            details[task]=checks
            network=candidate[task].net if descriptor else candidate[task]
            actual[task]=network.runtime_options
    finally:
        for value in candidate.values():value.close()
        close_models(baseline)
        if old_profiling is None:os.environ.pop('ROAD_GPU_PROFILE_DIRECTORY',None)
        else:os.environ['ROAD_GPU_PROFILE_DIRECTORY']=old_profiling
    placement={};per_model_gpu={task:0 for task in TASKS}
    for task in TASKS:
        for file in (output/'operator-profiles'/task).glob('*.json'):
            for event in profile_events(file):
                provider=event.get('args',{}).get('provider')
                if provider:placement[provider]=placement.get(provider,0)+1
                if provider=='CUDAExecutionProvider':per_model_gpu[task]+=1
    validate_plan(read(plan_path))
    if sha256(plan_path)!=plan_sha or before!={name:sha256(ROOT/name) for name in before}:raise ValueError('trial_source_or_plan_changed')
    result={'format':'frozen-model-set-cuda-validation-v1','stage':phase,'platform':GPU_PROFILE,'plan_sha256':plan_sha,
        'source_sha256':before,'passed':all(check['passed'] for values in details.values() for check in values),
        'validated_models':{task:{'manifest_sha256':entry['manifest']['sha256'],'weights_sha256':entry['weights_sha256'],'case_set_sha256':entry['case_set_sha256']} for task,entry in plan['models'].items()},
        'results':{task:{'cases':len(values),'raw_failures':sum(not c['raw']['passed'] for c in values),'post_failures':sum(not c['post']['passed'] for c in values),
            'max_raw_abs_error':max(c['raw']['max_abs_error'] for c in values)} for task,values in details.items()},
        'actual_runtime':actual,'profiled_node_events_by_provider':placement,'gpu_node_events_observed':placement.get('CUDAExecutionProvider',0)>0,
        'per_model_gpu_node_events':per_model_gpu,
        'all_nodes_claimed_gpu':False,'test_or_user_images_read':0,'threshold_or_model_selection_performed':False,
        'source_expiry_min':min(datetime.fromisoformat(row['expires_at']) for entry in plan['models'].values() for row in entry['cases']).isoformat(),
        'source_expiries_extended':False,
        'raw_tolerance':{'rtol':.001,'atol':.0001},'independent_accuracy_evaluated':False}
    if descriptor:result['descriptor_core_sha256']=descriptor['core_sha256']
    write(output/'private-checks.json',details);write(output/'aggregate.json',result)
    return {key:result[key] for key in ('stage','passed','results','gpu_node_events_observed')}


def freeze(source,output):
    specification=read(source)
    if set(specification.get('models',{}))!=set(TASKS):raise ValueError('three_candidate_models_required')
    plan={'format':'frozen-cuda-model-set-v1','platform':GPU_PROFILE,'input_specification':pointer(source),
        'frozen_at':datetime.now(timezone.utc).isoformat(),'planning_source_sha256':{name:sha256(ROOT/name) for name in RUNTIME_SOURCES},
        'raw_tolerance':{'rtol':.001,'atol':.0001},
        'models':{},'test_or_user_images_read':0,'speed_selection_on_validation':False,
        'selection':'Canonical cases plus deterministic source-id order; validation only. No GPU measurement before this freeze.',
        'model_or_threshold_changes_permitted':False,'new_gpu_support_approved':False}
    for task,entry in specification['models'].items():
        manifest=pinned(entry['manifest']);canon=set()
        for proof_ref in entry['cpu_verifications']:
            proof=pinned(proof_ref);proof_passed(proof,manifest,entry['manifest']['sha256']);canon.add(proof['image_sha256'])
        rows=pinned(entry['validation_manifest']);selected=select_cases(task,rows,canon,entry.get('case_sha256s'))
        cases=[{key:row[key] for key in ('path','sha256','expires_at','sample_id')}|
            ({'evaluation_class':row['evaluation_class']} if task=='redaction' else {}) for row in selected]
        plan['models'][task]={key:entry[key] for key in ('manifest','cpu_verifications','validation_manifest')}
        if 'case_sha256s' in entry:plan['models'][task]['case_sha256s']=entry['case_sha256s']
        plan['models'][task].update(weights_sha256=manifest['sha256'],cases=cases,case_set_sha256=canonical(cases))
    validate_plan(plan)
    for entry in plan['models'].values():
        for row in entry['cases']:
            if sha256(row['path'])!=row['sha256']:raise ValueError('public_validation_image_changed')
    write(output,plan);return {'frozen_plan_sha256':sha256(output),'case_counts':COUNTS,'gpu_sessions_started':0}


def exact_child(manifest,base):
    return {k:v for k,v in manifest.items() if k not in ('required_runtime_profile','gpu_compatibility')}=={k:v for k,v in base.items() if k!='required_runtime_profile'}


def strict_results(proof,plan,stage):
    if proof.get('stage')!=stage or proof.get('passed') is not True or proof.get('platform')!=GPU_PROFILE:raise ValueError('strict_model_set_gpu_proof_required')
    if proof.get('raw_tolerance')!={'rtol':.001,'atol':.0001}:raise ValueError('default_gpu_tolerance_required')
    if proof.get('threshold_or_model_selection_performed') is not False or type(proof.get('test_or_user_images_read')) is not int or proof['test_or_user_images_read']!=0:raise ValueError('validation_only_no_selection_required')
    for task,entry in plan['models'].items():
        count=proof.get('per_model_gpu_node_events',{}).get(task)
        if type(count) is not int or count<=0:raise ValueError('actual_gpu_nodes_required_per_model')
        value=proof.get('results',{}).get(task,{})
        if (proof.get('validated_models',{}).get(task)!={'manifest_sha256':entry['manifest']['sha256'],'weights_sha256':entry['weights_sha256'],'case_set_sha256':entry['case_set_sha256']} or
            type(value.get('cases')) is not int or value['cases']!=COUNTS[task] or
            type(value.get('raw_failures')) is not int or type(value.get('post_failures')) is not int or
            value.get('raw_failures')!=0 or value.get('post_failures')!=0):raise ValueError('gpu_model_or_case_proof_mismatch')


def validate_staged_child(manifest):
    evidence=manifest.get('gpu_compatibility',{})
    if manifest.get('required_runtime_profile')!=GPU_PROFILE or evidence.get('format')!='frozen-model-set-diagnostic-v1':raise ValueError('staged_gpu_profile_required')
    plan=pinned(evidence['plan']);validate_plan(plan)
    proof=pinned(evidence['diagnostic']);strict_results(proof,plan,'diagnostic')
    if proof.get('plan_sha256')!=evidence['plan']['sha256']:raise ValueError('gpu_model_set_plan_binding_mismatch')
    if set(proof.get('source_sha256',{}))!=set(RUNTIME_SOURCES) or any(sha256(ROOT/name)!=digest for name,digest in proof['source_sha256'].items()):raise ValueError('gpu_model_set_proof_source_changed')
    task=manifest['task'];base=pinned(plan['models'][task]['manifest'])
    if not exact_child(manifest,base):raise ValueError('gpu_child_changed_model_contract')
    return evidence


def descriptor_core(value):return canonical({'plan':value['plan'],'models':value['models'],'platform':value['platform']})


def validate_descriptor(descriptor,registered=False):
    plan=pinned(descriptor['plan']);validate_plan(plan)
    if descriptor.get('platform')!=GPU_PROFILE or set(descriptor.get('models',{}))!=set(TASKS):raise ValueError('gpu_descriptor_platform_or_tasks')
    if descriptor.get('core_sha256')!=descriptor_core(descriptor):raise ValueError('gpu_descriptor_core_changed')
    for task,item in descriptor['models'].items():
        child=pinned({'path':item['manifest'],'sha256':item['manifest_sha256']});validate_staged_child(child)
        if (child.get('task')!=task or child['gpu_compatibility']['plan']['sha256']!=descriptor['plan']['sha256'] or
            item['base_manifest_sha256']!=plan['models'][task]['manifest']['sha256']):raise ValueError('gpu_descriptor_parent_changed')
    if registered:
        if descriptor.get('format')!='registered-cuda-model-set-v1':raise ValueError('registered_gpu_model_set_required')
        proof=pinned(descriptor['native']);strict_results(proof,plan,'native')
        if proof.get('plan_sha256')!=descriptor['plan']['sha256'] or proof.get('descriptor_core_sha256')!=descriptor['core_sha256'] or proof.get('gpu_node_events_observed') is not True:raise ValueError('native_gpu_descriptor_binding_mismatch')
        if set(proof.get('source_sha256',{}))!=set(RUNTIME_SOURCES) or any(sha256(ROOT/name)!=digest for name,digest in proof['source_sha256'].items()):raise ValueError('native_gpu_proof_source_changed')
    return plan


def stage(plan_path,diagnostic_path,output):
    plan=read(plan_path);validate_plan(plan);proof=read(diagnostic_path);strict_results(proof,plan,'diagnostic')
    if proof.get('plan_sha256')!=sha256(plan_path):raise ValueError('diagnostic_plan_changed')
    output=new_directory(output)
    descriptor={'format':'staging-cuda-model-set-v1','plan':pointer(plan_path),'platform':GPU_PROFILE,'models':{},'research_only':True,'production_approved':False}
    for task,entry in plan['models'].items():
        child=dict(pinned(entry['manifest']));child['required_runtime_profile']=GPU_PROFILE
        child['gpu_compatibility']={'format':'frozen-model-set-diagnostic-v1','plan':pointer(plan_path),'diagnostic':pointer(diagnostic_path)}
        validate_staged_child(child);path=output/(task+'.json');write(path,child)
        descriptor['models'][task]={'manifest':path.relative_to(ROOT).as_posix(),'manifest_sha256':sha256(path),'base_manifest_sha256':entry['manifest']['sha256']}
    descriptor['core_sha256']=descriptor_core(descriptor);write(output/'staging.json',descriptor)
    return {'staging_sha256':sha256(output/'staging.json'),'registered_for_workflow':False}


def register(staging_path,native_path,output):
    descriptor=read(staging_path);plan=validate_descriptor(descriptor)
    proof=read(native_path);strict_results(proof,plan,'native')
    if proof.get('plan_sha256')!=descriptor['plan']['sha256'] or proof.get('descriptor_core_sha256')!=descriptor['core_sha256'] or proof.get('gpu_node_events_observed') is not True:raise ValueError('native_descriptor_proof_required')
    output=workspace_path(output)
    if not output.is_relative_to(REGISTRY_ROOT.resolve()):raise ValueError('registered_gpu_profile_root_required')
    descriptor={**descriptor,'format':'registered-cuda-model-set-v1','native':pointer(native_path),'hardware_path_admitted':True}
    validate_descriptor(descriptor,True);write(output,descriptor)
    return {'registered_profile_sha256':sha256(output),'models':list(descriptor['models'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    item=commands.add_parser('freeze');item.add_argument('specification');item.add_argument('--output',required=True)
    item=commands.add_parser('diagnose');item.add_argument('plan');item.add_argument('--output',required=True)
    item=commands.add_parser('stage');item.add_argument('plan');item.add_argument('--diagnostic',required=True);item.add_argument('--output',required=True)
    item=commands.add_parser('native');item.add_argument('staging');item.add_argument('--output',required=True)
    item=commands.add_parser('register');item.add_argument('staging');item.add_argument('--native',required=True);item.add_argument('--output',required=True)
    item=commands.add_parser('benchmark');item.add_argument('profile');item.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='freeze':result=freeze(args.specification,args.output)
    elif args.command=='diagnose':result=run_validation(args.plan,args.output)
    elif args.command=='stage':result=stage(args.plan,args.diagnostic,args.output)
    elif args.command=='native':
        descriptor=read(args.staging);validate_descriptor(descriptor)
        result=run_validation(descriptor['plan']['path'],args.output,args.staging)
    elif args.command=='register':result=register(args.staging,args.native,args.output)
    else:
        from scripts.benchmark_verified_pipeline import run
        descriptor=read(args.profile);validate_descriptor(descriptor,True)
        run(args.output,args.profile,descriptor['native']['path'],descriptor['plan']['path']);result={'benchmark_completed':True}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
