"""Create local research CUDA children only for checksum-bound strict evidence."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.gpu_worker_client import GPU_PROFILE,validate_gpu_manifest
from agent.paths import sha256


def create():
    proof=ROOT/'data/gpu-inference-probe/v2'
    gpu_plan=proof/'private-plan.json';aggregate=proof/'aggregate.json'
    cpu_plan=ROOT/'data/cpu-inference-benchmark/public-val-v1/private-plan.json'
    cpu=json.loads(cpu_plan.read_text())
    from scripts.benchmark_inference_cpu import MANIFESTS
    folder=ROOT/'data/training/gpu-runtime-ort1202-v2'
    descriptor=ROOT/'data/gpu-runtime-profiles/ort1202-v2.json'
    if folder.exists() or descriptor.exists():raise ValueError('gpu_profile_output_exists')
    native=ROOT/'data/gpu-inference-probe/native-v1'
    executed=ROOT/'data/gpu-runtime-profiles/ort1202-v1.json'
    executed_models=json.loads(executed.read_text())['models']
    def pointer(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path)}
    binding={'format':'native-gpu-proof-model-binding-v1','aggregate_sha256':sha256(aggregate),
        'gpu_plan_sha256':sha256(gpu_plan),'cpu_plan_sha256':sha256(cpu_plan),
        'native_aggregate':pointer(native/'aggregate.json'),'native_receipt':pointer(native/'receipt.json'),
        'executed_profile':pointer(executed),'validated_models':{task:{
            'base_manifest_sha256':sha256(ROOT/base_path),'weights_sha256':json.loads((ROOT/base_path).read_text())['sha256'],
            'executed_child_sha256':executed_models[task]['manifest_sha256']} for task,base_path in MANIFESTS.items()}}
    binding_path=ROOT/'data/gpu-runtime-evidence/native-old3-binding-v1.json'
    if binding_path.exists():raise ValueError('gpu_binding_receipt_exists')
    binding_path.parent.mkdir(parents=True,exist_ok=True);binding_path.write_text(json.dumps(binding,indent=2)+'\n')
    children={}
    for task,base_path in MANIFESTS.items():
        base=ROOT/base_path
        if sha256(base)!=cpu['model_manifest_sha256'][task]:raise ValueError('new_model_requires_gpu_revalidation')
        manifest=json.loads(base.read_text());child=dict(manifest)
        child['required_runtime_profile']=GPU_PROFILE
        child['gpu_compatibility']={key:{'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path)} for key,path in
            [('base_manifest',base),('aggregate',aggregate),('gpu_plan',gpu_plan),('cpu_plan',cpu_plan)]}
        child['gpu_compatibility']['binding']=pointer(binding_path)
        validate_gpu_manifest(child);children[task]=child
    folder.mkdir(parents=True);descriptor.parent.mkdir(parents=True,exist_ok=True)
    output={'format':'verified-local-cuda-bundle-v1','models':{},'research_only':True,'production_approved':False}
    for task,child in children.items():
        path=folder/(task+'.json');path.write_text(json.dumps(child,indent=2)+'\n')
        output['models'][task]={'manifest':path.relative_to(ROOT).as_posix(),'manifest_sha256':sha256(path),
            'base_manifest_sha256':child['gpu_compatibility']['base_manifest']['sha256']}
    descriptor.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'local_gpu_profile_created':True,'models':list(children),'gpu_sessions_started':0,'weights_copied':0}))


if __name__=='__main__':create()
