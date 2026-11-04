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
    folder=ROOT/'data/training/gpu-runtime-ort1202-v1'
    descriptor=ROOT/'data/gpu-runtime-profiles/ort1202-v1.json'
    if folder.exists() or descriptor.exists():raise ValueError('gpu_profile_output_exists')
    children={}
    for task,base_path in MANIFESTS.items():
        base=ROOT/base_path
        if sha256(base)!=cpu['model_manifest_sha256'][task]:raise ValueError('new_model_requires_gpu_revalidation')
        manifest=json.loads(base.read_text());child=dict(manifest)
        child['required_runtime_profile']=GPU_PROFILE
        child['gpu_compatibility']={key:{'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path)} for key,path in
            [('base_manifest',base),('aggregate',aggregate),('gpu_plan',gpu_plan),('cpu_plan',cpu_plan)]}
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
