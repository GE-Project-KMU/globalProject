"""CPU-only checkpoint-derived global1x1 ReduceMean export, diagnostic only.

Preserves the original model and its failed raw-parity report. This command
never trains, calibrates, reads TEST inputs, or promotes a deployment model.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,re,sys,uuid
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
REASON='checkpoint_raw_tensor_parity_unresolved'

def path(value,exists=False):
    p=Path(value);p=(ROOT/p if not p.is_absolute() else p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('path_outside_workspace')
    if exists and not p.is_file():raise ValueError('required_file_missing')
    return p
def sha(value):
    h=hashlib.sha256()
    with path(value,True).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def pin(value):
    if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value):raise ValueError('pinned_sha256_required')
    return value
def read(value):return json.loads(path(value,True).read_text(encoding='utf-8'))
def core_hash(value):return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!='models'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(value,body):
    target=path(value);target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('x',encoding='utf-8') as stream:json.dump(body,stream,allow_nan=False,sort_keys=True,indent=2)
        os.replace(tmp,target)
    finally:tmp.unlink(missing_ok=True)
@contextmanager
def lock(value):
    target=path(value);token=uuid.uuid4().hex
    try:fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise ValueError('portable_export_locked')
    def verify():
        if not target.is_file() or target.read_text()!=token:raise ValueError('portable_export_lock_changed')
    try:
        with os.fdopen(fd,'w') as stream:stream.write(token)
        yield verify;verify()
    finally:
        if target.is_file() and target.read_text()==token:target.unlink()

def export_checkpoint(checkpoint,manifest,weights):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import copy,torch,onnx
    from agent.lane_network import make_lane_network
    torch.set_num_threads(2)
    original=make_lane_network(pretrained=False,aspp_dilations=manifest.get('aspp_dilations',[12,24,36])).eval().cpu()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);state=state.get('model',state) if isinstance(state,dict) else state
    original.load_state_dict(state,strict=True);model=copy.deepcopy(original);replaced=[]
    class GlobalMean(torch.nn.Module):
        def forward(self,x):return x.mean((2,3),keepdim=True)
    def replace(module,prefix=''):
        for name,child in list(module.named_children()):
            if isinstance(child,torch.nn.AdaptiveAvgPool2d):
                if child.output_size not in (1,(1,1)):raise ValueError('only_global1_pooling_supported')
                module._modules[name]=GlobalMean();replaced.append(prefix+name)
            else:replace(child,prefix+name+'.')
    replace(model)
    if len(replaced)!=9:raise ValueError('expected_nine_global1_pools')
    if set(model.state_dict())!=set(original.state_dict()) or any(not torch.equal(value,model.state_dict()[key]) for key,value in original.state_dict().items()):raise ValueError('checkpoint_learned_state_changed')
    for parameter in model.parameters():parameter.requires_grad_(False)
    width,height=manifest['input_size']
    class Output(torch.nn.Module):
        def __init__(self,network):super().__init__();self.network=network
        def forward(self,x):return self.network(x)['out']
    with torch.inference_mode():torch.onnx.export(Output(model),torch.zeros(1,3,height,width),str(weights),input_names=['images'],output_names=['logits'],opset_version=17,dynamo=False)
    graph=onnx.load(str(weights));ops=[node.op_type for node in graph.graph.node]
    if ops.count('ReduceMean')!=9 or 'GlobalAveragePool' in ops:raise ValueError('explicit_mean_export_contract_failed')
    return {'global1_pools_replaced':9,'reduce_mean_operators':9,'global_average_pool_operators':0,'strict_checkpoint_load':True,'learned_state_tensors_unchanged':True,'torch':torch.__version__,'device':'cpu'}

def link(provenance,manifest):
    from agent.governance import link_model
    link_model(provenance,manifest)

def export(original_manifest,original_manifest_sha,checkpoint,checkpoint_sha,output,original_verification=None):
    original_manifest=path(original_manifest,True);checkpoint=path(checkpoint,True);output=path(output)
    if output.parent!=original_manifest.parent:raise ValueError('new_child_portable_directory_required')
    verification=path(original_verification or original_manifest.parent/'verification-final.json',True)
    with lock(output.with_name('.'+output.name+'.export.lock')) as verify_lock:
        if output.exists():raise ValueError('portable_output_already_exists')
        if sha(original_manifest)!=pin(original_manifest_sha) or sha(checkpoint)!=pin(checkpoint_sha):raise ValueError('original_source_or_checkpoint_changed')
        m=read(original_manifest);failure=read(verification)
        if m.get('format')!='onnx' or m.get('task')!='lane' or m.get('classes')!=['background','paint'] or m.get('research_only') is not True or m.get('production_approved') is not False:raise ValueError('research_lane_source_required')
        from scripts.verify_trained_models import validate_architecture
        validate_architecture(m)
        if failure.get('manifest_sha256')!=original_manifest_sha or failure.get('weight_sha256')!=m['sha256'] or failure.get('numerical_parity_passed') is not False or failure.get('torch_checkpoint_compared') is not True:raise ValueError('original_raw_parity_failure_required')
        original_weights=path(m['weights'],True)
        if sha(original_weights)!=m['sha256']:raise ValueError('original_onnx_changed')
        provenance=path(m['training_provenance'],True);p=read(provenance);core=core_hash(p)
        if p.get('task')!='lane' or p.get('data_revoked') or p.get('needs_revalidation') or not p.get('assets'):raise ValueError('training_provenance_blocked')
        if any(datetime.fromisoformat(asset['expires_at'].replace('Z','+00:00'))<=datetime.now(timezone.utc) or asset.get('revoked_at') for asset in p['assets']):raise ValueError('training_source_expired_or_revoked')
        original_failure_sha=sha(verification);output.mkdir();weights=output/'lane.onnx'
        result=export_checkpoint(checkpoint,m,weights);verify_lock()
        if sha(original_manifest)!=original_manifest_sha or sha(checkpoint)!=checkpoint_sha or sha(original_weights)!=m['sha256'] or sha(verification)!=original_failure_sha:raise ValueError('frozen_source_changed_during_export')
        portable={k:v for k,v in m.items() if k!='threshold_calibration'}
        portable.update(weights=weights.relative_to(ROOT).as_posix(),sha256=sha(weights),original_model_manifest_sha256=original_manifest_sha,original_onnx_sha256=m['sha256'],original_checkpoint_sha256=checkpoint_sha,trained_checkpoint_sha256=checkpoint_sha,original_raw_verification_sha256=original_failure_sha,original_threshold_calibration=m.get('threshold_calibration'),export_transform={'kind':'checkpoint_derived_global1x1_explicit_mean','global_pools_replaced':9,'onnx_reduce_mean_count':9,'learned_state_unchanged':True},export_script_sha256=sha(Path(__file__)),no_new_gradient_steps=True,numerical_parity_passed=False,checkpoint_raw_tensor_parity_passed=False,checkpoint_raw_tensor_parity_unresolved=True,diagnostic_accuracy_only=True,production_approved=False,project_acceptance_eligible=False)
        portable['deployment_blocked_reasons']=sorted(set(portable.get('deployment_blocked_reasons',[])+[REASON]))
        target=output/'lane.json';write(target,portable);link(provenance,target)
        if core_hash(read(provenance))!=core:raise ValueError('provenance_assets_or_ttl_changed')
        verify_lock()
        receipt={'format':'lane-portable-export-v1','exported_at':datetime.now(timezone.utc).isoformat(),'original_model_manifest_sha256':original_manifest_sha,'original_onnx_sha256':m['sha256'],'original_checkpoint_sha256':checkpoint_sha,'original_raw_verification_sha256':original_failure_sha,'portable_model_manifest_sha256':sha(target),'portable_onnx_sha256':sha(weights),'export_contract':result,'export_script_sha256':sha(Path(__file__)),'numerical_parity_passed':False,'deployment_blocked_reasons':[REASON],'source_expiries_extended':False,'original_failure_and_artifacts_unchanged':True,'no_new_gradient_steps':True,'test_accessed':False,'gpu_operations':0}
        write(output/'export-receipt.json',receipt);return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--original-manifest',required=True);parser.add_argument('--original-manifest-sha256',required=True);parser.add_argument('--checkpoint',required=True);parser.add_argument('--checkpoint-sha256',required=True);parser.add_argument('--output',required=True);parser.add_argument('--original-verification');args=parser.parse_args()
    try:
        print(json.dumps(export(args.original_manifest,args.original_manifest_sha256,args.checkpoint,args.checkpoint_sha256,args.output,args.original_verification),indent=2));return 0
    except Exception as error:
        reason=str(error);reason=reason if re.fullmatch('[a-z0-9_]{1,100}',reason) else 'portable_export_failed';print(json.dumps({'status':'portable_export_refused','error_type':type(error).__name__,'reason':reason}));return 2
if __name__=='__main__':raise SystemExit(main())
