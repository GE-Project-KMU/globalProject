"""CPU-only checkpoint state-byte and inference-contract identity evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def scoped(value):
    path=Path(value);path=(path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('outside_workspace')
    return path


def sha(path):return hashlib.sha256(scoped(path).read_bytes()).hexdigest()


def compare_records(left,right):
    same_keys=set(left)==set(right)
    common=set(left)&set(right)
    shapes=sum(left[k]['shape']!=right[k]['shape'] for k in common)
    dtypes=sum(left[k]['dtype']!=right[k]['dtype'] for k in common)
    raw=sum(left[k]['raw']!=right[k]['raw'] for k in common)
    return {'state_keys_exact':same_keys,'state_tensor_count':len(left),
        'shape_mismatches':shapes,'dtype_mismatches':dtypes,'raw_byte_mismatches':raw,
        'all_state_tensors_exact':same_keys and shapes==dtypes==raw==0}


def records(checkpoint):
    import torch
    value=torch.load(scoped(checkpoint),map_location='cpu',weights_only=True)
    state=value.get('model') if isinstance(value,dict) else None
    if not isinstance(state,dict) or not state:raise ValueError('model_state_dict_required')
    result={}
    for key,tensor in state.items():
        if not isinstance(key,str) or not isinstance(tensor,torch.Tensor) or tensor.layout!=torch.strided:raise ValueError('dense_tensor_state_required')
        tensor=tensor.detach().cpu().contiguous()
        result[key]={'shape':list(tensor.shape),'dtype':str(tensor.dtype),
            'raw':tensor.reshape(-1).view(torch.uint8).numpy().tobytes()}
    return result


def inference_contract(manifest):
    return {key:manifest.get(key,default) for key,default in {
        'format':None,'task':None,'classes':None,'architecture':'YOLOX-Nano',
        'input_size':None,'output_layout':None,'confidence':.35,'nms_iou':.5,
        'strides':[8,16,32]}.items()}


def verify(old_checkpoint,new_checkpoint,old_manifest,new_manifest):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2)
    result=compare_records(records(old_checkpoint),records(new_checkpoint))
    old=json.loads(scoped(old_manifest).read_text(encoding='utf-8'));new=json.loads(scoped(new_manifest).read_text(encoding='utf-8'))
    result.update(format='checkpoint-inference-identity-v1',device='cpu',torch_version=torch.__version__,
        original_checkpoint_sha256=sha(old_checkpoint),selected_checkpoint_sha256=sha(new_checkpoint),
        checkpoint_serialization_identical=sha(old_checkpoint)==sha(new_checkpoint),
        original_manifest_sha256=sha(old_manifest),selected_manifest_sha256=sha(new_manifest),
        inference_contract=inference_contract(new),inference_contract_exact=inference_contract(old)==inference_contract(new),
        original_onnx_sha256=sha(old['weights']),selected_onnx_sha256=sha(new['weights']),
        onnx_bytes_exact=scoped(old['weights']).read_bytes()==scoped(new['weights']).read_bytes(),
        source_sha256=sha(Path(__file__)),test_inference_performed=False,gpu_operations=0)
    result['prediction_function_identical']=result['all_state_tensors_exact'] and result['inference_contract_exact'] and result['onnx_bytes_exact']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('old-checkpoint','new-checkpoint','old-manifest','new-manifest','output'):p.add_argument('--'+arg,required=True)
    a=p.parse_args();output=scoped(a.output)
    if output.exists():raise ValueError('output_exists')
    result=verify(a.old_checkpoint,a.new_checkpoint,a.old_manifest,a.new_manifest)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['prediction_function_identical']:sys.exit(2)
