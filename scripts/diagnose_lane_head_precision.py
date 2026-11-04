"""One additional, predeclared numerical candidate: FP64 global mean + head dot.

The last learned1x1 linear map uses FP64 MatMul accumulation then restores FP32.
No parameter value, threshold, logit offset, clipping or rounding is changed.
"""
import argparse,json,os,sys,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from agent.paths import workspace_path,atomic_json,sha256

def state_digest(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def pointwise64(conv):
    import torch
    if not isinstance(conv,torch.nn.Conv2d) or conv.kernel_size!=(1,1) or conv.stride!=(1,1) or conv.padding!=(0,0) or conv.dilation!=(1,1) or conv.groups!=1:
        raise ValueError("only_unstrided_unpadded_pointwise_linear_map_supported")
    class Linear64(torch.nn.Module):
        def __init__(self,source):
            super().__init__();self.weight=source.weight;self.bias=source.bias
        def forward(self,x):
            matrix=self.weight[:,:,0,0].transpose(0,1).double()
            output=torch.matmul(x.permute(0,2,3,1).double(),matrix)
            if self.bias is not None:output=output+self.bias.double()
            return output.permute(0,3,1,2).to(x.dtype)
    return Linear64(conv)

def build(original):
    from diagnose_lane_export_precision import build_variant
    model,_=build_variant(original,"pool_fp64")
    model.network.classifier[-1]=pointwise64(model.network.classifier[-1])
    if state_digest(original)!=state_digest(model.network):raise ValueError("original_learned_state_bytes_changed")
    return model.eval()

def prepare(output):
    output=workspace_path(output)
    if output.exists():raise ValueError("fresh_candidate_study_directory_required")
    prior=ROOT/"data/training/lane-a2d2-fp64-mean-v3/full-validation-results.json"
    results=json.loads(prior.read_text())
    if results.get("complete") is not True or len(results["cases"])!=100:raise ValueError("previous_100_validation_scope_must_finish")
    output.mkdir(parents=True)
    plan={"additional_candidate_count":1,"expression":"9 FP64 global mean accumulations + final1x1 FP64 MatMul then FP32 output",
        "selection_scope":"numerical validation only; no model weights/threshold/TEST selection",
        "previous_full_validation_sha256":sha256(prior),"previous_raw_passed":results["all_raw_passed"],"previous_post_passed":results["all_post_passed"],
        "three_validation_plan_sha256":sha256(ROOT/"data/training/lane-numeric-diagnostics-v3/validation-plan.json"),
        "checkpoint_sha256":results["checkpoint_sha256"],"rtol":.001,"atol":.0001,"paint_threshold":.4,
        "learned_state_must_remain_byte_identical":True,"rounding_clipping_logit_offset_forbidden":True,"GPU_operations":0,"test_accessed":False}
    atomic_json(output/"head-plan.json",plan)
    source=json.loads((ROOT/"data/training/lane-numeric-diagnostics-v3/source.json").read_text());atomic_json(output/"source.json",source)
    print(json.dumps({"status":"single_additional_candidate_predeclared","plan_sha256":sha256(output/"head-plan.json")}),flush=True)

def run(output):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import numpy as np,torch,onnx,onnxruntime as ort
    from agent.lane_network import make_lane_network
    from diagnose_lane_export_precision import compare
    output=workspace_path(output,exists=True);plan=json.loads((output/"head-plan.json").read_text())
    checkpoint=ROOT/"data/training/lane-a2d2-v2/lane_best.pt"
    if sha256(checkpoint)!=plan["checkpoint_sha256"]:raise ValueError("checkpoint_changed")
    source_plan=ROOT/"data/training/lane-numeric-diagnostics-v3/validation-plan.json"
    if sha256(source_plan)!=plan["three_validation_plan_sha256"]:raise ValueError("validation_plan_changed")
    cases=json.loads(source_plan.read_text())["cases"];torch.set_num_threads(2)
    original=make_lane_network(pretrained=False,aspp_dilations=(6,12,18)).eval().cpu()
    state=torch.load(checkpoint,map_location="cpu",weights_only=True);original.load_state_dict(state.get("model",state),strict=True)
    transformed=build(original);graph=output/"lane.onnx"
    blobs=[np.load(workspace_path(c["input_blob"],exists=True),allow_pickle=False) for c in cases]
    with torch.inference_mode():torch.onnx.export(transformed,torch.from_numpy(blobs[0]),str(graph),input_names=["images"],output_names=["logits"],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(str(graph)),full_check=True)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    results={"checkpoint_sha256":plan["checkpoint_sha256"],"onnx_sha256":sha256(graph),"plan_sha256":sha256(output/"head-plan.json"),
        "original_learned_state_digest":state_digest(original),"candidate_learned_state_digest":state_digest(transformed.network),
        "learned_state_byte_identical":state_digest(original)==state_digest(transformed.network),
        "recipe_sha256":sha256(Path(__file__)),"rtol":.001,"atol":.0001,"test_accessed":False,"GPU_operations":0,"cases":[]}
    try:
        session=ort.InferenceSession(str(graph),sess_options=options,providers=["CPUExecutionProvider"])
        for i,(case,blob) in enumerate(zip(cases,blobs)):
            with torch.inference_mode():native=original(torch.from_numpy(blob))["out"].numpy();modified=transformed(torch.from_numpy(blob)).numpy()
            actual=session.run(None,{session.get_inputs()[0].name:blob})[0]
            row={"index":i,"image_sha256":case["image_sha256"],"modified_torch_vs_original":compare(native,modified,.4),
                "ORT_vs_original":compare(native,actual,.4),"ORT_vs_modified_torch":compare(modified,actual,.4)}
            results["cases"].append(row)
        results["all_three_raw_passed"]=all(c["ORT_vs_original"]["passed"] for c in results["cases"])
        results["all_three_mask_identical"]=all(c["ORT_vs_original"]["mask_pixel_agreement"]==1. for c in results["cases"])
    except Exception as error:
        results.update(backend_error_type=type(error).__name__,all_three_raw_passed=False,all_three_mask_identical=False)
    if sha256(checkpoint)!=plan["checkpoint_sha256"]:raise ValueError("checkpoint_changed_during_probe")
    atomic_json(output/"head-results.json",results)
    print(json.dumps({"all_three_raw_passed":results["all_three_raw_passed"],"all_three_mask_identical":results["all_three_mask_identical"],
        "mismatch_elements":[c["ORT_vs_original"]["mismatch_elements"] for c in results["cases"]],"onnx_sha256":results["onnx_sha256"]}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","run"]);p.add_argument("output");a=p.parse_args()
    prepare(a.output) if a.action=="prepare" else run(a.output)
