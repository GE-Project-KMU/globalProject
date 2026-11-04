"""Independent state/linear-algebra contracts for the single additional recipe."""
import os,sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch,numpy as np
from agent.lane_network import make_lane_network
from agent.paths import atomic_json,sha256
from diagnose_lane_head_precision import pointwise64,build,state_digest

def main():
    torch.set_num_threads(2);torch.manual_seed(20261104);checks={}
    layer=torch.nn.Conv2d(4,2,1)
    with torch.no_grad():layer.weight.copy_(torch.arange(8,dtype=torch.float32).reshape(2,4,1,1)-3);layer.bias.copy_(torch.tensor([.5,-.25]))
    before=state_digest(layer);linear=pointwise64(layer)
    a=(np.arange(2*4*7*11,dtype=np.float32).reshape(2,4,7,11)%17)-8
    expected=np.einsum('oc,nchw->nohw',layer.weight.detach().numpy()[:,:,0,0].astype(np.float64),a.astype(np.float64))+layer.bias.detach().numpy().astype(np.float64)[None,:,None,None]
    with torch.inference_mode():result=linear(torch.from_numpy(a)).numpy()
    checks["NHWC_MatMul_matches_independent_numpy_einsum"]=bool(np.array_equal(result,expected.astype(np.float32)))
    checks["input_output_dtype_stays_float32"]=result.dtype==np.float32
    checks["learned_parameter_bytes_preserved"]=before==state_digest(layer)==state_digest(linear)
    try:pointwise64(torch.nn.Conv2d(4,2,3,padding=1))
    except ValueError:checks["nonpointwise_kernel_rejected"]=True
    else:checks["nonpointwise_kernel_rejected"]=False
    try:pointwise64(torch.nn.Conv2d(4,2,1,stride=2))
    except ValueError:checks["strided_kernel_rejected"]=True
    else:checks["strided_kernel_rejected"]=False
    model=make_lane_network(pretrained=False,aspp_dilations=(6,12,18)).eval();before=state_digest(model);candidate=build(model)
    checks["full_checkpoint_state_bytes_identical"]=before==state_digest(candidate.network)==state_digest(model)
    x=torch.randn(1,3,64,96)
    with torch.inference_mode():reference=model(x)["out"];actual=candidate(x)
    checks["synthetic_function_equivalence_default_tolerance"]=bool(torch.allclose(reference,actual,rtol=.001,atol=.0001))
    checks["source_factory_not_mutated"]=isinstance(model.classifier[-1],torch.nn.Conv2d) and sum(isinstance(m,torch.nn.AdaptiveAvgPool2d) for m in model.modules())==9
    checks["CPU_only_no_cuda_initialized"]=not torch.cuda.is_initialized()
    report={"checks":checks,"passed":all(checks.values()),"accuracy_evidence":False,"recipe_sha256":sha256(ROOT/"scripts/diagnose_lane_head_precision.py"),
        "scope":"synthetic independent NumPy linear map, shape, dtype and state identity only"}
    atomic_json(ROOT/"docs/reports/LANE-HEAD-PRECISION-CONTRACT.json",report);print(json.dumps(report),flush=True)
    if not report["passed"]:raise ValueError("head_precision_contract_failed")

if __name__=="__main__":main()
