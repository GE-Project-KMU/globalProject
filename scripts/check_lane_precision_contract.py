"""Independent synthetic operator/state contracts, never accuracy evidence."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch,numpy as np
from agent.lane_network import make_lane_network
from agent.paths import atomic_json,sha256
from diagnose_lane_export_precision import MODES,build_variant,compare

def main():
    torch.set_num_threads(2);torch.manual_seed(20261104)
    model=make_lane_network(pretrained=False,aspp_dilations=(6,12,18)).eval()
    original={k:v.clone() for k,v in model.state_dict().items()};x=torch.randn(1,3,64,96)
    with torch.inference_mode():expected=model(x)["out"]
    checks={}
    for mode in MODES:
        variant,_=build_variant(model,mode)
        checks[mode+"_checkpoint_state_exact"]=set(original)==set(variant.network.state_dict()) and all(torch.equal(v,variant.network.state_dict()[k]) for k,v in original.items())
        with torch.inference_mode():actual=variant(x)
        checks[mode+"_geometry_and_finite"]=actual.shape==expected.shape and bool(torch.isfinite(actual).all())
        checks[mode+"_synthetic_forward_equivalence"]=bool(torch.allclose(actual,expected,rtol=.001,atol=.0001))
    checks["original_factory_not_mutated"]=sum(isinstance(m,torch.nn.AdaptiveAvgPool2d) for m in model.modules())==9 and all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    high,_=build_variant(model,"pool_fp64")
    pool=high.network.get_submodule("backbone.4.block.2.avgpool")
    signed=(np.arange(2*3*17*19,dtype=np.float32).reshape(2,3,17,19)%13-6)
    expected_mean=signed.astype(np.float64).sum(axis=(2,3),keepdims=True)/(17*19)
    with torch.inference_mode():actual_mean=pool(torch.from_numpy(signed)).numpy()
    checks["signed_rectangular_mean_matches_independent_numpy_double_sum"]=bool(np.array_equal(actual_mean,expected_mean.astype(np.float32)))
    with torch.inference_mode():
        checks["constant_mean_has_no_shift_or_clamp"]=all(bool(torch.equal(pool(torch.full((1,2,7,11),v)),torch.full((1,2,1,1),v))) for v in (0.,7.,-3.5))
    reference=np.zeros((1,2,2,2),np.float32);reference[:,1]=3.;perturbed=reference.copy();perturbed[0,0,0,0]=.0002
    result=compare(reference,perturbed,.4)
    checks["identical_masks_do_not_override_raw_failure"]=result["passed"] is False and result["mismatch_elements"]==1 and result["mask_pixel_agreement"]==1.
    checks["CPU_only_no_cuda_initialization"]=not torch.cuda.is_initialized()
    report={"accuracy_evidence":False,"checks":checks,"passed":all(checks.values()),"torch":torch.__version__,
        "factory_sha256":sha256(ROOT/"agent/lane_network.py"),"diagnostic_script_sha256":sha256(ROOT/"scripts/diagnose_lane_export_precision.py"),
        "scope":"synthetic numerical/shape/learned-state safety only; not model quality or speed"}
    atomic_json(ROOT/"docs/reports/LANE-PRECISION-CONTRACT.json",report);print(json.dumps(report),flush=True)
    if not report["passed"]:raise ValueError("precision_operator_contract_failed")

if __name__=="__main__":main()
