"""CPU-only, checkpoint-derived export diagnostics on predeclared positive VAL.

No gradient updates, logit offsets, rounding, clipping or tolerance changes.
Original checkpoint/exports/proofs remain read-only. TEST is never loaded.
"""
import argparse,copy,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
MODES=("mean_control","resize_width_height","resize_height_width","resize_fp64","pool_fp64","pool_and_resize_fp64")

def prepare(output):
    import cv2,numpy as np
    output=workspace_path(output)
    if output.exists():raise ValueError("fresh_diagnostic_directory_required")
    manifest_path=ROOT/"data/training/lane-a2d2-v2/lane.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    development=workspace_path(manifest["training_development_release_path"],exists=True)
    if sha256(development)!=manifest["training_development_release_sha256"]:raise ValueError("development_release_changed")
    rows=json.loads(development.read_text(encoding="utf-8"))
    selected=sorted((r for r in rows if r["split"]=="validation" and r["positive_pixels"]>0),key=lambda r:r["source_id"])[:3]
    if len(selected)!=3:raise ValueError("three_positive_validation_inputs_required")
    output.mkdir(parents=True);cases=[];width,height=manifest["input_size"]
    for i,row in enumerate(selected):
        image_path=workspace_path(row["path"],exists=True);mask_path=workspace_path(row["mask_path"],exists=True)
        if sha256(image_path)!=row["sha256"] or sha256(mask_path)!=row["mask_sha256"]:raise ValueError("validation_source_changed")
        image=cv2.imread(str(image_path));mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or not (mask==1).any():raise ValueError("positive_validation_GT_required")
        h,w=image.shape[:2];scale=min(width/w,height/h);rw,rh=round(w*scale),round(h*scale)
        canvas=np.full((height,width,3),114,np.uint8);canvas[:rh,:rw]=cv2.resize(image,(rw,rh))
        normalized=(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).astype(np.float32)/255-np.asarray(manifest["mean"],np.float32))/np.asarray(manifest["std"],np.float32)
        blob=normalized.transpose(2,0,1)[None].copy();path=output/f"validation-{i}.npy";np.save(path,blob,allow_pickle=False)
        cases.append({"index":i,"source_id":row["source_id"],"image_sha256":row["sha256"],"image_path":row["path"],
            "mask_path":row["mask_path"],"input_blob":path.relative_to(ROOT).as_posix(),"blob_sha256":sha256(path),
            "source_expires_at":row["expires_at"],"positive_GT_pixels":int((mask==1).sum()),"resized_size":[rh,rw],"original_size":[h,w]})
    plan={"scope":"predeclared_positive_validation_only_CPU_export_diagnostics","modes":list(MODES),
        "selection":"first three positive-GT validation source_ids; no prediction/test selection",
        "rtol":.001,"atol":.0001,"paint_threshold":manifest["paint_threshold"],"gpu_operations":0,"test_accessed":False,
        "original_manifest_sha256":sha256(manifest_path),"checkpoint_sha256":sha256(ROOT/"data/training/lane-a2d2-v2/lane_best.pt"),
        "original_onnx_sha256":manifest["sha256"],"development_release_sha256":sha256(development),"cases":cases}
    atomic_json(output/"validation-plan.json",plan)
    atomic_json(output/"source.json",{"expires_at":min(c["source_expires_at"] for c in cases),"source_deadline_extended":False,
        "purpose":"private reconstructible normalized VAL image blobs and per-image numerical diagnostics"})
    print(json.dumps({"status":"plan_frozen","cases":3,"modes":list(MODES),"plan_sha256":sha256(output/"validation-plan.json")}),flush=True)

def build_variant(original,mode):
    import torch
    model=copy.deepcopy(original)
    class Mean(torch.nn.Module):
        def forward(self,x):return x.mean((2,3),keepdim=True)
    class Mean64(torch.nn.Module):
        def forward(self,x):return x.double().mean((2,3),keepdim=True).to(x.dtype)
    replaced=[]
    def replace(parent,prefix=""):
        for name,child in list(parent.named_children()):
            if isinstance(child,torch.nn.AdaptiveAvgPool2d):
                if child.output_size not in (1,(1,1)):raise ValueError("global1x1_pool_only")
                parent._modules[name]=Mean64() if mode in ("pool_fp64","pool_and_resize_fp64") else Mean();replaced.append(prefix+name)
            else:replace(child,prefix+name+".")
    replace(model)
    if len(replaced)!=9:raise ValueError("nine_global_pool_sites_expected")
    before=original.state_dict();after=model.state_dict()
    if set(before)!=set(after) or any(not torch.equal(value,after[key]) for key,value in before.items()):raise ValueError("learned_state_changed")
    class Output(torch.nn.Module):
        def __init__(self,network):super().__init__();self.network=network
        def forward(self,x):
            out=self.network.classifier(self.network.backbone(x)["out"])
            if mode=="resize_width_height":
                out=torch.nn.functional.interpolate(out,size=(out.shape[-2],x.shape[-1]),mode="bilinear",align_corners=False)
                return torch.nn.functional.interpolate(out,size=x.shape[-2:],mode="bilinear",align_corners=False)
            if mode=="resize_height_width":
                out=torch.nn.functional.interpolate(out,size=(x.shape[-2],out.shape[-1]),mode="bilinear",align_corners=False)
                return torch.nn.functional.interpolate(out,size=x.shape[-2:],mode="bilinear",align_corners=False)
            if mode in ("resize_fp64","pool_and_resize_fp64"):
                return torch.nn.functional.interpolate(out.double(),size=x.shape[-2:],mode="bilinear",align_corners=False).to(out.dtype)
            return torch.nn.functional.interpolate(out,size=x.shape[-2:],mode="bilinear",align_corners=False)
    return Output(model).eval(),replaced

def compare(reference,actual,threshold):
    import numpy as np
    from scripts.verify_trained_models import compare as raw_compare
    result=raw_compare(actual,reference,.001,.0001)
    close=np.isclose(actual,reference,rtol=.001,atol=.0001)
    delta=np.abs(actual-reference);allowed=.0001+.001*np.abs(reference)
    result.update(mismatch_elements=int((~close).sum()),total_elements=int(reference.size),max_exceed_ratio=float((delta/allowed).max()))
    a=(actual[0,1]-actual[0,0])>math.log(threshold/(1-threshold));b=(reference[0,1]-reference[0,0])>math.log(threshold/(1-threshold))
    result["mask_pixel_agreement"]=float((a==b).mean());result["reference_paint_pixels"]=int(b.sum());result["actual_paint_pixels"]=int(a.sum())
    return result

def run(output):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import torch,numpy as np,onnx,onnxruntime as ort
    from agent.lane_network import make_lane_network
    torch.set_num_threads(2)
    output=workspace_path(output,exists=True);plan_path=output/"validation-plan.json";plan=json.loads(plan_path.read_text())
    checkpoint=ROOT/"data/training/lane-a2d2-v2/lane_best.pt";manifest_path=ROOT/"data/training/lane-a2d2-v2/lane.json"
    if sha256(checkpoint)!=plan["checkpoint_sha256"] or sha256(manifest_path)!=plan["original_manifest_sha256"]:raise ValueError("frozen_source_changed")
    info=json.loads(manifest_path.read_text());original=make_lane_network(pretrained=False,aspp_dilations=info["aspp_dilations"]).eval().cpu()
    state=torch.load(checkpoint,map_location="cpu",weights_only=True);state=state.get("model",state)
    original.load_state_dict(state,strict=True)
    blobs=[];native=[]
    for case in plan["cases"]:
        path=workspace_path(case["input_blob"],exists=True)
        if sha256(path)!=case["blob_sha256"]:raise ValueError("validation_blob_changed")
        blob=np.load(path,allow_pickle=False);blobs.append(blob)
        with torch.inference_mode():native.append(original(torch.from_numpy(blob))["out"].numpy())
    results={"plan_sha256":sha256(plan_path),"checkpoint_sha256":sha256(checkpoint),"rtol":.001,"atol":.0001,
        "torch":torch.__version__,"ort":ort.__version__,"gpu_operations":0,"test_accessed":False,"variants":{}}
    for mode in plan["modes"]:
        start=time.perf_counter();variant,_=build_variant(original,mode);rows=[];onnx_path=output/(mode+".onnx")
        with torch.inference_mode():
            transformed=[variant(torch.from_numpy(blob)).numpy() for blob in blobs]
            torch.onnx.export(variant,torch.from_numpy(blobs[0]),str(onnx_path),input_names=["images"],output_names=["logits"],opset_version=17,dynamo=False)
        onnx.checker.check_model(onnx.load(str(onnx_path)),full_check=True)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        try:
            session=ort.InferenceSession(str(onnx_path),sess_options=options,providers=["CPUExecutionProvider"])
            for i,(blob,reference,changed) in enumerate(zip(blobs,native,transformed)):
                actual=session.run(None,{session.get_inputs()[0].name:blob})[0]
                rows.append({"index":i,"image_sha256":plan["cases"][i]["image_sha256"],
                    "transformed_torch_vs_original":compare(reference,changed,info["paint_threshold"]),
                    "ORT_vs_original_torch":compare(reference,actual,info["paint_threshold"]),
                    "ORT_vs_transformed_torch":compare(changed,actual,info["paint_threshold"])})
            del session
            result={"cases":rows,"all_three_vs_original_passed":all(r["ORT_vs_original_torch"]["passed"] for r in rows),
                "all_three_vs_transformed_passed":all(r["ORT_vs_transformed_torch"]["passed"] for r in rows)}
        except Exception as error:result={"cases":rows,"backend_error_type":type(error).__name__,"all_three_vs_original_passed":False,"all_three_vs_transformed_passed":False}
        result.update(learned_state_unchanged=True,onnx_sha256=sha256(onnx_path),seconds=time.perf_counter()-start)
        results["variants"][mode]=result;atomic_json(output/"precision-results.json",results)
        print(json.dumps({"mode":mode,"original_pass":result["all_three_vs_original_passed"],"transformed_pass":result["all_three_vs_transformed_passed"],
            "mismatches":[r["ORT_vs_original_torch"]["mismatch_elements"] for r in rows],"seconds":result["seconds"]}),flush=True)
        del variant
    if sha256(checkpoint)!=plan["checkpoint_sha256"] or sha256(manifest_path)!=plan["original_manifest_sha256"]:raise ValueError("original_source_changed_during_diagnostics")
    if torch.cuda.is_initialized():raise ValueError("unexpected_GPU_initialization")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["prepare","run"]);parser.add_argument("output");args=parser.parse_args()
    prepare(args.output) if args.action=="prepare" else run(args.output)
