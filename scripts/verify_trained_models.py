"""CPU-only research artifact verification. Numerical parity is not accuracy."""
from pathlib import Path
import argparse,hashlib,json,math,os,re,subprocess,sys,tempfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
EXPECTED={"lane":["background","paint"],"pothole":["pothole"],"redaction":["face","license_plate"]}

def default_training_python(os_name=None):
    return ROOT/(".venv-training/python.exe" if (os.name if os_name is None else os_name)=="nt" else ".venv-training/bin/python")

def scoped(value,*,exists=True,portable=False):
    if not isinstance(value,(str,Path)):raise ValueError("path_required")
    value=str(value)
    if portable and ("\\" in value or ":" in value or Path(value).is_absolute()):raise ValueError("project_relative_posix_path_required")
    path=Path(value);path=(ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):raise ValueError("path_outside_workspace")
    if exists and not path.is_file():raise ValueError("artifact_file_missing")
    return path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def read(path):
    if Path(path).stat().st_size>20_000_000:raise ValueError("metadata_too_large")
    return json.loads(Path(path).read_text(encoding="utf-8"))

def validate_architecture(info):
    """Validate reconstructible structure, including non-parameter operators."""
    task=info.get("task");architecture=info.get("architecture")
    if task=="lane":
        if "architecture" in info and architecture!="deeplabv3_mobilenet_v3_large":raise ValueError("unsupported_lane_architecture")
        rates=info.get("aspp_dilations",[12,24,36])
        if not isinstance(rates,(list,tuple)) or len(rates)!=3 or any(type(x) is not int or not 1<=x<=64 for x in rates):raise ValueError("invalid_lane_aspp_dilations")
    elif task in ("pothole","redaction"):
        tiled=architecture=="two YOLOX-Nano specialists global plus four tiles"
        if not tiled and any(key in info for key in ("shared_input_size","face_tile_grid")):raise ValueError("unexpected_tiled_geometry")
        if "architecture" not in info or architecture=="YOLOX-Nano":return
        if task!="redaction" or architecture not in ("two YOLOX-Nano specialists","two YOLOX-Nano specialists global plus four tiles"):raise ValueError("unsupported_detector_architecture")
        if not isinstance(info.get("input_size"),list) or len(info["input_size"])!=2 or any(type(x) is not int for x in info["input_size"]):raise ValueError("invalid_specialist_geometry")
        if type(info.get("face_input_size")) is not int or info["face_input_size"]!=960 or type(info.get("plate_input_size")) is not int or info["plate_input_size"]!=640:raise ValueError("invalid_specialist_geometry")
        if tiled:
            if info.get("input_size")!=[1920,1920] or type(info.get("shared_input_size")) is not int or info["shared_input_size"]!=1920 or type(info.get("face_tile_grid")) is not int or info["face_tile_grid"]!=2:raise ValueError("invalid_tiled_geometry")
        elif info.get("input_size")!=[960,960]:raise ValueError("invalid_specialist_geometry")
    else:raise ValueError("unsupported_model_task")

def build_torch_reference(info,yolox_source):
    validate_architecture(info)
    if info["task"]=="lane":
        from agent.lane_network import make_lane_network
        return make_lane_network(pretrained=False,aspp_dilations=info.get("aspp_dilations",[12,24,36]))
    source=scoped(yolox_source,exists=False)
    if info.get("architecture")=="two YOLOX-Nano specialists global plus four tiles":
        from agent.redaction_tiled_network import build_redaction_tiled_network
        return build_redaction_tiled_network(yolox_root=source,shared_input_size=info["shared_input_size"],face_input_size=info["face_input_size"],plate_input_size=info["plate_input_size"])
    if info.get("architecture")=="two YOLOX-Nano specialists":
        from agent.redaction_network import build_redaction_network
        return build_redaction_network(yolox_root=source,face_input_size=info["face_input_size"],plate_input_size=info["plate_input_size"])
    sys.path.insert(0,str(source))
    from yolox.exp import get_exp
    exp=get_exp(exp_file=str(source/"exps/default/yolox_nano.py"));exp.num_classes=len(info["classes"])
    model=exp.get_model();model.head.decode_in_inference=True
    return model

def check_model_output(info,array):
    check_output(info["task"],info["classes"],array)
    if info.get("architecture")=="two YOLOX-Nano specialists" and tuple(array.shape)!=(1,27300,7):raise ValueError("invalid_specialist_output_shape")
    if info.get("architecture")=="two YOLOX-Nano specialists global plus four tiles" and tuple(array.shape)!=(1,102900,7):raise ValueError("invalid_tiled_output_shape")

def validate_metadata(path):
    from datetime import datetime,timezone
    path=scoped(path);m=read(path);task=m.get("task")
    if m.get("format")!="onnx" or task not in EXPECTED or m.get("classes")!=EXPECTED[task]:raise ValueError("task_class_format_mismatch")
    validate_architecture(m)
    for key,value in (("target_trained",True),("research_only",True),("production_approved",False),("project_acceptance_eligible",False),("test_split_used",False)):
        if m.get(key) is not value:raise ValueError("invalid_research_flag:"+key)
    if type(m.get("independent_route_verified")) is not bool:raise ValueError("independent_route_disclosure_required")
    if m.get("data_revoked") or m.get("needs_revalidation"):raise ValueError("model_revoked_or_needs_revalidation")
    if not isinstance(m.get("source_licenses"),list) or not m["source_licenses"] or any(not isinstance(x,str) or not x.strip() for x in m["source_licenses"]):raise ValueError("source_license_disclosure_required")
    if not isinstance(m.get("input_size"),list) or len(m["input_size"])!=2 or any(type(x) is not int or not 16<=x<=2048 for x in m["input_size"]):raise ValueError("invalid_input_size")
    if not re.fullmatch("[a-f0-9]{64}",str(m.get("sha256",""))):raise ValueError("invalid_weight_sha256")
    weight=scoped(m["weights"],exists=False,portable=True)
    if weight.suffix.lower()!=".onnx":raise ValueError("onnx_weight_path_required")
    if task!="lane" and m.get("output_layout")!="yolox_decoded_xywh":raise ValueError("decoded_yolox_contract_required")
    for key in ("confidence","nms_iou"):
        value=m.get(key,.35 if key=="confidence" else .5)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError("invalid_probability_threshold:"+key)
    if task=="lane":
        for key,default in (("mean",[.485,.456,.406]),("std",[.229,.224,.225])):
            values=m.get(key,default)
            if not isinstance(values,list) or len(values)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or key=="std" and x<=0 for x in values):raise ValueError("invalid_normalization:"+key)
        threshold=m.get("paint_threshold",.5)
        if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<threshold<1:raise ValueError("invalid_paint_threshold")
        if m.get("threshold_calibration"):
            c=m["threshold_calibration"]
            if set(c)-{"selected_on","report_sha256","validation_samples","test_images_used"} or c.get("selected_on")!="validation" or c.get("test_images_used")!=0 or type(c.get("validation_samples")) is not int or c["validation_samples"]<1 or not re.fullmatch("[a-f0-9]{64}",str(c.get("report_sha256",""))):raise ValueError("validation_only_threshold_calibration_required")
    source=scoped(m["source_release_path"],portable=True);release=read(source)
    provenance_path=scoped(m["training_provenance"],portable=True);provenance=read(provenance_path)
    if provenance.get("test_split_exported") is not False or provenance.get("production_approved") is not False:raise ValueError("invalid_training_split_declaration")
    if provenance.get("data_revoked") or provenance.get("needs_revalidation"):raise ValueError("provenance_revoked")
    if isinstance(release,list):
        source_hash=digest(source);counts={split:sum(row.get("split")==split for row in release) for split in ("train","validation","test")}
        byid={row["sample_id"]:row for row in release}
        splits=[byid.get(asset.get("sample_id"),{}).get("split") for asset in provenance.get("assets",[])]
    elif isinstance(release,dict) and release.get("format")=="model-release-summary-v1":
        source_hash=release["original_release_sha256"];counts=release["split_counts"];splits=[asset.get("training_split") for asset in provenance.get("assets",[])]
        if m.get("release_summary_sha256")!=digest(source):raise ValueError("release_summary_hash_mismatch")
    else:raise ValueError("invalid_release_metadata")
    if m.get("dataset_release")!="manifest-sha256:"+source_hash or provenance.get("release_sha256")!=source_hash:raise ValueError("dataset_provenance_hash_mismatch")
    if any(type(counts.get(split)) is not int or counts[split]<1 for split in ("train","validation","test")):raise ValueError("three_nonempty_splits_required")
    if not splits or any(split not in ("train","validation") for split in splits):raise ValueError("test_or_unknown_asset_in_training")
    for asset in provenance["assets"]:
        expiry=datetime.fromisoformat(asset["expires_at"].replace("Z","+00:00"))
        if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc) or asset.get("revoked_at"):raise ValueError("source_expired_or_revoked")
        if not asset.get("identity_hashes") or any(not re.fullmatch("[a-f0-9]{64}",x) for x in asset["identity_hashes"]):raise ValueError("source_identity_hash_required")
    return {"manifest":m,"manifest_path":path,"weight_path":weight,"provenance":provenance,"source_hash":source_hash,"split_counts":counts}

def check_output(task,classes,array):
    import numpy as np
    if not array.size or not np.isfinite(array).all():raise ValueError("non_finite_or_empty_output")
    if task=="lane":
        if array.ndim!=4 or array.shape[0]!=1 or array.shape[1]!=2:raise ValueError("invalid_lane_output_shape")
    else:
        if array.ndim!=3 or array.shape[0]!=1 or array.shape[2]!=5+len(classes) or not array.shape[1]:raise ValueError("invalid_yolox_output_shape")
        if (array[:,:,2:4]<0).any() or (array[:,:,4:]<0).any() or (array[:,:,4:]>1).any():raise ValueError("invalid_decoded_box_or_probability")

def worker(args):
    import numpy as np
    info=read(scoped(args.manifest));blob=np.load(scoped(args.blob),allow_pickle=False)
    if args.worker in ("ort","inspect"):
        import onnx,onnxruntime as ort
        path=scoped(info["weights"]);graph=onnx.load(str(path),load_external_data=False)
        if any(t.data_location==onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):raise ValueError("external_tensor_data_not_supported")
        onnx.checker.check_model(graph,full_check=True)
        if len(graph.graph.input)!=1 or len(graph.graph.output)!=1:raise ValueError("single_input_output_required")
        if args.worker=="inspect":
            scoped(args.worker_report,exists=False).write_text(json.dumps({"onnx_checked":True}),encoding="utf-8");return
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
        if len(session.get_inputs())!=1 or len(session.get_outputs())!=1:raise ValueError("single_input_output_required")
        output=session.run(None,{session.get_inputs()[0].name:blob})[0]
        metadata={"onnx_checked":True,"operators":sorted({n.op_type for n in graph.graph.node}),"opsets":[{"domain":x.domain,"version":x.version} for x in graph.opset_import],"onnxruntime":ort.__version__,"input_shape":session.get_inputs()[0].shape,"output_shape":list(output.shape)}
    else:
        import torch
        torch.set_num_threads(2)
        model=build_torch_reference(info,args.yolox_source)
        state=torch.load(scoped(args.checkpoint),map_location="cpu",weights_only=True)
        state=state.get("model",state) if isinstance(state,dict) else state
        if all(key.startswith("module.") for key in state):state={key[7:]:value for key,value in state.items()}
        model.load_state_dict(state,strict=True);model.eval().cpu()
        with torch.inference_mode():output=model(torch.from_numpy(blob));output=output["out"] if isinstance(output,dict) else output;output=output.numpy()
        metadata={"torch":torch.__version__,"device":"cpu","output_shape":list(output.shape),"architecture":info.get("architecture","legacy-task-default")}
        if info["task"]=="lane":metadata["aspp_dilations"]=info.get("aspp_dilations",[12,24,36])
    check_model_output(info,output)
    np.save(scoped(args.result,exists=False),output,allow_pickle=False)
    scoped(args.worker_report,exists=False).write_text(json.dumps(metadata),encoding="utf-8")

def compare(a,b,rtol,atol):
    import numpy as np
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("comparison_shape_or_finite_mismatch")
    matched=bool(np.allclose(a,b,rtol=rtol,atol=atol));difference=np.abs(a-b)
    aa=a.astype(np.float64).ravel();bb=b.astype(np.float64).ravel();norm=float(np.linalg.norm(aa)*np.linalg.norm(bb))
    relative=float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-12))
    return {"passed":matched,"max_abs_error":float(difference.max()),"mean_abs_error":float(difference.mean()),"p99_abs_error":float(np.quantile(difference,.99)),"rmse":float(np.sqrt(np.mean((aa-bb)**2))),"cosine_similarity":float(np.clip(np.dot(aa,bb)/norm,-1,1)) if norm else None,"relative_l2_error":relative,"rtol":rtol,"atol":atol}

def postprocess_detection(manifest,raw,image):
    """Replay reference tensors through the real application postprocessor."""
    from agent.models import OnnxModel
    class FixedOutput:
        def setInput(self,value):pass
        def forward(self):return raw
    model=object.__new__(OnnxModel);model.manifest=manifest;model.net=FixedOutput()
    return model.predict(image)

def compare_detections(application,reference,*,bbox_atol=.05,confidence_atol=.0001):
    from collections import Counter
    def iou(a,b):
        ax,ay,aw,ah=a;bx,by,bw,bh=b
        inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
        union=aw*ah+bw*bh-inter
        return inter/union if union>0 else 0.
    available=set(range(len(reference)));errors=[];scores=[];overlaps=[]
    for item in sorted(application,key=lambda x:(x["class"],-x["confidence"],*x["bbox_xywh_px"])):
        choices=[index for index in available if reference[index]["class"]==item["class"]]
        if not choices:continue
        index=max(choices,key=lambda j:iou(item["bbox_xywh_px"],reference[j]["bbox_xywh_px"]))
        candidate=reference[index];available.remove(index)
        errors.append(max(abs(a-b) for a,b in zip(item["bbox_xywh_px"],candidate["bbox_xywh_px"])))
        scores.append(abs(item["confidence"]-candidate["confidence"]));overlaps.append(iou(item["bbox_xywh_px"],candidate["bbox_xywh_px"]))
    ac=dict(sorted(Counter(item["class"] for item in application).items()));rc=dict(sorted(Counter(item["class"] for item in reference).items()))
    matched=len(errors);passed=ac==rc and len(application)==len(reference)==matched and max(errors,default=0)<=bbox_atol and max(scores,default=0)<=confidence_atol
    return {"passed":passed,"application_count":len(application),"reference_count":len(reference),"application_class_counts":ac,"reference_class_counts":rc,"matched_count":matched,"unmatched_application":len(application)-matched,"unmatched_reference":len(available),"min_matched_iou":min(overlaps) if overlaps else None,"max_abs_bbox_error_px":max(errors) if errors else None,"max_abs_confidence_error":max(scores) if scores else None,"bbox_atol_px":bbox_atol,"confidence_atol":confidence_atol,"no_positive_predictions":not application and not reference}

def verify(args):
    validated=validate_metadata(args.manifest);m=validated["manifest"]
    result={"schema_version":1,"task":m["task"],"classes":m["classes"],"manifest_sha256":digest(validated["manifest_path"]),"weight_sha256":m["sha256"],"research_only":True,"production_approved":False,"project_acceptance_eligible":False,"source_licenses":m["source_licenses"],"split_counts":validated["split_counts"],"independent_route_verified":m["independent_route_verified"],"accuracy_evaluation_performed":False,"remaining_prerequisites":["independent ZED 2i vehicle evaluation","license-compatible intended use","agency acceptance"],"metadata_valid":True}
    result["weight_hash_verified"]=False
    if args.metadata_only:return result
    if not validated["weight_path"].is_file() or digest(validated["weight_path"])!=m["sha256"]:raise ValueError("weight_hash_mismatch_or_missing")
    result["weight_hash_verified"]=True
    if not math.isfinite(args.rtol) or not math.isfinite(args.atol) or not 0<=args.rtol<=.01 or not 0<=args.atol<=.01:raise ValueError("finite_bounded_parity_tolerances_required")
    import cv2,numpy as np
    from agent.models import OnnxModel
    cv2.setNumThreads(2)
    if args.image:
        image=cv2.imread(str(scoped(args.image)))
        if image is None:raise ValueError("image_decode_failed")
    else:
        image=np.random.default_rng(2026).integers(0,256,(240,400,3),dtype=np.uint8)
    from scripts.setup_training import environment
    env=environment();env["CUDA_VISIBLE_DEVICES"]=""
    cache=ROOT/".cache/model-verification";cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        folder=Path(temporary);blob=folder/"inspect.npy";np.save(blob,np.zeros((1,3,16,16),np.float32),allow_pickle=False)
        command=[str(scoped(args.training_python)),str(Path(__file__).resolve()),str(validated["manifest_path"]),"--worker","inspect","--blob",str(blob),"--worker-report",str(folder/"inspect.json")]
        if subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60).returncode:raise ValueError("onnx_graph_preflight_failed")
    adapter=OnnxModel(str(validated["manifest_path"]))
    class Capture:
        def __init__(self,net):self.net=net
        def setInput(self,value):self.blob=value.copy();self.net.setInput(value)
        def forward(self):self.output=self.net.forward();return self.output
    capture=Capture(adapter.net);adapter.net=capture;prediction=adapter.predict(image)
    original=capture.output.copy();adapter.predict(image)
    result["app_repeatability"]=compare(original,capture.output,args.rtol,args.atol);capture.output=original
    check_model_output(m,capture.output)
    width,height=m["input_size"];scale=min(width/image.shape[1],height/image.shape[0]);rw,rh=round(image.shape[1]*scale),round(image.shape[0]*scale)
    canvas=np.full((height,width,3),114,np.uint8);canvas[:rh,:rw]=cv2.resize(image,(rw,rh))
    if m["task"]=="lane":expected=(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).astype(np.float32)/255-np.array(m.get("mean",[.485,.456,.406]),np.float32))/np.array(m.get("std",[.229,.224,.225]),np.float32)
    else:expected=canvas.astype(np.float32)
    np.testing.assert_allclose(capture.blob,expected.transpose(2,0,1)[None],rtol=0,atol=1e-6)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        folder=Path(temporary);blob=folder/"input.npy";np.save(blob,capture.blob,allow_pickle=False)
        raw_outputs={"app":capture.output}
        for mode in ("ort","torch"):
            if mode=="torch" and not args.checkpoint:continue
            output=folder/(mode+".npy");report=folder/(mode+".json")
            command=[str(scoped(args.training_python)),str(Path(__file__).resolve()),str(validated["manifest_path"]),"--worker",mode,"--blob",str(blob),"--result",str(output),"--worker-report",str(report),"--yolox-source",str(args.yolox_source)]
            if args.checkpoint:command.extend(["--checkpoint",str(scoped(args.checkpoint))])
            completed=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=300)
            if completed.returncode:raise ValueError("cpu_worker_failed:"+mode)
            raw=np.load(output,allow_pickle=False);result[mode+"_metadata"]=read(report);result["app_vs_"+mode]=compare(capture.output,raw,args.rtol,args.atol)
            raw_outputs[mode]=raw
        if args.checkpoint:result["ort_vs_torch"]=compare(raw_outputs["ort"],raw_outputs["torch"],args.rtol,args.atol)
        if m["task"]=="lane":
            masks={}
            for mode,raw in raw_outputs.items():
                threshold=m.get("paint_threshold",.5);mask=((raw[0,1]-raw[0,0])>math.log(threshold/(1-threshold))).astype(np.uint8)
                ch=max(1,round(rh*mask.shape[0]/height));cw=max(1,round(rw*mask.shape[1]/width))
                masks[mode]=cv2.resize(mask[:ch,:cw],(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
            result["lane_adapter_postprocess_matches_reference"]=bool(np.array_equal(prediction,masks["app"]))
            result["mask_agreement"]={mode:{"pixel_agreement":float((prediction==mask).mean()),"paint_pixels":int(mask.sum()),"paint_iou_against_app":float(((prediction==1)&(mask==1)).sum()/((prediction==1)|(mask==1)).sum()) if ((prediction==1)|(mask==1)).any() else None} for mode,mask in masks.items()}
        else:
            result["detection_postprocess_parity"]={mode:compare_detections(prediction,postprocess_detection(m,raw,image)) for mode,raw in raw_outputs.items() if mode!="app"}
            result["postprocess_parity_passed"]=all(value["passed"] for value in result["detection_postprocess_parity"].values())
    result.update(runtime_python=sys.version.split()[0],opencv_preprocessing_version=cv2.__version__,app_backend=adapter.backend,app_backend_version=adapter.backend_version,finite_output=True,output_shape=list(capture.output.shape),preprocess_contract_verified=True,adapter_result_count=int((prediction==1).sum()) if m["task"]=="lane" else len(prediction),synthetic_input=args.image is None,torch_checkpoint_compared=bool(args.checkpoint),production_approved=False)
    result["numerical_parity_passed"]=all(value["passed"] for key,value in result.items() if key.startswith("app_vs_") or key in ("ort_vs_torch","app_repeatability"))
    result["artifact_contract_verified"]=result["numerical_parity_passed"] and result.get("postprocess_parity_passed",True)
    result["status"]="artifact_verified" if result["artifact_contract_verified"] else "numerical_mismatch" if not result["numerical_parity_passed"] else "postprocess_mismatch"
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("manifest");parser.add_argument("--metadata-only",action="store_true");parser.add_argument("--image");parser.add_argument("--checkpoint");parser.add_argument("--training-python",default=str(default_training_python()));parser.add_argument("--yolox-source",default=str(ROOT/".cache/vendor/YOLOX"));parser.add_argument("--output");parser.add_argument("--rtol",type=float,default=.001);parser.add_argument("--atol",type=float,default=.0001);parser.add_argument("--worker",choices=["ort","torch","inspect"]);parser.add_argument("--blob");parser.add_argument("--result");parser.add_argument("--worker-report");args=parser.parse_args()
    try:
        if args.worker:worker(args);return 0
        result=verify(args)
        if args.output:
            path=scoped(args.output,exists=False);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result,indent=2));return 0 if result.get("artifact_contract_verified",result.get("numerical_parity_passed",True)) else 2
    except Exception as error:
        print(json.dumps({"status":"verification_failed","reason":str(error)[:200],"production_approved":False}));return 2

if __name__=="__main__":raise SystemExit(main())
