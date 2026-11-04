"""Real application CPU validation of a checkpoint-derived FP64 global mean.

Writes only a new research candidate and VAL-only evidence. No TEST, gradient,
threshold selection, original metadata update or runtime/factory edits.
"""
import argparse,json,os,shutil,subprocess,sys,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256

def read(path):return json.loads(path.read_text(encoding="utf-8"))

def create_candidate(folder,head=False):
    folder=workspace_path(folder)
    if folder.exists():raise ValueError("fresh_candidate_directory_required")
    diagnostic=ROOT/"data/training/lane-numeric-diagnostics-v3"
    if head:
        study=ROOT/"data/training/lane-head-precision-diagnostics-v3";results=read(study/"head-results.json");selected=results
        if not selected["all_three_raw_passed"] or not selected["all_three_mask_identical"] or not selected["learned_state_byte_identical"]:
            raise ValueError("preliminary_original_checkpoint_parity_required")
        graph=study/"lane.onnx";recipe=ROOT/"scripts/diagnose_lane_head_precision.py"
    else:
        results=read(diagnostic/"precision-results.json");selected=results["variants"]["pool_fp64"]
        if not selected["all_three_vs_original_passed"] or not selected["learned_state_unchanged"]:raise ValueError("preliminary_original_checkpoint_parity_required")
        graph=diagnostic/"pool_fp64.onnx";recipe=ROOT/"scripts/diagnose_lane_export_precision.py"
    original_path=ROOT/"data/training/lane-a2d2-v2/lane.json";original=read(original_path)
    folder.mkdir(parents=True)
    if sha256(graph)!=selected["onnx_sha256"]:raise ValueError("diagnostic_graph_changed")
    shutil.copyfile(graph,folder/"lane.onnx")
    provenance=read(workspace_path(original["training_provenance"],exists=True))
    provenance["models"]=[(folder/"lane.json").relative_to(ROOT).as_posix()]
    atomic_json(folder/"provenance.json",provenance)
    candidate=dict(original,weights=(folder/"lane.onnx").relative_to(ROOT).as_posix(),sha256=sha256(folder/"lane.onnx"),
        training_provenance=(folder/"provenance.json").relative_to(ROOT).as_posix(),
        original_model_manifest_sha256=sha256(original_path),trained_checkpoint_sha256=results["checkpoint_sha256"],
        export_transform={"kind":"global1x1_FP64_mean_then_original_dtype"+("_and_head_FP64_MatMul" if head else ""),"global_pool_sites":9,
                          "learned_state_unchanged":True,"input_output_dtype":"float32","internal_global_mean_accumulation_dtype":"float64"},
        export_recipe_sha256=sha256(recipe),no_new_gradient_steps=True,
        inherited_validation_threshold_no_reselection=True,production_approved=False,project_acceptance_eligible=False)
    atomic_json(folder/"lane.json",candidate)
    atomic_json(folder/"artifact-status.json",{"status":"pending_canonical_and_full_validation_proof","accuracy_acceptance_passed":False,
        "checkpoint_sha256":results["checkpoint_sha256"],"original_failure_preserved":True})
    plan=read(diagnostic/"validation-plan.json")
    atomic_json(folder/"source.json",{"expires_at":min(c["source_expires_at"] for c in plan["cases"]),"source_expiry_extended":False})
    print(json.dumps({"status":"research_candidate_created_unpromoted","manifest_sha256":sha256(folder/"lane.json"),"weights_sha256":candidate["sha256"]}),flush=True)

def worker(manifest):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import cv2,numpy as np,torch,onnxruntime as ort
    from agent.lane_network import make_lane_network
    from scripts.verify_trained_models import compare
    info=read(workspace_path(manifest,exists=True));torch.set_num_threads(2);cv2.setNumThreads(2)
    model=make_lane_network(pretrained=False,aspp_dilations=info["aspp_dilations"]).eval().cpu()
    checkpoint=ROOT/"data/training/lane-a2d2-v2/lane_best.pt"
    if sha256(checkpoint)!=info["trained_checkpoint_sha256"]:raise ValueError("checkpoint_changed")
    state=torch.load(checkpoint,map_location="cpu",weights_only=True);model.load_state_dict(state.get("model",state),strict=True)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(workspace_path(info["weights"],exists=True)),sess_options=options,providers=["CPUExecutionProvider"])
    print(json.dumps({"status":"ready","torch":torch.__version__,"ort":ort.__version__,"GPU_initialized":torch.cuda.is_initialized()}),flush=True)
    for line in sys.stdin:
        item=json.loads(line)
        if item.get("stop"):break
        blob=np.load(workspace_path(item["blob"],exists=True),allow_pickle=False)
        app=np.load(workspace_path(item["raw"],exists=True),allow_pickle=False)
        app_mask=np.load(workspace_path(item["mask"],exists=True),allow_pickle=False)
        with torch.inference_mode():native=model(torch.from_numpy(blob))["out"].numpy()
        reference=session.run(None,{session.get_inputs()[0].name:blob})[0]
        comparisons={"app_vs_original_torch":compare(app,native,.001,.0001),"ORT_reference_vs_original_torch":compare(reference,native,.001,.0001),
                     "app_vs_ORT_reference":compare(app,reference,.001,.0001)}
        mismatch={key:int((~np.isclose(a,b,rtol=.001,atol=.0001)).sum()) for key,(a,b) in
            {"app_vs_original_torch":(app,native),"ORT_reference_vs_original_torch":(reference,native),"app_vs_ORT_reference":(app,reference)}.items()}
        h,w=item["original_size"];rh,rw=item["resized_size"];masks={}
        for name,raw in (("app",app),("original_torch",native),("ORT_reference",reference)):
            mask=((raw[0,1]-raw[0,0])>math.log(info["paint_threshold"]/(1-info["paint_threshold"]))).astype(np.uint8)
            cropped=mask[:max(1,round(rh*mask.shape[0]/info["input_size"][1])),:max(1,round(rw*mask.shape[1]/info["input_size"][0]))]
            masks[name]=cv2.resize(cropped,(w,h),interpolation=cv2.INTER_NEAREST)
        post={name:{"matches_actual_adapter":bool(np.array_equal(mask,app_mask)),"pixel_agreement":float((mask==app_mask).mean())} for name,mask in masks.items()}
        print(json.dumps({"index":item["index"],"raw":comparisons,"mismatch_elements":mismatch,"postprocess":post,
            "raw_passed":all(v["passed"] for v in comparisons.values()),"post_passed":all(v["matches_actual_adapter"] for v in post.values())}),flush=True)

def validate_all(folder,training_python=None):
    import cv2,numpy as np
    from agent.models import OnnxModel
    from scripts.setup_training import environment,PYTHON
    folder=workspace_path(folder,exists=True);manifest_path=folder/"lane.json";info=read(manifest_path)
    source=workspace_path(info["training_development_release_path"],exists=True)
    if sha256(source)!=info["training_development_release_sha256"]:raise ValueError("development_manifest_changed")
    rows=sorted((r for r in read(source) if r["split"]=="validation"),key=lambda r:r["source_id"])
    if len(rows)!=100:raise ValueError("complete_100_validation_rows_required")
    plan_path=folder/"full-validation-plan.json"
    if plan_path.exists() or (folder/"full-validation-results.json").exists():raise ValueError("full_validation_outputs_already_exist")
    atomic_json(plan_path,{"role":"all_100_VAL_numerical_equivalence_only","manifest_sha256":sha256(manifest_path),
        "development_manifest_sha256":sha256(source),"rtol":.001,"atol":.0001,"threshold":info["paint_threshold"],"test_accessed":False,
        "cases":[{"source_id":r["source_id"],"image_sha256":r["sha256"],"path":r["path"],"expires_at":r["expires_at"]} for r in rows]})
    working=folder/"working";working.mkdir()
    env=environment();env["CUDA_VISIBLE_DEVICES"]=""
    log=(folder/"full-validation-worker.log").open("w",encoding="utf-8")
    reference_python=workspace_path(training_python,exists=True) if training_python else PYTHON
    process=subprocess.Popen([str(reference_python),str(Path(__file__)),"worker",str(manifest_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
                             cwd=ROOT,env=env,text=True,bufsize=1)
    results={"manifest_sha256":sha256(manifest_path),"weights_sha256":info["sha256"],"checkpoint_sha256":info["trained_checkpoint_sha256"],
        "plan_sha256":sha256(plan_path),"rtol":.001,"atol":.0001,"threshold":info["paint_threshold"],"test_accessed":False,"GPU_operations":0,"cases":[]}
    try:
        handshake=json.loads(process.stdout.readline());results["reference_runtimes"]=handshake
        if handshake.get("status")!="ready" or handshake.get("GPU_initialized") is not False:raise ValueError("CPU_reference_worker_not_ready")
        adapter=OnnxModel(str(manifest_path));results["app_backend"]=adapter.backend;results["app_backend_version"]=adapter.backend_version
        class Capture:
            def __init__(self,net):self.net=net
            def setInput(self,value):self.blob=value.copy();self.net.setInput(value)
            def forward(self):self.raw=self.net.forward();return self.raw
        capture=Capture(adapter.net);adapter.net=capture
        for i,row in enumerate(rows):
            image_path=workspace_path(row["path"],exists=True)
            if sha256(image_path)!=row["sha256"]:raise ValueError("validation_image_changed")
            image=cv2.imread(str(image_path));mask=adapter.predict(image)
            for name,value in (("blob",capture.blob),("raw",capture.raw),("mask",mask)):np.save(working/(name+".npy"),value,allow_pickle=False)
            h,w=image.shape[:2];scale=min(info["input_size"][0]/w,info["input_size"][1]/h)
            request={"index":i,"blob":(working/"blob.npy").relative_to(ROOT).as_posix(),"raw":(working/"raw.npy").relative_to(ROOT).as_posix(),
                "mask":(working/"mask.npy").relative_to(ROOT).as_posix(),"original_size":[h,w],"resized_size":[round(h*scale),round(w*scale)]}
            process.stdin.write(json.dumps(request)+"\n");process.stdin.flush()
            response=json.loads(process.stdout.readline());response["image_sha256"]=row["sha256"];results["cases"].append(response)
            atomic_json(folder/"full-validation-results.json",results)
            print(json.dumps({"completed_VAL":i+1,"raw_failures":sum(not r["raw_passed"] for r in results["cases"]),"post_failures":sum(not r["post_passed"] for r in results["cases"])}),flush=True)
        results["complete"]=len(results["cases"])==100
        results["all_raw_passed"]=all(r["raw_passed"] for r in results["cases"])
        results["all_post_passed"]=all(r["post_passed"] for r in results["cases"])
        if sha256(manifest_path)!=results["manifest_sha256"] or sha256(workspace_path(info["weights"]))!=results["weights_sha256"]:raise ValueError("candidate_changed_during_validation")
        atomic_json(folder/"full-validation-results.json",results)
    finally:
        try:process.stdin.write('{"stop":true}\n');process.stdin.flush()
        except (BrokenPipeError,OSError):pass
        process.stdin.close()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=30)
        process.stdout.close();log.close()
        for name in ("blob.npy","raw.npy","mask.npy"):(working/name).unlink(missing_ok=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["create","create-head","worker","validate-all"]);parser.add_argument("folder");parser.add_argument("--training-python");args=parser.parse_args()
    if args.action=="create":create_candidate(args.folder)
    elif args.action=="create-head":create_candidate(args.folder,head=True)
    elif args.action=="worker":worker(args.folder)
    else:validate_all(args.folder,args.training_python)
