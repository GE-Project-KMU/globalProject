"""One predeclared Nano candidate: official augmentation/SGD/EMA, fresh transfer.

Never edits prior runs. Model selection uses the new validation split only.
The new frozen final holdout and OLD Czech regression run only after selection.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/".cache/vendor/YOLOX"))
import cv2
import numpy as np
import torch
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release
from agent.governance import begin_training,register_asset,link_model
from train_yolox_research import make_model,evaluate_model,operating_metrics,seed_all
from yolox.data.datasets.datasets_wrapper import Dataset
from yolox.data.datasets.mosaicdetection import MosaicDetection
from yolox.data.data_augment import TrainTransform
from yolox.utils import ModelEMA,LRScheduler

class OriginalSamples(Dataset):
    def __init__(self,rows,size,cache_gib=0):
        super().__init__((size,size));self.rows=rows;self.cache={};used=0
        for index,row in enumerate(rows):
            if used>=cache_gib*2**30:break
            image=cv2.imread(str(workspace_path(row["path"],exists=True)))
            if image is None:raise ValueError("image_decode_failed")
            if used+image.nbytes>cache_gib*2**30:break
            self.cache[index]=image;used+=image.nbytes
        self.cache_bytes=used
    def __len__(self):return len(self.rows)
    def pull_item(self,index):
        row=self.rows[index]
        image=self.cache[index].copy() if index in self.cache else cv2.imread(str(workspace_path(row["path"],exists=True)))
        if image is None:raise ValueError("image_decode_failed")
        labels=np.array([[x,y,x+w,y+h,0] for obj in row["objects"] for x,y,w,h in [obj["bbox_xywh_px"]]],dtype=np.float32).reshape(-1,5)
        if len(labels)>30:raise ValueError("more_than_30_original_targets_reject_unsafe_mosaic_padding")
        return image,labels,image.shape[:2],np.array([index])

def calibration(result):
    grid=[.01,.03,.05,.1,.15,.2,.25,.35,.5]
    scores=[]
    for value in grid:
        metric=operating_metrics(result,["pothole"],value);m=metric["classes"]["pothole"]
        p,r=m["precision"],m["recall"] or 0
        metric["f2"]=5*p*r/(4*p+r) if 4*p+r else 0.;scores.append(metric)
    selected=max(scores,key=lambda x:(x["f2"],x["min_class_recall"],x["macro_precision"]))
    return {"grid":grid,"scores":scores,"selected_confidence":selected["confidence"],
            "criterion":"max validation F2; recall then precision break ties; no test tuning",
            "selection_data":"selected checkpoint NEW validation only","frozen_at":datetime.now(timezone.utc).isoformat()}

def train(args):
    seed_all(20261104)
    cv2.setNumThreads(1)
    release=workspace_path(args.release,exists=True);output=workspace_path(args.output)
    if output.exists():raise ValueError("fresh_run_directory_required")
    rows=json.loads(release.read_text());check=validate_research_release(rows)
    if not check["valid"]:raise ValueError(str(check["errors"][:10]))
    if not torch.cuda.is_available():raise RuntimeError("CUDA_required")
    if any(r["country"]=="Czech" for r in rows):raise ValueError("old_test_country_must_not_enter_new_release")
    groups={split:{r["leakage_group_id"] for r in rows if r["split"]==split} for split in ("train","validation","test")}
    if groups["train"]&groups["test"] or groups["train"]&groups["validation"] or groups["validation"]&groups["test"]:raise ValueError("group_leak")
    training=[r for r in rows if r["split"]=="train"];validation=[r for r in rows if r["split"]=="validation"];test=[r for r in rows if r["split"]=="test"]
    provenance_path,provenance=begin_training(output,release,training+validation,"pothole")
    provenance["research_only"]=True;provenance["assets"]=[register_asset(provenance,r,[]) for r in training+validation]
    atomic_json(provenance_path,provenance)
    model,transferred=make_model(["pothole"],"data/research/yolox_nano_coco.pth");model=model.cuda()
    from yolox.exp import get_exp
    exp=get_exp(exp_file=str(ROOT/".cache/vendor/YOLOX/exps/default/yolox_nano.py"));exp.model=model
    exp.warmup_epochs=3;exp.basic_lr_per_img=.01/64;optimizer=exp.get_optimizer(args.batch_size)
    original=OriginalSamples(training,args.input_size,args.cache_gib)
    dataset=MosaicDetection(original,img_size=(args.input_size,args.input_size),
        preproc=TrainTransform(max_labels=120,hsv_prob=1.,flip_prob=.5),mosaic_prob=.5,enable_mixup=False,
        degrees=10.,translate=.1,mosaic_scale=(.5,1.5),shear=2.)
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,drop_last=True,pin_memory=True)
    noaug=max(5,args.epochs//6)
    scheduler=LRScheduler("yoloxwarmcos",.01/64*args.batch_size,len(loader),args.epochs,warmup_epochs=3,
                         warmup_lr_start=0.,no_aug_epochs=noaug,min_lr_ratio=.05)
    ema=ModelEMA(model);scaler=torch.amp.GradScaler("cuda")
    config={"candidate_count":1,"candidate_id":"native-nano-expanded-D40-v4","release_sha256":sha256(release),
        "executed_trainer_script_sha256":sha256(Path(__file__).resolve()),"executed_argv":sys.argv,
        "epochs":args.epochs,"batch_size":args.batch_size,"input_size":args.input_size,"seed":20261104,
        "optimizer":"official Exp.get_optimizer SGD momentum.9 nesterov; decay5e-4 weights only, no BN/bias decay",
        "basic_lr_per_img":.01/64,"warmup_epochs":3,"no_aug_epochs":noaug,"ema":True,"l1_in_noaug":True,
        "augmentation":"official TrainTransform HSV/flip + MosaicDetection probability.5 scale.5-1.5 degree10 translate.1 shear2; no mixup",
        "multiscale":False,"checkpoint_selection":"max NEW validation AP50 every2epochs plus last/noaug epochs; never test",
        "confidence_selection":"max NEW validation F2 on frozen grid, after checkpoint selection",
        "nms_iou":.5,"ap_confidence_floor":.01,"test_used_for_selection":False,"old_test_role":"regression benchmark only",
        "source_commit":"6ddff4824372906469a7fae2dc3206c7aa4bbaee","pretrained_sha256":sha256(ROOT/"data/research/yolox_nano_coco.pth"),
        "transferred_tensors":transferred,"research_only":True,"shared_gpu_training":True,"independent_route_verified":False,
        "raw_RAM_cache_bytes":original.cache_bytes,"cv2_threads":1,
        "early_stop":{"minimum_epochs":15,"patience_epochs":12,"delta":.001,"monitored":"new validation AP50 only","no_final_test_access":True},
        "torch":torch.__version__,"cuda":torch.version.cuda,"device":torch.cuda.get_device_name()}
    atomic_json(output/"training-config.json",config)
    history=[];best=-1.;best_epoch=None;started=time.perf_counter();significant=-1.;last_significant=0;stop_reason="max_epochs"
    for epoch in range(args.epochs):
        model.train();totals=[];closing=epoch>=args.epochs-noaug
        if closing:dataset.enable_mosaic=False;model.head.use_l1=True
        for step,(images,labels,_,_) in enumerate(loader):
            lr=scheduler.update_lr(epoch*len(loader)+step+1)
            for group in optimizer.param_groups:group["lr"]=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):losses=model(images.cuda(non_blocking=True),labels.cuda(non_blocking=True));loss=losses["total_loss"]
            if not torch.isfinite(loss):raise RuntimeError("non_finite_loss")
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            scaler.step(optimizer);scaler.update();ema.update(model);totals.append(float(loss.detach()))
            if step%100==0:print(json.dumps({"epoch":epoch+1,"step":step,"loss":totals[-1],"lr":lr,"peak_cuda_MiB":torch.cuda.max_memory_allocated()/2**20}),flush=True)
        item={"epoch":epoch+1,"mean_training_loss":float(np.mean(totals)),"lr":lr,"noaug_l1":closing}
        if (epoch+1)%2==0 or closing:
            evaluated=evaluate_model(ema.ema,validation,["pothole"],size=args.input_size,batch_size=args.batch_size)
            metric=evaluated["metrics"]["map50"];item["validation_map50"]=metric
            if metric>significant+.001:significant=metric;last_significant=epoch+1
            if metric>best:
                best,best_epoch=metric,epoch+1
                torch.save({"model":ema.ema.state_dict(),"epoch":best_epoch,"classes":["pothole"]},output/"best.pt")
        history.append(item);atomic_json(output/"training-history.json",{"history":history,"best_epoch":best_epoch,"best_validation_map50":best,"elapsed_seconds":time.perf_counter()-started})
        print(json.dumps(item),flush=True)
        if epoch+1>=15 and epoch+1-last_significant>=12:
            stop_reason="predeclared_validation_plateau_min15_patience12_delta001";break
    model.load_state_dict(torch.load(output/"best.pt",map_location="cpu",weights_only=True)["model"]);model.eval()
    selected_validation=evaluate_model(model,validation,["pothole"],size=args.input_size,batch_size=args.batch_size)
    atomic_json(output/"selected-validation-predictions.json",selected_validation)
    frozen=calibration(selected_validation);atomic_json(output/"confidence-calibration.json",frozen)
    confidence=frozen["selected_confidence"]
    # Final checkpoint and operating point are immutable before these calls.
    for label,evaluation_rows in (("heldout",test),("old-czech-regression",[r for r in json.loads((ROOT/"data/training/research-pothole-rdd2020/release.json").read_text()) if r["split"]=="test"])):
        result=evaluate_model(model,evaluation_rows,["pothole"],size=args.input_size,batch_size=args.batch_size)
        result["operating_metrics"]=operating_metrics(result,["pothole"],confidence)
        result["role"]="fresh final holdout, no selection" if label=="heldout" else "previously examined regression benchmark, no independent final claim"
        atomic_json(output/(label+"-predictions.json"),result)
    heldout=json.loads((output/"heldout-predictions.json").read_text())
    model=model.cpu();model.head.decode_in_inference=True
    torch.onnx.export(model,torch.zeros(1,3,args.input_size,args.input_size),str(output/"model.onnx"),
                      input_names=["images"],output_names=["predictions"],opset_version=17,dynamo=False)
    manifest={"format":"onnx","task":"pothole","classes":["pothole"],"architecture":"YOLOX-Nano",
        "weights":(output/"model.onnx").relative_to(ROOT).as_posix(),"sha256":sha256(output/"model.onnx"),"target_trained":True,
        "dataset_release":"manifest-sha256:"+sha256(release),"source_release_path":release.relative_to(ROOT).as_posix(),
        "training_provenance":provenance_path.relative_to(ROOT).as_posix(),"input_size":[args.input_size,args.input_size],
        "output_layout":"yolox_decoded_xywh","confidence":confidence,"nms_iou":.5,"production_approved":False,"research_only":True,
        "human_privacy_reviewed":False,"project_acceptance_eligible":False,"source_licenses":["CC-BY-NC-3.0"],"independent_route_verified":False,
        "test_split_used":False,"test_evaluated":True,"test_split_used_scope":"training_and_selection","data_revoked":False,"needs_revalidation":False,
        "best_epoch":best_epoch,"heldout_map50":heldout["metrics"]["map50"]}
    atomic_json(output/"model.json",manifest);link_model(provenance_path,output/"model.json")
    atomic_json(output/"summary.json",{"best_epoch":best_epoch,"validation_map50":best,"heldout_metrics":heldout["metrics"],
        "operating_metrics":heldout["operating_metrics"],"confidence_calibration":frozen,"elapsed_seconds":time.perf_counter()-started,
        "completed_epochs":len(history),"stop_reason":stop_reason,"noaug_phase_reached":len(history)>args.epochs-noaug,
        "research_only":True,"acceptance_approved":False,"genuine_gradient_training":True,"training_images":len(training),
        "validation_images":len(validation),"test_images":len(test),"environment":config})
    print(json.dumps({"status":"completed","manifest":str(output/"model.json"),"metrics":heldout["metrics"]}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("release");parser.add_argument("output")
    parser.add_argument("--epochs",type=int,default=60);parser.add_argument("--batch-size",type=int,default=16);parser.add_argument("--input-size",type=int,default=640)
    parser.add_argument("--cache-gib",type=float,default=4.)
    args=parser.parse_args()
    if args.epochs<15 or args.batch_size<2 or args.input_size%32 or not 0<=args.cache_gib<=6:raise ValueError("invalid_training_parameters")
    train(args)
