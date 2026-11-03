"""Real official YOLOX-Nano fine-tuning with publisher research labels.

Uses YOLOX's native detector and SimOTA/loss; this portable loop avoids the
upstream distributed-launch assumptions on Windows. No test-set model selection.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/".cache/vendor/YOLOX"))
import cv2
import numpy as np
import torch
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release
from agent.governance import link_model,begin_training,register_asset


def detection_metrics(truth,predictions,classes):
    """101-point AP50, one-to-one class/image matching; CPU-only dependency."""
    results={}
    def iou(a,b):
        x,y,w,h=a;xx,yy,ww,hh=b
        area=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
        total=w*h+ww*hh-area
        return area/total if total else 0.
    for label in classes:
        gt=[row for row in truth if row["class"]==label];matched=set();hits=[]
        ordered=sorted([row for row in predictions if row["class"]==label],key=lambda row:row["confidence"],reverse=True)
        for prediction in ordered:
            score,index=max(((iou(prediction["bbox_xywh_px"],row["bbox_xywh_px"]),i) for i,row in enumerate(gt)
                if i not in matched and row["sample_id"]==prediction["sample_id"]),default=(0,None))
            hit=score>=.5;hits.append(int(hit))
            if hit:matched.add(index)
        tp=np.cumsum(hits);precision=tp/np.maximum(np.arange(len(hits))+1,1);recall=tp/len(gt) if gt else np.zeros(len(hits))
        ap=float(np.mean([max(precision[recall>=value],default=0.) for value in np.linspace(0,1,101)])) if gt else None
        results[label]={"ap50":ap,"gt_count":len(gt),"prediction_count":len(ordered),"tp":len(matched),"fp":len(ordered)-len(matched),"fn":len(gt)-len(matched)}
    values=[row["ap50"] for row in results.values() if row["ap50"] is not None]
    return {"classes":results,"map50":sum(values)/len(values) if values else None,"matching":"one_to_one_IoU0.5_AP101"}


def operating_metrics(result,classes,confidence):
    metrics=detection_metrics(result["truth"],[p for p in result["predictions"] if p["confidence"]>=confidence],classes)
    for value in metrics["classes"].values():
        value["recall"]=value["tp"]/value["gt_count"] if value["gt_count"] else None
        value["precision"]=value["tp"]/(value["tp"]+value["fp"]) if value["tp"]+value["fp"] else 0.
    values=list(metrics["classes"].values())
    metrics.update(confidence=confidence,min_class_recall=min(v["recall"] or 0. for v in values),macro_precision=float(np.mean([v["precision"] for v in values])))
    return metrics


def calibrate(validation_result,classes):
    grid=[.01,.03,.05,.1,.2,.35,.5]
    scores=[operating_metrics(validation_result,classes,value) for value in grid]
    eligible=[row for row in scores if row["min_class_recall"]>=.95]
    selected=max(eligible,key=lambda row:(row["macro_precision"],row["min_class_recall"],row["confidence"])) if eligible else max(scores,key=lambda row:(row["min_class_recall"],row["macro_precision"],row["confidence"]))
    return {"selection_data":"validation only; selected checkpoint; no test/fullscene tuning","grid":grid,
        "minimum_recall_goal":.95,"both_class_goal_achieved":bool(eligible),"scores":scores,"selected_confidence":selected["confidence"],
        "selected_metrics":selected,"frozen_at":datetime.now(timezone.utc).isoformat()}


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


class Samples(torch.utils.data.Dataset):
    def __init__(self,rows,classes,size=416,training=False,max_labels=120):self.rows,self.classes,self.size,self.training,self.max_labels=rows,classes,size,training,max_labels
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];image=cv2.imread(str(workspace_path(row["path"],exists=True)))
        if image is None:raise ValueError("image_decode_failed")
        h,w=image.shape[:2];scale=min(self.size/w,self.size/h)
        labels=[]
        for obj in row["objects"]:
            if obj["class"] not in self.classes:continue
            x,y,bw,bh=obj["bbox_xywh_px"]
            labels.append([self.classes.index(obj["class"]),(x+bw/2)*scale,(y+bh/2)*scale,bw*scale,bh*scale])
        labels=np.asarray(labels,np.float32).reshape(-1,5)
        image=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
        canvas=np.full((self.size,self.size,3),114,np.uint8);canvas[:image.shape[0],:image.shape[1]]=image
        if self.training and random.random()<.5:
            canvas=canvas[:,::-1].copy()
            if len(labels):labels[:,1]=self.size-labels[:,1]
        if self.training and random.random()<.5:
            factor=random.uniform(.75,1.25)
            canvas=np.clip(canvas.astype(np.float32)*factor,0,255).astype(np.uint8)
        packed=np.zeros((self.max_labels,5),np.float32)
        if len(labels)>self.max_labels and self.training:raise ValueError("too_many_targets_for_locked_padding_reject_entire_image")
        packed[:min(len(labels),self.max_labels)]=labels[:self.max_labels]
        return torch.from_numpy(canvas.transpose(2,0,1).copy().astype(np.float32)),torch.from_numpy(packed),index


def make_model(classes,pretrained=None):
    from yolox.exp import get_exp
    experiment=get_exp(exp_file=str(ROOT/".cache/vendor/YOLOX/exps/default/yolox_nano.py"))
    experiment.num_classes=len(classes)
    model=experiment.get_model()
    transferred=0
    if pretrained:
        # Official publisher checkpoint only, path/hash recorded in training report.
        checkpoint=torch.load(workspace_path(pretrained,exists=True),map_location="cpu",weights_only=True)
        state=checkpoint.get("model",checkpoint)
        own=model.state_dict();compatible={k:v for k,v in state.items() if k in own and own[k].shape==v.shape}
        model.load_state_dict(compatible,strict=False);transferred=len(compatible)
    return model,transferred


@torch.no_grad()
def evaluate_model(model,rows,classes,*,confidence=.01,size=416,batch_size=16,fullscene=False):
    from yolox.utils import postprocess
    model.eval();loader=torch.utils.data.DataLoader(Samples(rows,classes,size),batch_size=batch_size,shuffle=False,num_workers=0)
    truth=[];predictions=[];started=time.perf_counter();ignored=0
    for row in rows:
        truth.extend({"sample_id":row["sample_id"],"class":o["class"],"bbox_xywh_px":o["bbox_xywh_px"]} for o in row["objects"] if o["class"] in classes)
    for images,_,indices in loader:
        output=model(images.cuda())
        outputs=postprocess(output,len(classes),conf_thre=confidence,nms_thre=.5,class_agnostic=False)
        for detection,index in zip(outputs,indices.tolist()):
            if detection is None:continue
            row=rows[index];source=cv2.imread(str(workspace_path(row["path"],exists=True)));h,w=source.shape[:2];scale=min(size/w,size/h)
            for box in detection.cpu().numpy():
                label=classes[int(box[6])]
                if fullscene and label!=row["evaluation_class"]:continue
                x1,y1,x2,y2=box[:4]/scale
                x1,y1=max(0.,float(x1)),max(0.,float(y1));x2,y2=min(float(w),float(x2)),min(float(h),float(y2))
                if x2<=x1 or y2<=y1:continue
                if fullscene and row.get("ignore_bboxes_xywh_px"):
                    pred=[x1,y1,x2-x1,y2-y1]
                    def overlap(a,b):
                        x,y,w,h=a;xx,yy,ww,hh=b
                        return max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
                    valid_overlap=any(overlap(pred,obj["bbox_xywh_px"])/(pred[2]*pred[3]+obj["bbox_xywh_px"][2]*obj["bbox_xywh_px"][3]-overlap(pred,obj["bbox_xywh_px"]))>=.5 for obj in row["objects"] if obj["class"]==label)
                    if not valid_overlap and any(overlap(pred,box)/(pred[2]*pred[3])>=.5 for box in row["ignore_bboxes_xywh_px"]):ignored+=1;continue
                predictions.append({"sample_id":row["sample_id"],"class":classes[int(box[6])],
                    "bbox_xywh_px":[x1,y1,x2-x1,y2-y1],"confidence":float(box[4]*box[5])})
    return {"metrics":detection_metrics(truth,predictions,classes),"images":len(rows),"elapsed_seconds":time.perf_counter()-started,
            "confidence_floor":confidence,"nms_iou":.5,"truth":truth,"predictions":predictions,"ignored_predictions":ignored}


def train(release,output,task,*,epochs=20,batch_size=16,learning_rate=.0005,pretrained="data/research/yolox_nano_coco.pth",seed=2026,
          input_size=416,max_labels=120,known_class_evaluation=False):
    if epochs<1 or batch_size<2 or not 0<learning_rate<1:raise ValueError("invalid_training_parameters")
    if input_size%32 or not 320<=input_size<=1024 or not 1<=max_labels<=1024:raise ValueError("invalid_input_or_label_limit")
    seed_all(seed)
    if not torch.cuda.is_available():raise RuntimeError("actual_cuda_training_required")
    release=workspace_path(release,exists=True);output=workspace_path(output)
    if release.with_suffix(".quarantine.json").exists():raise ValueError("source_release_quarantined")
    rows=json.loads(release.read_text(encoding="utf-8"));validation=validate_research_release(rows)
    if not validation["valid"]:raise ValueError(json.dumps(validation["errors"][:10]))
    classes=["pothole"] if task=="pothole" else ["face","license_plate"]
    source_licenses=sorted({json.loads(workspace_path(row["license_evidence_path"],exists=True).read_text(encoding="utf-8"))["license_id"] for row in rows})
    train_rows=[r for r in rows if r["split"]=="train"];val_rows=[r for r in rows if r["split"]=="validation"];test_rows=[r for r in rows if r["split"]=="test"]
    if min(len(train_rows),len(val_rows),len(test_rows))<2:raise ValueError("independent_train_validation_test_required")
    if any(sum(o["class"] in classes for o in row["objects"])>max_labels for row in train_rows):
        raise ValueError("training_image_exceeds_max_labels_reject_not_truncate")
    # Sources already live under the controlled training root. Track their TTL
    # directly; no redundant COCO image copy or O(n²) provenance fsync loop.
    provenance_path,provenance=begin_training(output,release,train_rows+val_rows,task)
    provenance["research_only"]=True
    provenance["assets"]=[register_asset(provenance,row,[]) for row in train_rows+val_rows]
    atomic_json(provenance_path,provenance)
    model,transferred=make_model(classes,pretrained);model=model.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=5e-4)
    loader=torch.utils.data.DataLoader(Samples(train_rows,classes,size=input_size,training=True,max_labels=max_labels),batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True,pin_memory=True)
    scaler=torch.amp.GradScaler("cuda")
    started=time.perf_counter();history=[];best=-1.;best_epoch=None
    environment={"torch":torch.__version__,"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(),
        "architecture":"official YOLOX-Nano depth=.33,width=.25,depthwise=True","source_commit":"6ddff4824372906469a7fae2dc3206c7aa4bbaee",
        "nano_source_sha256":sha256(ROOT/".cache/vendor/YOLOX/exps/default/yolox_nano.py"),"pretrained_source":"official YOLOX0.1.1rc0 COCO Nano transfer, not target-trained substitute",
        "pretrained_sha256":sha256(workspace_path(pretrained,exists=True)),"transferred_parameter_tensors":transferred}
    atomic_json(output/"training-config.json",{"epochs":epochs,"batch_size":batch_size,"learning_rate":learning_rate,"seed":seed,
        "release_sha256":sha256(release),"classes":classes,"environment":environment,"research_only":True,"test_used_for_selection":False,
        "input_size":input_size,"max_labels":max_labels,"known_class_evaluation":known_class_evaluation,
        "partial_annotation_limit":"Unknown other-class objects can create false-negative objectness supervision; research-only, no exhaustive privacy coverage claim." if known_class_evaluation else None})
    for epoch in range(epochs):
        model.train();totals=[]
        lr=learning_rate*(.1+.9*(1+math.cos(math.pi*epoch/max(1,epochs-1)))/2)
        for group in optimizer.param_groups:group["lr"]=lr
        for iteration,(images,labels,_) in enumerate(loader):
            images,labels=images.cuda(non_blocking=True),labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):losses=model(images,labels);loss=losses["total_loss"]
            if not torch.isfinite(loss):raise RuntimeError("non_finite_training_loss")
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            scaler.step(optimizer);scaler.update();totals.append(float(loss.detach()))
            if iteration%50==0:print(json.dumps({"epoch":epoch+1,"step":iteration,"loss":totals[-1],"lr":lr}),flush=True)
        validation_result=evaluate_model(model,val_rows,classes,size=input_size,fullscene=known_class_evaluation)
        metric=validation_result["metrics"]["map50"]
        item={"epoch":epoch+1,"mean_training_loss":float(np.mean(totals)),"validation_map50":metric,"lr":lr}
        history.append(item);print(json.dumps(item),flush=True)
        if metric is not None and metric>best:
            best,best_epoch=metric,epoch+1
            torch.save({"model":model.state_dict(),"epoch":best_epoch,"classes":classes},output/"best.pt")
        atomic_json(output/"training-history.json",{"history":history,"elapsed_seconds":time.perf_counter()-started,"best_epoch":best_epoch,"best_validation_map50":best})
    checkpoint=torch.load(output/"best.pt",map_location="cpu",weights_only=True);model.load_state_dict(checkpoint["model"]);model.eval()
    final_validation=evaluate_model(model,val_rows,classes,size=input_size,fullscene=known_class_evaluation)
    atomic_json(output/"selected-validation-predictions.json",final_validation)
    calibration=calibrate(final_validation,classes) if task=="redaction" else {"selection_data":"fixed SRS confidence; no test tuning","selected_confidence":.35,"frozen_at":datetime.now(timezone.utc).isoformat()}
    atomic_json(output/"confidence-calibration.json",calibration)
    operating_confidence=calibration["selected_confidence"]
    heldout=evaluate_model(model,test_rows,classes,size=input_size,fullscene=known_class_evaluation)
    heldout["operating_metrics"]=operating_metrics(heldout,classes,operating_confidence)
    atomic_json(output/"heldout-predictions.json",heldout)
    challenge_path=ROOT/"data/training/research-redaction/fullscene-test.json"
    if task=="redaction" and challenge_path.exists():
        challenge_rows=json.loads(challenge_path.read_text(encoding="utf-8"))
        challenge=evaluate_model(model,challenge_rows,classes,size=input_size,fullscene=True)
        challenge["operating_metrics"]=operating_metrics(challenge,classes,operating_confidence)
        challenge["source_release_sha256"]=sha256(challenge_path);challenge["independent_route_verified"]=False
        challenge["previously_examined_v2_challenge"]=known_class_evaluation
        challenge["limits"]="Only publisher-known class per image evaluated; unknown other class not counted negative; invalid face regions ignored. Same frozen confidence as validation calibration. Not route/privacy acceptance."
        atomic_json(output/"fullscene-heldout-predictions.json",challenge)
    model=model.cpu();model.head.decode_in_inference=True
    onnx=output/"model.onnx"
    torch.onnx.export(model,torch.zeros(1,3,input_size,input_size),str(onnx),input_names=["images"],output_names=["predictions"],opset_version=17,dynamo=False)
    manifest={"format":"onnx","task":task,"classes":classes,"architecture":"YOLOX-Nano","weights":onnx.relative_to(ROOT).as_posix(),"sha256":sha256(onnx),
        "target_trained":True,"dataset_release":"manifest-sha256:"+sha256(release),"source_release_path":release.relative_to(ROOT).as_posix(),
        "training_provenance":(output/"provenance.json").relative_to(ROOT).as_posix(),"input_size":[input_size,input_size],"output_layout":"yolox_decoded_xywh",
        "confidence":operating_confidence,"nms_iou":.5,"production_approved":False,"research_only":True,"human_privacy_reviewed":False,
        "project_acceptance_eligible":False,"source_licenses":source_licenses,"independent_route_verified":False,
        "data_revoked":False,"needs_revalidation":False,"best_epoch":best_epoch,"heldout_map50":heldout["metrics"]["map50"]}
    atomic_json(output/"model.json",manifest);link_model(output/"provenance.json",output/"model.json")
    atomic_json(output/"summary.json",{"architecture":"YOLOX-Nano","genuine_gradient_training":True,"best_epoch":best_epoch,
        "training_images":len(train_rows),"validation_images":len(val_rows),"test_images":len(test_rows),"validation_map50":best,
        "heldout_metrics":heldout["metrics"],"operating_metrics":heldout["operating_metrics"],"confidence_calibration":calibration,
        "elapsed_seconds":time.perf_counter()-started,"research_only":True,"acceptance_approved":False,"environment":environment,
        "shared_gpu_training":True,"training_time_is_not_isolated_inference_benchmark":True})
    print(json.dumps({"status":"trained_research_model","manifest":str(output/"model.json"),"heldout_metrics":heldout["metrics"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("release");parser.add_argument("output");parser.add_argument("--task",choices=["pothole","redaction"],required=True)
    parser.add_argument("--epochs",type=int,default=20);parser.add_argument("--batch-size",type=int,default=16);parser.add_argument("--learning-rate",type=float,default=.0005)
    parser.add_argument("--input-size",type=int,default=416);parser.add_argument("--max-labels",type=int,default=120)
    parser.add_argument("--known-class-evaluation",action="store_true")
    args=parser.parse_args();train(args.release,args.output,args.task,epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,
                                 input_size=args.input_size,max_labels=args.max_labels,known_class_evaluation=args.known_class_evaluation)
