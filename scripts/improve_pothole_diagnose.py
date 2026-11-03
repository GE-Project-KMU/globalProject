"""Read-only prior-run diagnosis; publisher GT is never changed from predictions."""
from pathlib import Path
import json,sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from agent.paths import atomic_json,workspace_path

def iou(a,b):
    x,y,w,h=a;xx,yy,ww,hh=b
    area=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
    return area/(w*h+ww*hh-area) if w*h+ww*hh>area else 0.

def diagnose():
    base=ROOT/"data/training/pothole-nano-rdd2020-v1"
    release=json.loads((ROOT/"data/training/research-pothole-rdd2020/release.json").read_text())
    predictions=json.loads((base/"selected-validation-predictions.json").read_text())
    history=json.loads((base/"training-history.json").read_text())["history"]
    sizes={};by_id={r["sample_id"]:r for r in release};distribution={}
    for row in release:
        key=row["split"]+":"+row["country"];distribution.setdefault(key,{"images":0,"positive_images":0,"objects":0})
        distribution[key]["images"]+=1;distribution[key]["positive_images"]+=bool(row["objects"]);distribution[key]["objects"]+=len(row["objects"])
        for obj in row["objects"]:
            area=obj["bbox_xywh_px"][2]*obj["bbox_xywh_px"][3]
            sizes.setdefault(row["split"],[]).append(area**.5)
    gt=predictions["truth"];pred=sorted(predictions["predictions"],key=lambda x:x["confidence"],reverse=True)
    matches=set();false=[]
    for p in pred:
        best,index=max(((iou(p["bbox_xywh_px"],g["bbox_xywh_px"]),i) for i,g in enumerate(gt) if i not in matches and p["sample_id"]==g["sample_id"]),default=(0,None))
        if best>=.5:matches.add(index)
        else:false.append(p)
    misses=[g for i,g in enumerate(gt) if i not in matches]
    report={"baseline":"pothole-nano-rdd2020-v1","selection_evidence":"validation only; no GT relabeling from predictions",
        "counts":distribution,"bbox_sqrt_area_px_percentiles":{k:dict(zip(["p10","p50","p90"],np.percentile(v,[10,50,90]).tolist())) for k,v in sizes.items()},
        "first_epoch":history[0],"best_validation_epoch":max(history,key=lambda x:x["validation_map50"]),"last_epoch":history[-1],
        "validation_floor01":predictions["metrics"],"validation_FP_confidence_ge035":sum(p["confidence"]>=.35 for p in false),
        "unmatched_validation_gt_count_floor01":len(misses),
        "observations":["India-only train vs Japan-only validation; country domain shift confounded with model recipe",
            "training loss declines while validation AP peaks early: consistent with overfit, not causal proof",
            "old recipe omitted official native mosaic/HSV, SGD decay groups/warmup/EMA/noaug-L1",
            "publisher D40 original VOC verified, quarantined third-party labels never used"],
        "planned_candidate_count":1,"selection":"new grouped validation AP50; max validation F2 operating threshold",
        "old_test_role":"already examined regression benchmark only; never train","synthetic_accuracy_evidence":False}
    atomic_json(ROOT/"docs/reports/POTHOLE-IMPROVEMENT-DIAGNOSIS.json",report)
    output=ROOT/"data/training/pothole-improvement-diagnostics";output.mkdir(parents=True,exist_ok=True)
    panels=[]
    for label,objects in (("FP",false[:6]),("FN",misses[:6])):
        for obj in objects:
            row=by_id[obj["sample_id"]];image=cv2.imread(str(workspace_path(row["path"],exists=True)))
            for ground in row["objects"]:
                x,y,w,h=map(round,ground["bbox_xywh_px"]);cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            x,y,w,h=map(round,obj["bbox_xywh_px"]);cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
            image=cv2.resize(image,(384,288));cv2.putText(image,label+" green=GT red=error",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(image)
    while len(panels)<12:panels.append(np.zeros((288,384,3),np.uint8))
    cv2.imwrite(str(output/"validation-errors.jpg"),np.vstack([np.hstack(panels[i:i+3]) for i in range(0,12,3)]))
    atomic_json(output/"source.json",{"expires_at":min(r["expires_at"] for r in release),"private_diagnostic":True,
        "source_release":"data/training/research-pothole-rdd2020/release.json","human_privacy_reviewed":False})
    print(json.dumps(report),flush=True)

if __name__=="__main__":diagnose()
