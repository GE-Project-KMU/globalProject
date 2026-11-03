"""Validation-only diagnosis for the next private redaction research recipe."""
from collections import defaultdict
import json
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def iou(a,b):
    x,y,w,h=a;xx,yy,ww,hh=b
    intersection=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
    return intersection/(w*h+ww*hh-intersection) if intersection else 0.

def main():
    folder=ROOT/"data/training/redaction-nano-v3"
    release=ROOT/"data/training/research-redaction-fullframe/release.json"
    rows=json.loads(release.read_text());selected=folder/"selected-validation-predictions.json"
    predictions=json.loads(selected.read_text())["predictions"]
    indexed=defaultdict(list)
    for pred in predictions:
        if pred["class"]=="face" and pred["confidence"]>=.01:indexed[pred["sample_id"]].append(pred)
    buckets=defaultdict(lambda:{"gt":0,"matched":0})
    original_sizes=[]
    for row in rows:
        if row["split"]!="validation" or row["evaluation_class"]!="face":continue
        gt=[o for o in row["objects"] if o["class"]=="face"];matched=set()
        for pred in sorted(indexed[row["sample_id"]],key=lambda p:p["confidence"],reverse=True):
            score,index=max(((iou(pred["bbox_xywh_px"],obj["bbox_xywh_px"]),i) for i,obj in enumerate(gt) if i not in matched),default=(0,None))
            if score>=.5:matched.add(index)
        scale=min(640/row["image"]["width"],640/row["image"]["height"])
        for index,obj in enumerate(gt):
            size=min(obj["bbox_xywh_px"][2:])*scale
            bucket="under8px" if size<8 else "8to16px" if size<16 else "16to32px" if size<32 else "atleast32px"
            buckets[bucket]["gt"]+=1;buckets[bucket]["matched"]+=int(index in matched)
            original_sizes.append(size)
    for value in buckets.values():value["recall_at_frozen_confidence_001"]=value["matched"]/value["gt"]
    report={"scope":"Old validation-only diagnosis, no new holdout predictions or test-based selection",
            "release_sha256":sha(release),"selected_validation_predictions_sha256":sha(selected),
            "face_gt_by_min_dimension_at640":dict(buckets),
            "known_causes":["783 face training images vs complex small-object WIDER scenes","Single shared two-class objectness sees unannotated other-class objects as negatives","Generic COCO object initialization, simple flip/brightness only; no EMA or scale-aware sampling in v3","Face validation recall remains below goal even at frozen confidence .01; lowering confidence alone does not solve localization/recall"],
            "no_old_test_sources_added_to_training":True,"new_recipe_not_started":True,
            "no_independent_person_identity_or_privacy_acceptance_claim":True}
    (ROOT/"docs/reports/REDACTION-IMPROVEMENT-DIAGNOSIS.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
