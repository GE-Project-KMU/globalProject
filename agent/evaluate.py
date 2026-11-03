"""Independent metrics from supplied predictions and locked human ground truth."""
import argparse
import json
import sys
import numpy as np
from .pipeline import intersection_over_union
from .paths import workspace_path,atomic_json,sha256


def detection_metrics(truth,predictions,classes):
    scores={}
    for label in classes:
        gt=[r for r in truth if r["class"]==label]
        predictions_for_class=sorted([r for r in predictions if r["class"]==label],key=lambda r:r["confidence"],reverse=True)
        matched=set();hits=[]
        for prediction in predictions_for_class:
            options=[(intersection_over_union(prediction["bbox_xywh_px"],g["bbox_xywh_px"]),i) for i,g in enumerate(gt)
                     if i not in matched and g["sample_id"]==prediction["sample_id"]]
            overlap,index=max(options,default=(0,None))
            hit=overlap>=.5
            if hit:matched.add(index)
            hits.append(int(hit))
        tp=np.cumsum(hits);fp=np.cumsum(1-np.asarray(hits))
        recall=tp/len(gt) if gt else np.zeros(len(hits))
        precision=tp/np.maximum(tp+fp,1)
        ap=float(np.mean([max(precision[recall>=r],default=0.) for r in np.linspace(0,1,101)])) if gt else None
        scores[label]={"gt_count":len(gt),"prediction_count":len(hits),"tp":len(matched),"fp":len(hits)-len(matched),
                       "fn":len(gt)-len(matched),"ap50":ap,"recall":len(matched)/len(gt) if gt else None}
    aps=[r["ap50"] for r in scores.values() if r["ap50"] is not None]
    return {"classes":scores,"map50":sum(aps)/len(aps) if aps else None,"matching":"one_to_one_IoU0.5_AP101"}


def lane_metrics(truth,predictions):
    by_id={r["segment_id"]:r for r in predictions}
    if len(by_id)!=len(predictions):raise ValueError("duplicate_predicted_segment")
    truth_ids={r["segment_id"] for r in truth}
    if len(truth_ids)!=len(truth):raise ValueError("duplicate_truth_segment")
    tp=fp=fn=assessable=0;errors=[]
    for gt in truth:
        prediction=by_id.get(gt["segment_id"])
        if not 0<=gt["retention_pct"]<=100:raise ValueError("invalid_truth_retention")
        positive=gt["retention_pct"]<60
        value=prediction.get("retention_pct") if prediction else None
        if value is None:
            fn+=int(positive);continue
        if not 0<=value<=100:raise ValueError("invalid_retention")
        assessable+=1;errors.append(abs(value-gt["retention_pct"]))
        tp+=int(positive and value<60);fp+=int(not positive and value<60);fn+=int(positive and value>=60)
    extras=[r for r in predictions if r["segment_id"] not in truth_ids]
    fp+=sum(r.get("retention_pct") is not None and r["retention_pct"]<60 for r in extras)
    return {"tp":tp,"fp":fp,"fn":fn,"unmatched_predictions":len(extras),"recall":tp/(tp+fn) if tp+fn else None,
            "precision":tp/(tp+fp) if tp+fp else None,"retention_mae_pp":float(np.mean(errors)) if errors else None,
            "assessable_ratio":assessable/len(truth) if truth else None,"unassessable_positive_counts_as_fn":True}


def segmentation_iou(truth,predicted):
    truth,predicted=np.asarray(truth),np.asarray(predicted)
    if truth.shape!=predicted.shape or truth.ndim!=2 or not np.isin(truth,[0,1,255]).all() or not np.isin(predicted,[0,1]).all():
        raise ValueError("segmentation_shape_or_values_invalid")
    valid=truth!=255
    intersection=int(((truth==1)&(predicted==1)&valid).sum());union=int((((truth==1)|(predicted==1))&valid).sum())
    return {"intersection":intersection,"union":union,"paint_iou":intersection/union if union else None}


def evaluate(request_path,output):
    request=json.loads(workspace_path(request_path,exists=True).read_text(encoding="utf-8"))
    if request.get("independent_locked_truth") is not True:raise ValueError("locked_independent_ground_truth_required")
    if not any(k in request for k in ("detection","lane","segmentation")):raise ValueError("metrics_input_required")
    result={"request_sha256":sha256(workspace_path(request_path,exists=True)),"synthetic":request.get("synthetic",False),"acceptance_approved":False}
    if "detection" in request:
        d=request["detection"];result["detection"]=detection_metrics(d["truth"],d["predictions"],d["classes"])
    if "lane" in request:
        d=request["lane"];result["lane"]=lane_metrics(d["truth"],d["predictions"])
    if "segmentation" in request:
        import cv2
        intersection=union=0
        for row in request["segmentation"]:
            gt=cv2.imread(str(workspace_path(row["truth"],exists=True)),cv2.IMREAD_GRAYSCALE)
            pred=cv2.imread(str(workspace_path(row["predicted"],exists=True)),cv2.IMREAD_GRAYSCALE)
            metric=segmentation_iou(gt,pred);intersection+=metric["intersection"];union+=metric["union"]
        result["segmentation"]={"paint_iou":intersection/union if union else None,"intersection":intersection,"union":union}
    atomic_json(output,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("request");parser.add_argument("output");args=parser.parse_args()
    try:print(json.dumps(evaluate(args.request,args.output),ensure_ascii=False,indent=2));return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:print(json.dumps({"status":"evaluation_blocked","reason":str(exc)}));return 2
if __name__=="__main__":sys.exit(main())
