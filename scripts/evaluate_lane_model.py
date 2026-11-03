"""Locked research holdout evaluation using the application's actual ONNX adapter."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys,time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.dataset import validate_research_release
from agent.models import OnnxModel
from agent.paths import workspace_path,sha256,atomic_json


def confusion(truth,predicted):
    if truth.shape!=predicted.shape:raise ValueError("mask_shape_mismatch")
    valid=truth!=255
    return {"tp":int(((truth==1)&(predicted==1)&valid).sum()),"fp":int(((truth==0)&(predicted==1)&valid).sum()),
            "fn":int(((truth==1)&(predicted==0)&valid).sum()),"tn":int(((truth==0)&(predicted==0)&valid).sum()),"ignored":int((~valid).sum())}


def metrics(c):
    tp,fp,fn=c["tp"],c["fp"],c["fn"]
    return {"paint_iou":tp/(tp+fp+fn) if tp+fp+fn else None,"paint_precision":tp/(tp+fp) if tp+fp else None,
            "paint_recall":tp/(tp+fn) if tp+fn else None,"paint_dice":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def evaluate(manifest_path,model_path,output,split="test"):
    manifest_path,model_path=workspace_path(manifest_path,exists=True),workspace_path(model_path,exists=True)
    rows=json.loads(manifest_path.read_text(encoding="utf-8"));validation=validate_research_release(rows)
    if not validation["valid"]:raise ValueError("invalid_research_release:"+json.dumps(validation["errors"][:3]))
    model=OnnxModel(model_path);artifact=model.manifest
    if artifact["task"]!="lane" or not artifact.get("research_only"):raise ValueError("research_lane_artifact_required")
    if artifact["dataset_release"]!="manifest-sha256:"+sha256(manifest_path):raise ValueError("training_manifest_mismatch")
    chosen=sorted([row for row in rows if row["split"]==split],key=lambda row:row["source_id"])
    if not chosen:raise ValueError("empty_holdout")
    totals=Counter();samples=[];timings=[];predictions=[]
    for index,row in enumerate(chosen):
        image=cv2.imread(str(workspace_path(row["path"],exists=True)))
        truth=cv2.imread(str(workspace_path(row["mask_path"],exists=True)),cv2.IMREAD_GRAYSCALE)
        if sha256(workspace_path(row["mask_path"]))!=row["mask_sha256"]:raise ValueError("holdout_mask_changed")
        started=time.perf_counter();predicted=model.predict(image);timings.append(time.perf_counter()-started)
        c=confusion(truth,predicted);totals.update(c)
        samples.append({"sample_id":row["sample_id"],"source_id":row["source_id"],"confusion":c,**metrics(c)})
        if index<8:predictions.append((image,truth,predicted))
    model_sha=artifact["sha256"]
    report={"evaluated_at":datetime.now(timezone.utc).isoformat(),"task":"lane","scope":"offline_public_data_research",
            "split":split,"sample_count":len(chosen),"foreground_samples":sum(s["confusion"]["tp"]+s["confusion"]["fn"]>0 for s in samples),
            "manifest_sha256":sha256(manifest_path),"model_sha256":model_sha,"model_manifest_sha256":sha256(model_path),
            "runtime":"OpenCV DNN CPU application adapter","runtime_version":cv2.__version__,"input_size":artifact["input_size"],
            "paint_threshold":artifact.get("paint_threshold",.5),"threshold_selection":artifact.get("threshold_calibration",{"selected_on":"fixed_default","test_images_used":0}),"confusion":dict(totals),**metrics(totals),
            "latency_seconds":{"median":float(np.median(timings)),"p95":float(np.percentile(timings,95)),"first":timings[0]},
            "target_paint_iou":.75,"target_passed_on_this_research_holdout":metrics(totals)["paint_iou"]>=.75 if metrics(totals)["paint_iou"] is not None else False,
            "independent_route_verified":all(row.get("independent_route_verified",False) for row in chosen),"project_acceptance_passed":False,
            "retention_mae_pp":None,"retention_mae_unavailable_reason":"No independent expected-outline/gap/valid O/G/V ground truth in this public dataset",
            "limitations":["RTK roadMarking is a public low-resolution proxy for paint, not ZED 2i worn-lane acceptance data",
                           "Blocked filename partitions and duplicate checks do not prove independent physical road objects/routes",
                           "Publisher labels plus machine QC; no fabricated independent human privacy review"],"samples":samples}
    output=workspace_path(output);atomic_json(output,report)
    # Fixed first eight holdouts, never curated for apparent quality. Local only.
    panels=[]
    for image,truth,predicted in predictions:
        overlay=image.copy();overlay[predicted==1]=(overlay[predicted==1]*.4+np.array([20,20,240])*.6).astype(np.uint8)
        ground=image.copy();ground[truth==1]=(ground[truth==1]*.4+np.array([20,220,20])*.6).astype(np.uint8)
        panels.append(np.hstack([cv2.resize(image,(264,216)),cv2.resize(ground,(264,216)),cv2.resize(overlay,(264,216))]))
    if panels:cv2.imwrite(str(output.with_suffix(".preview.jpg")),np.vstack(panels))
    return {k:v for k,v in report.items() if k!="samples"}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("manifest");p.add_argument("model");p.add_argument("output");p.add_argument("--split",choices=["validation","test"],default="test");a=p.parse_args()
    print(json.dumps(evaluate(a.manifest,a.model,a.output,a.split),ensure_ascii=False,indent=2))
