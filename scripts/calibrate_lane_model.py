"""Choose a paint threshold on validation only, before any final test evaluation."""
import argparse,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release
from agent.train_lane import prepare_pair
from agent.models import OnnxModel
from scripts.evaluate_lane_model import confusion,metrics

def calibrate(manifest_path,model_path,report_path):
    manifest_path,model_path=workspace_path(manifest_path,exists=True),workspace_path(model_path,exists=True)
    rows=json.loads(manifest_path.read_text(encoding="utf-8"));state=validate_research_release(rows)
    if not state["valid"]:raise ValueError("invalid_research_release")
    artifact=json.loads(model_path.read_text(encoding="utf-8"))
    if not artifact.get("research_only") or artifact["dataset_release"]!="manifest-sha256:"+sha256(manifest_path):raise ValueError("research_provenance_mismatch")
    if artifact.get("threshold_calibration"):raise ValueError("threshold_already_frozen_create_new_model_version")
    weights=workspace_path(artifact["weights"],exists=True)
    if sha256(weights)!=artifact["sha256"]:raise ValueError("weights_hash_mismatch")
    model=OnnxModel(model_path);net=model.net
    thresholds=[.1,.2,.3,.4,.5,.6,.7,.8,.85,.9,.925,.95,.975,.99]
    counters={t:{"tp":0,"fp":0,"fn":0,"tn":0,"ignored":0} for t in thresholds}
    validation=sorted([r for r in rows if r["split"]=="validation"],key=lambda r:r["source_id"])
    if not validation:raise ValueError("validation_required")
    for row in validation:
        pixels,_=prepare_pair(row["path"],row["mask_path"])
        truth=cv2.imread(str(workspace_path(row["mask_path"])),cv2.IMREAD_GRAYSCALE)
        h,w=truth.shape;scale=min(512/w,512/h);rh,rw=max(1,round(h*scale)),max(1,round(w*scale))
        net.setInput(pixels[None]);logits=net.forward();delta=(logits[0,1]-logits[0,0])[:rh,:rw]
        for threshold in thresholds:
            predicted=cv2.resize((delta>math.log(threshold/(1-threshold))).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
            c=confusion(truth,predicted)
            for k,v in c.items():counters[threshold][k]+=v
    scores=[{"paint_threshold":t,**metrics(counters[t])} for t in thresholds]
    best=max(scores,key=lambda r:(r["paint_iou"] if r["paint_iou"] is not None else -1,-abs(r["paint_threshold"]-.5)))
    report={"selected_on":"validation_only","inference_backend":getattr(model,"backend","unknown"),"calibrated_at":datetime.now(timezone.utc).isoformat(),"validation_samples":len(validation),"model_sha256":artifact["sha256"],"manifest_sha256":sha256(manifest_path),"scores":scores,"selected":best,"test_images_used":0,"project_acceptance_passed":False}
    atomic_json(report_path,report)
    artifact.update(paint_threshold=best["paint_threshold"],threshold_calibration={"selected_on":"validation","report_sha256":sha256(workspace_path(report_path)),"validation_samples":len(validation),"test_images_used":0})
    atomic_json(model_path,artifact)
    return best
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("manifest");p.add_argument("model");p.add_argument("report");a=p.parse_args();print(json.dumps(calibrate(a.manifest,a.model,a.report),indent=2))
