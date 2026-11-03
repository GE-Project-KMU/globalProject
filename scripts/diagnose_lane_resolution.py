"""Training/validation-only nearest-label roundtrip, not a model score or IoU bound."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json


def diagnose(manifest,output):
    cv2.setNumThreads(1)
    source=workspace_path(manifest,exists=True);rows=json.loads(source.read_text(encoding="utf-8"))
    if any(r["split"] not in ("train","validation") for r in rows):raise ValueError("development_only_required")
    totals={s:{"intersection":0,"union":0,"positive_images":0,"paint_disappeared_images":0} for s in (512,1024)}
    for r in rows:
        path=workspace_path(r["mask_path"],exists=True)
        if sha256(path)!=r["mask_sha256"]:raise ValueError("mask_changed")
        truth=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE);h,w=truth.shape;valid=truth!=255
        for size,c in totals.items():
            scale=min(size/w,size/h)
            low=cv2.resize(truth,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_NEAREST)
            recovered=cv2.resize((low==1).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
            c["intersection"]+=int(((truth==1)&(recovered==1)&valid).sum())
            c["union"]+=int((((truth==1)|(recovered==1))&valid).sum())
            c["positive_images"]+=int((truth==1).any())
            c["paint_disappeared_images"]+=int((truth==1).any() and not (low==1).any())
    for c in totals.values():c["nearest_label_roundtrip_iou"]=c["intersection"]/c["union"] if c["union"] else None
    report={"scope":"development_label_aliasing_diagnostic","model_accuracy_measurement":False,"mathematical_upper_bound":False,
            "manifest_sha256":sha256(source),"samples":len(rows),"test_images_used":0,"sizes":totals,
            "method":"Original publisher GT -> nearest low-resolution GT -> nearest original-resolution paint; ignore255 excluded"}
    atomic_json(workspace_path(output),report);return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("manifest");p.add_argument("output");a=p.parse_args()
    print(json.dumps(diagnose(a.manifest,a.output),ensure_ascii=False,indent=2))
