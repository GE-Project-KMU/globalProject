"""CPU mathematical/geometry contracts, never evidence of detector accuracy."""
from datetime import datetime,timezone
from pathlib import Path
import sys
from unittest.mock import patch
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from scripts.train_yolox_research import detection_metrics,operating_metrics,calibrate,Samples,evaluate_model
from agent.paths import atomic_json,sha256


def main():
    passed=[]
    truth=[{"sample_id":"fixture-a","class":"face","bbox_xywh_px":[5.,2.,10.,8.]}]
    good=truth[0]|{"confidence":.9}
    perfect=detection_metrics(truth,[good],["face"])
    assert perfect["map50"]==1 and perfect["classes"]["face"]["tp"]==1
    passed.append("perfect one-to-one AP50=1")
    duplicate=operating_metrics({"truth":truth,"predictions":[good,good|{"confidence":.8}]},["face"],.5)
    assert duplicate["classes"]["face"]["tp"]==1 and duplicate["classes"]["face"]["fp"]==1 and duplicate["classes"]["face"]["precision"]==.5
    passed.append("duplicate counts FP; AP interpolation may remain1 after earlier full recall")
    missed=detection_metrics(truth,[],["face"])
    assert missed["map50"]==0 and missed["classes"]["face"]["fn"]==1
    passed.append("missed GT AP50=0 and FN=1")
    empty=detection_metrics([],[],["face"])
    false_positive=detection_metrics([],[good],["face"])
    assert empty["map50"] is None and false_positive["map50"] is None and false_positive["classes"]["face"]["fp"]==1
    passed.append("no-GT AP undefined, empty or false-positive counts retained")
    mixed_truth=truth+[truth[0]|{"sample_id":"fixture-b","class":"license_plate"}]
    calibration=calibrate({"truth":mixed_truth,"predictions":[good,mixed_truth[1]|{"confidence":.3},
        good|{"confidence":.1,"bbox_xywh_px":[25.,2.,10.,8.]}]},["face","license_plate"])
    assert calibration["selected_confidence"]==.2 and calibration["both_class_goal_achieved"]
    passed.append("validation-only fixed-grid confidence .2 excludes FP and preserves both recall")
    folder=ROOT/".cache/research-contract"/str(uuid.uuid4());folder.mkdir(parents=True)
    image=folder/"synthetic.png";pixels=np.full((20,40,3),50,np.uint8);pixels[2:10,5:15]=200;cv2.imwrite(str(image),pixels)
    row={"sample_id":"fixture-a","path":str(image),"objects":[{"class":"face","bbox_xywh_px":[5.,2.,10.,8.]}]}
    samples=Samples([row],["face"],size=640,training=True,max_labels=1)
    with patch("scripts.train_yolox_research.random.random",side_effect=[0.,1.]):
        prepared,labels,_=samples[0]
    assert prepared.shape==(3,640,640) and prepared.dtype==torch.float32
    assert np.allclose(labels[0].numpy(),[0.,480.,96.,160.,128.])
    assert float(prepared[0,500,10])==114
    passed.append("BGR float255 top-left letterbox and horizontal flip bbox center/extent")
    over=row|{"objects":row["objects"]*2}
    try:
        with patch("scripts.train_yolox_research.random.random",return_value=1.):Samples([over],["face"],training=True,max_labels=1)[0]
        raise AssertionError("excess training labels were silently truncated")
    except ValueError as exc:assert "reject_entire_image" in str(exc)
    passed.append("training target overflow rejects whole image")
    plate=row|{"objects":[{"class":"license_plate","bbox_xywh_px":[5.,2.,10.,8.]}],"evaluation_class":"license_plate","ignore_bboxes_xywh_px":[]}
    class CPUModel:
        def __init__(self,label):self.label=label
        def eval(self):return self
        def __call__(self,images):
            # 416/40 letterbox scale10.4. Correct original bbox center=(10,6).
            scores=[.9,.1] if self.label=="face" else [.1,.9]
            return torch.tensor([[[104.,62.4,104.,83.2,.9,*scores],[0.,0.,1.,1.,0.,.1,.1]]]*len(images),dtype=torch.float32)
    # The production evaluator intentionally transfers to CUDA. For this CPU
    # contract only, bypass that transfer while retaining native CPU YOLOX NMS.
    with patch.object(torch.Tensor,"cuda",lambda value,*args,**kwargs:value):
        known=evaluate_model(CPUModel("face"),[plate],["face","license_plate"],fullscene=True)
        projected=evaluate_model(CPUModel("license_plate"),[plate],["face","license_plate"],fullscene=True)
    assert known["predictions"]==[] and known["metrics"]["classes"]["license_plate"]["fn"]==1 and known["metrics"]["classes"]["face"]["fp"]==0
    assert np.allclose(projected["predictions"][0]["bbox_xywh_px"],[5.,2.,10.,8.],atol=1e-4)
    assert projected["metrics"]["classes"]["license_plate"]["tp"]==1
    passed.append("known-class scope excludes unknown-class FP without hiding known missed GT")
    passed.append("native postprocess coordinate inverse maps letterbox box to original pixels")
    report={"status":"passed","device":"CPU; cuda input transfer stubbed only for isolated contract","synthetic_fixtures":True,
        "performance_evidence":False,"tests":passed,"count":len(passed),"at":datetime.now(timezone.utc).isoformat(),
        "training_script_sha256":sha256(ROOT/"scripts/train_yolox_research.py"),"torch":torch.__version__}
    path=ROOT/"docs/reports/DETECTION-TRAINING-CONTRACT.json";atomic_json(path,report)
    print(f"CPU detector contract passed: {len(passed)} checks; synthetic, not model accuracy")


if __name__=="__main__":main()
