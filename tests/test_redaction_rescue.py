import numpy as np
import pytest
from agent import redaction_rescue_network as rescue
from scripts.verify_trained_models import postprocess_detection

def test_lazy_factory_constants_match_frozen_prefix_and_full_ordered_grid():
    assert rescue.LEGACY_PROPOSALS==5*(120**2+60**2+30**2)+(80**2+40**2+20**2)
    assert rescue.RESCUE_PROPOSALS==4*240**2
    assert rescue.LEGACY_PROPOSALS+rescue.RESCUE_PROPOSALS==333300
    assert rescue.RESCUE_SCORE_SCALE==.5 and rescue.RESCUE_INITIAL_PROBABILITY==1e-4 and rescue.MAX_LABELS==512

def test_inactive_rescue_preserves_positive_classwise_NMS_overlap_and_boundary_clip():
    baseline=np.zeros((1,rescue.LEGACY_PROPOSALS,7),np.float32)
    baseline[0,0]=[200,200,40,40,.9,.9,0]
    baseline[0,1]=[202,202,40,40,.8,.9,0]
    baseline[0,2]=[500,500,80,40,.9,0,.9]
    baseline[0,3]=[0,0,20,20,.9,.9,0]
    extra=np.zeros((1,rescue.RESCUE_PROPOSALS,7),np.float32)
    extra[:,:,:4]=[200,200,40,40];extra[:,:,4]=rescue.RESCUE_SCORE_SCALE*rescue.RESCUE_INITIAL_PROBABILITY
    extra[:,:,5]=rescue.RESCUE_INITIAL_PROBABILITY
    candidate=np.concatenate((baseline,extra),1)
    manifest={'task':'redaction','classes':['face','license_plate'],'input_size':[1920,1920],
              'output_layout':'yolox_decoded_xywh','confidence':.01,'nms_iou':.5}
    image=np.zeros((64,64,3),np.uint8)
    before=postprocess_detection(manifest,baseline,image);after=postprocess_detection(manifest,candidate,image)
    assert before==after and len(before)==3
    assert sorted(p['class'] for p in before)==['face','face','license_plate']
    assert any(p['bbox_xywh_px'][0]==p['bbox_xywh_px'][1]==0 for p in before)

def test_missing_or_extra_legacy_state_refused_before_transplant():
    class Bundle:
        def state_dict(self):return {'face.weight':0,'plate.weight':0,'rescue_head.weight':0}
    for state in [{'face.weight':0},{'face.weight':0,'plate.weight':0,'unknown':0}]:
        with pytest.raises(ValueError,match='complete_legacy'):rescue.load_frozen_legacy(Bundle(),state)
