import numpy as np
import pytest
from scripts import privacy_efficiency_study as study


def row(x,y,w=30,h=30,score=.05,label=0):
    return [x,y,w,h,1.,score if label==0 else 0.,score if label==1 else 0.]


def test_tile_selection_count_score_ties_and_boundary():
    rows=np.array([row(1,1),row(1000,1),row(1,1000),row(1000,1000)],np.float32)
    assert study.select_tiles(rows,'small_evidence_top2')==[0,1]
    assert study.select_tiles(rows,'density_top2')==[0,1]
    assert study.select_tiles(np.array([row(960,960)],np.float32),'density_top2')==[3]
    assert study.select_tiles(np.array([row(1920,20)],np.float32),'density_top2')==[]


def test_small_criterion_does_not_use_plate_or_large_global_faces():
    rows=np.array([row(1,1,100,100),row(1000,1,score=.9,label=1)],np.float32)
    assert study.select_tiles(rows,'small_evidence_top2')==[]
    assert study.select_tiles(rows,'density_top2')==[0]
    assert study.select_tiles(np.empty((0,7),np.float32),'density_top2')==[]
    with pytest.raises(ValueError):study.select_tiles(rows,'unspecified')


def test_consensus_preserves_global_plate_and_strong_tiles_filters_isolated_weak():
    parts=[np.array([row(20,20)],np.float32),
           np.array([row(20,20),row(200,200),row(300,300,score=.11)],np.float32),
           np.array([row(400,400)],np.float32),np.array([row(400,400)],np.float32),
           np.empty((0,7),np.float32),np.array([row(700,700,label=1)],np.float32)]
    result=study.consensus(parts)
    assert len(result)==7-1
    assert 200 not in result[:,0]
    assert 300 in result[:,0] and sum(result[:,0]==400)==2
    assert result[-1,6]>0


def test_consensus_does_not_accept_same_branch_duplicate_as_support():
    empty=np.empty((0,7),np.float32)
    parts=[empty,np.array([row(20,20),row(20,20)],np.float32),empty,empty,empty,empty]
    assert len(study.consensus(parts))==0


def test_postprocess_keeps_cross_class_boxes_and_clips_only_after_nms():
    raw=np.array([row(5,5,30,30,.9),row(5,5,30,30,.8),row(5,5,30,30,.7,1)],np.float32)
    result=study.postprocess(raw,1,(100,100,3))
    assert len(result)==2 and {r['class'] for r in result}=={'face','license_plate'}
    assert all(r['bbox_xywh_px']==[0.,0.,20.,20.] for r in result)


def test_study_postprocess_matches_unchanged_real_adapter_on_fixed_synthetic_raw():
    from agent.models import OnnxModel
    rng=np.random.default_rng(1900)
    raw=np.zeros((100,7),np.float32)
    raw[:,:2]=rng.uniform(-50,1970,(100,2));raw[:,2:4]=rng.uniform(-5,300,(100,2))
    raw[:,4:]=rng.uniform(0,1,(100,3))
    raw[0]=row(0,0,30,30,.01)
    class FixedNetwork:
        def setInput(self,value):assert value.shape==(1,3,1920,1920)
        def forward(self):return raw[None]
    model=OnnxModel.__new__(OnnxModel)
    model.manifest={'task':'redaction','input_size':[1920,1920],
        'classes':['face','license_plate'],'output_layout':'yolox_decoded_xywh','confidence':.01,'nms_iou':.5}
    model.net=FixedNetwork()
    frame=np.zeros((384,640,3),np.uint8)
    assert model.predict(frame)==study.postprocess(raw,3.,frame.shape)
