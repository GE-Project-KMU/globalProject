import numpy as np
import pytest
from scripts.redaction_rescue_data import shared_view,tile_view,tile_targets,check_conservation,source_boxes
from scripts.train_redaction_rescue_v8 import paired_tile_order,large_recall,calibrate,family_winner,train,gpu_preflight

def test_all_four_boundary_intersections_preserve_positive_fragments_and_total_area():
    image=np.zeros((200,200,3),np.uint8);original=[[90,90,20,20],[0,0,100,100],[150,150,10,10],[190,190,10,10]]
    canvas,boxes,_=shared_view(image,original)
    assert check_conservation(boxes)=={'original_GT':4,'visible_GT_per_tile':[2,1,1,3],'all_original_GT_covered':True,'clipped_area_conserved':True}
    for tile in range(4):
        pixels,labels,parents=tile_view(canvas,boxes,tile)
        assert pixels.shape==(3,960,960) and pixels.dtype==np.float32 and pixels.flags.c_contiguous
        assert 0 in parents
        first=labels[parents.index(0)];assert first[3]==first[4]==96
    assert original[0]==[90,90,20,20]

def test_round_resize_nominal_GT_scale_and_top_left_padding_match_application():
    image=np.full((359,641,3),[7,11,13],np.uint8);canvas,boxes,geometry=shared_view(image,[[640,358,1,1]])
    assert geometry['resized_width']==1920 and geometry['resized_height']==round(359*1920/641)==1075
    assert np.array_equal(canvas[0,0],[7,11,13]) and np.array_equal(canvas[1075,0],[114,114,114])
    assert boxes[0,0]==pytest.approx(640*1920/641) and boxes[0,3]==pytest.approx(1920/641)
    assert check_conservation(boxes)['all_original_GT_covered']

def test_original_flip_moves_GT_before_shared_letterbox_and_keeps_all_targets():
    image=np.zeros((100,200,3),np.uint8);image[:,0]=[1,2,3]
    canvas,boxes,_=shared_view(image,[[10,20,20,30]],flip=True)
    assert boxes[0,0]==pytest.approx(170*9.6)
    assert len(tile_targets(boxes,0)[1])==0 and len(tile_targets(boxes,1)[1])==1
    assert np.array_equal(canvas[959,1919],[1,2,3])

def test_512_original_GT_survive_and_513_are_rejected_instead_of_truncated():
    boxes=[[0,0,1,1]]*512;canvas,scaled,_=shared_view(np.zeros((100,100,3),np.uint8),boxes)
    _,packed,parents=tile_view(canvas,scaled,0)
    assert len(parents)==512 and int((packed.sum(-1)>0).sum())==512
    with pytest.raises(ValueError,match='over512'):shared_view(np.zeros((100,100,3),np.uint8),boxes+boxes[:1])

@pytest.mark.parametrize('boxes',[[[0,0,-1,1]],[[float('nan'),0,1,1]],[[99,0,2,1]]])
def test_invalid_original_GT_not_silently_clipped_into_new_labels(boxes):
    with pytest.raises(ValueError):source_boxes(boxes,100,100)

def test_paired_sampler_visits_every_tile_once_with_distinct_originals_per_batch():
    order=paired_tile_order(10,2026,0)
    assert len(order)==40 and sorted(order)==list(range(40))
    assert all(order[i]//4!=order[i+1]//4 for i in range(0,len(order),2))
    assert order==paired_tile_order(10,2026,0) and order!=paired_tile_order(10,2026,1)
    with pytest.raises(ValueError):paired_tile_order(9,2026,0)

def fixture_predictions(face_conf=.9,extra_FP=0):
    truth=[{'sample_id':'faceVAL','class':'face','bbox_xywh_px':[0,0,100,100]},
           {'sample_id':'plateVAL','class':'license_plate','bbox_xywh_px':[0,0,100,50]}]
    pred=[dict(t,confidence=face_conf if t['class']=='face' else .9) for t in truth]
    pred.extend({'sample_id':'faceVAL','class':'face','bbox_xywh_px':[200+i*20,200,10,10],'confidence':.9} for i in range(extra_FP))
    return {'truth':truth,'predictions':pred}

def test_large_face_gate_is_fixed_conf01_not_AP_floor001():
    result=fixture_predictions(face_conf=.05)
    assert large_recall(result)==0 and calibrate(result)['selected']['min_recall']==1

def test_complete_family_required_and_incomplete_P2_is_diagnostic_only():
    base={'name':'legacy_v4','training_complete':True,'validation':fixture_predictions(extra_FP=1)}
    partial={'name':'p2_incomplete','training_complete':False,'validation':fixture_predictions()}
    assert family_winner([base,partial],1.)=='legacy_v4'
    with pytest.raises(ValueError,match='baseline'):family_winner([partial],1.)

def test_family_precision_precedes_recall_when_both95_feasible_and_baseline_wins_exact_tie():
    base={'name':'legacy_v4','training_complete':True,'validation':fixture_predictions()}
    rescue={'name':'rescue_v8','training_complete':True,'validation':fixture_predictions(extra_FP=1)}
    assert family_winner([rescue,base],1.)=='legacy_v4'
    rescue['validation']=fixture_predictions()
    assert family_winner([rescue,base],1.)=='legacy_v4'

def test_GPU_paths_reject_before_Torch_or_plan_access_without_root_authorization():
    with pytest.raises(ValueError,match='authorization'):train(False)
    with pytest.raises(ValueError,match='authorization'):gpu_preflight(False)
