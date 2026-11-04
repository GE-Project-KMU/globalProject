from scripts.analyze_pothole_validation import size_bin,match

def test_640_letterbox_size_boundaries_and_odd_aspect_ratio():
    assert size_bin([0,0,15,15],640,640)=='<16'
    assert size_bin([0,0,16,16],640,640)=='16_to_32'
    assert size_bin([0,0,64,64],1280,720)=='32_to_64'
    assert size_bin([0,0,128,128],1280,720)=='64_plus'
    assert size_bin([0,0,16,16],1280,641)=='<16'

def test_greedy_operating_point_match_counts_duplicate_predictions_as_false_positive():
    truth=[{'bbox_xywh_px':[0,0,10,10]},{'bbox_xywh_px':[20,20,10,10]}]
    predictions=[{'bbox_xywh_px':[0,0,10,10],'confidence':.9},{'bbox_xywh_px':[0,0,10,10],'confidence':.8},{'bbox_xywh_px':[50,50,10,10],'confidence':.1}]
    assert match(truth,predictions)==({0},2)

def test_matched_GT_size_does_not_change_matching_or_make_new_small_GT_false_negatives():
    truth=[{'bbox_xywh_px':[0,0,10,10]},{'bbox_xywh_px':[0,0,30,30]}]
    predictions=[{'bbox_xywh_px':[0,0,30,30],'confidence':.8},{'bbox_xywh_px':[0,0,10,10],'confidence':.5}]
    assert match(truth,predictions)==({0,1},0)
