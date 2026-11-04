from scripts.analyze_pothole_validation_errors import decompose

def test_disjoint_miss_localization_and_duplicate_prediction_are_distinguished():
    truth=[{'bbox_xywh_px':[0,0,10,10]},{'bbox_xywh_px':[30,30,10,10]},{'bbox_xywh_px':[60,60,10,10]}]
    pred=[{'bbox_xywh_px':[0,0,10,10],'confidence':.9},{'bbox_xywh_px':[0,0,10,10],'confidence':.8},{'bbox_xywh_px':[30,30,20,20],'confidence':.7}]
    stats=decompose(truth,pred)
    assert stats['tp']==1 and stats['fn']==2 and stats['fp']==2
    assert stats['fn_best_prediction_IoU']=={'0_1_to_0_5':1,'below_0_1':1}
    assert stats['fp_best_GT_IoU']=={'at_least_0_5':1,'0_1_to_0_5':1}

def test_publisher_negative_image_predictions_are_label_relative_background_FP():
    assert decompose([],[{'bbox_xywh_px':[0,0,10,10],'confidence':.9}])=={'tp':0,'fn':0,'fp':1,'fn_best_prediction_IoU':{},'fp_best_GT_IoU':{'below_0_1':1}}
