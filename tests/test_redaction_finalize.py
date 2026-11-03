"""Publication and parity evidence gates; synthetic metadata only, no photographs."""
from copy import deepcopy
import pytest
from scripts.improve_redaction_finalize import strict_report,assert_public,count_metrics,aggregate_comparison

ITEM={'model_manifest_sha256':'a'*64,'onnx_sha256':'b'*64}
CHECKPOINT='c'*64
CASE={'image_sha256':'d'*64,'gt_count':1,'predicted_known_class_count':1}
def valid_report():
    value={'manifest_sha256':ITEM['model_manifest_sha256'],'weight_sha256':ITEM['onnx_sha256'],'checkpoint_sha256':CHECKPOINT,'image_sha256':CASE['image_sha256'],'synthetic_input':False,'adapter_result_count':1}
    for key in ('numerical_parity_passed','postprocess_parity_passed','artifact_contract_verified','preprocess_contract_verified','finite_output','torch_checkpoint_compared'):value[key]=True
    for key in ('app_vs_ort','app_vs_torch','ort_vs_torch','app_repeatability'):value[key]={'passed':True,'rtol':.001,'atol':.0001}
    return value

def test_complete_default_strict_real_case_passes():strict_report(valid_report(),ITEM,CHECKPOINT,CASE)

@pytest.mark.parametrize('key',['manifest_sha256','weight_sha256','checkpoint_sha256','image_sha256'])
def test_other_model_checkpoint_or_other_class_image_rejected(key):
    report=valid_report();report[key]='e'*64
    with pytest.raises(ValueError,match='binding'):strict_report(report,ITEM,CHECKPOINT,CASE)

@pytest.mark.parametrize('key',['numerical_parity_passed','postprocess_parity_passed','artifact_contract_verified','preprocess_contract_verified','finite_output','torch_checkpoint_compared'])
def test_raw_postprocessing_and_real_checkpoint_all_required(key):
    report=valid_report();report[key]=False
    with pytest.raises(ValueError):strict_report(report,ITEM,CHECKPOINT,CASE)

def test_synthetic_case_cannot_substitute_for_positive_face_or_plate():
    report=valid_report();report['synthetic_input']=True
    with pytest.raises(ValueError):strict_report(report,ITEM,CHECKPOINT,CASE)

@pytest.mark.parametrize('mode',['app_vs_ort','app_vs_torch','ort_vs_torch','app_repeatability'])
@pytest.mark.parametrize('field,value',[('passed',False),('rtol',.01),('atol',.001)])
def test_no_pair_or_repeatability_tolerance_relaxation(mode,field,value):
    report=valid_report();report[mode][field]=value
    with pytest.raises(ValueError,match='strict'):strict_report(report,ITEM,CHECKPOINT,CASE)

@pytest.mark.parametrize('field',['gt_count','predicted_known_class_count'])
def test_zero_known_class_support_not_a_coverage_case(field):
    case=dict(CASE);case[field]=0
    with pytest.raises(ValueError,match='positive'):strict_report(valid_report(),ITEM,CHECKPOINT,case)

@pytest.mark.parametrize('key',['sample_id','source_id','image_id','bbox_xywh_px','path','image_path','mask_path','source_path','filename','file_name','predictions','truth','pixels','weights','weights_path','manifest','benchmark_manifest','source_release_path'])
def test_individual_identifier_or_picture_path_blocked_recursively(key):
    with pytest.raises(ValueError,match='private'):assert_public({'results':[{'nested':{key:'synthetic-private-value'}}]})

@pytest.mark.parametrize('path',['C:/Users/user/proj/globalProject/data/private.jpg','data/training/private.json','.cache/private.json','photo.jpeg','model.onnx','checkpoint.pt'])
def test_path_in_an_innocent_key_is_still_private(path):
    with pytest.raises(ValueError,match='private'):assert_public({'description':path})

def test_aggregate_metrics_and_artifact_hashes_publishable():
    assert_public({'weights_sha256':'a'*64,'classes':{'face':{'gt_count':1140,'recall':.9,'precision':.4},'license_plate':{'gt_count':200,'recall':.99,'precision':.99}},'production_approved':False,'source_licenses':['CC-BY-NC-ND-4.0','MIT']})

@pytest.mark.parametrize('key',['image_sha256','imageSHA256','source_hash','source_member','pixel_coordinates','paint_mask_rle','sampleHash','image_bytes','image_base64','mask_bytes','sourceImageSHA256'])
def test_per_image_hashes_members_and_pixel_payload_forbidden(key):
    with pytest.raises(ValueError,match='private'):assert_public({'harmless':[{'nested':{key:'a'*64}}]})

@pytest.mark.parametrize('extension',['.npy','.npz','.pth','.ckpt','.safetensors','.bmp','.webp','.tif','.tiff','.gif','.zip'])
def test_binary_tensor_mask_photo_and_weight_suffixes_blocked(extension):
    with pytest.raises(ValueError,match='private'):assert_public({'note':'private'+extension})

def balanced():return {'gt_count':10,'prediction_count':10,'tp':9,'fp':1,'fn':1,'recall':.9,'precision':.9,'ap50':.8}

@pytest.mark.parametrize('field,value',[('gt_count',11),('prediction_count',11),('tp',11),('fp',-1),('fn',False),('recall',.95),('precision',float('nan')),('ap50',1.1)])
def test_publisher_refuses_unbalanced_or_invented_counts(field,value):
    value_dict=balanced();value_dict[field]=value
    with pytest.raises(ValueError):count_metrics(value_dict)

def comparison():
    groups={key:{'classes':{label:balanced() for label in ('face','license_plate')}} for key in ('ap_metrics','operating_metrics','fixed_035_reference')}
    return {'scope':'fixed-role','samples':400,'benchmark_sha256':'z'*64,'plan_sha256':'p'*64,'test_used_for_selection':False,'project_acceptance_passed':False,'known_class_conditional':True,'results':{'baseline':{'manifest_sha256':'a'*64,**deepcopy(groups)},'candidate':{'manifest_sha256':'b'*64,**deepcopy(groups)}}}

def aggregate(report):return aggregate_comparison(report,{},'p'*64,{'samples':400,'benchmark_sha256':'z'*64,'role':'fixed-role'},'a'*64,'b'*64)

def test_frozen_same_data_pair_accepted():assert aggregate(comparison())['candidate_both_class_recall95_goal_met'] is False

@pytest.mark.parametrize('field,value',[('samples',399),('benchmark_sha256','changed'),('plan_sha256','changed'),('scope','other-test'),('test_used_for_selection',True),('project_acceptance_passed',True),('known_class_conditional',False)])
def test_publisher_binds_benchmark_plan_count_and_nonacceptance(field,value):
    report=comparison();report[field]=value
    with pytest.raises(ValueError):aggregate(report)

def test_candidate_manifest_cannot_be_reselected():
    report=comparison();report['results']['candidate']['manifest_sha256']='changed'
    with pytest.raises(ValueError,match='model_binding'):aggregate(report)

def test_same_class_support_must_match_between_models():
    report=comparison()
    for group in ('ap_metrics','operating_metrics','fixed_035_reference'):
        report['results']['candidate'][group]['classes']['face'].update(gt_count=20,fn=11,recall=.45)
    with pytest.raises(ValueError,match='support_mismatch'):aggregate(report)

def test_both_classes_required_in_every_metric_group():
    report=comparison();del report['results']['candidate']['operating_metrics']['classes']['face']
    with pytest.raises(ValueError,match='both_class'):aggregate(report)

@pytest.mark.parametrize('text',['see data/training/private.jpg here','local D:/secret.png is private','embedded data:image/png;base64,AAAA'])
def test_private_paths_or_image_data_embedded_in_prose_blocked(text):
    with pytest.raises(ValueError,match='private'):assert_public(text)

def test_fixed_case_coverage_counts_duplicate_predictions_once(tmp_path,monkeypatch):
    import scripts.improve_redaction_finalize as helper
    private=tmp_path/'validation-details.json';private.write_text('{}',encoding='utf-8')
    truth=[{'sample_id':'fixed-face','class':'face','bbox_xywh_px':[0,0,10,10]},{'sample_id':'fixed-face','class':'face','bbox_xywh_px':[20,20,10,10]},{'sample_id':'fixed-plate','class':'license_plate','bbox_xywh_px':[0,0,20,10]}]
    predictions=[{'sample_id':'fixed-face','class':'face','bbox_xywh_px':[0,0,10,10],'confidence':.9},{'sample_id':'fixed-face','class':'face','bbox_xywh_px':[0,0,10,10],'confidence':.8},{'sample_id':'fixed-face','class':'face','bbox_xywh_px':[90,90,10,10],'confidence':.7},{'sample_id':'fixed-plate','class':'license_plate','bbox_xywh_px':[0,0,20,10],'confidence':.9}]
    cases={'cases':{'face':{'sample_id':'fixed-face','gt_count':2,'predicted_known_class_count':3},'license_plate':{'sample_id':'fixed-plate','gt_count':1,'predicted_known_class_count':1}}}
    freeze={'selected_option':'tiled','options':{'tiled':{'confidence':.01}},'validation_details_sha256':helper.sha256(private)}
    monkeypatch.setattr(helper,'OPTIONS',tmp_path);monkeypatch.setattr(helper,'frozen',lambda:(freeze,None));monkeypatch.setattr(helper,'gate_reports',lambda *_:({},cases));monkeypatch.setattr(helper,'read',lambda _: {'truth':truth,'predictions':{'tiled':predictions}})
    counts=helper.case_coverage()
    assert counts=={'face':{'gt_count':2,'pred_count':3,'matched_count':1},'license_plate':{'gt_count':1,'pred_count':1,'matched_count':1}}
    assert cases['cases']['face']['sample_id']=='fixed-face'
