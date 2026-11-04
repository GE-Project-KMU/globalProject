import copy,json
import pytest
from scripts import evaluate_privacy_family as tool

def row(index,label='face'):
    return {'sample_id':label+'-'+str(index),'source_id':str(index).zfill(5),'evaluation_class':label,
            'objects':[{'class':label,'bbox_xywh_px':[10,10,100,100]}],
            'path':'synthetic-not-read.jpg','sha256':'a'*64,'expires_at':'2099-01-01T00:00:00+00:00'}

def result(rows):
    truth=[dict(o,sample_id=r['sample_id']) for r in rows for o in r['objects']]
    return {'truth':truth,'predictions':[dict(o,confidence=.9) for o in truth],'images':len(rows)}

def test_complete_knownclass_VAL_binding_accepts_exact_source():
    rows=[row(0),row(0,'license_plate')];tool.assert_validation_binding(result(rows),rows)

@pytest.mark.parametrize('change',['missing_truth','extra_TEST_prediction','unknown_otherclass','NaN_score','boolean_score','bad_box','partial_image_count'])
def test_partial_stale_or_test_prediction_cannot_enter_family_selection(change):
    rows=[row(0),row(0,'license_plate')];value=result(rows)
    if change=='missing_truth':value['truth'].pop()
    elif change=='extra_TEST_prediction':value['predictions'].append(dict(value['predictions'][0],sample_id='TEST'))
    elif change=='unknown_otherclass':value['predictions'][0]['class']='license_plate'
    elif change=='NaN_score':value['predictions'][0]['confidence']=float('nan')
    elif change=='boolean_score':value['predictions'][0]['confidence']=True
    elif change=='bad_box':value['predictions'][0]['bbox_xywh_px']=[0,0,-1,1]
    else:value['images']=1
    with pytest.raises(ValueError):tool.assert_validation_binding(value,rows)

def components():
    rows=[row(i) for i in range(400)]+[row(i,'license_plate') for i in range(108)]
    def item(label):
        selected=[r for r in rows if r['evaluation_class']==label]
        return {'role':'baseline','manifest_sha256':tool.PIN['baseline_manifest'],'checkpoint_sha256':tool.PIN['baseline_checkpoint'],
                'development_sha256':tool.PIN['development'],'backend':'native-gpu','runtime_profile':tool.runtime_profile('native-gpu'),
                'case_sample_ids':sorted(r['sample_id'] for r in selected),'result':result(selected)}
    return rows,item('face'),item('license_plate')

def test_only_bound_immutable_face400_and_plate108_can_form_complete508():
    rows,face,plate=components();combined=tool.compose_baseline(face,plate,rows)
    tool.assert_validation_binding(combined,rows);assert combined['images']==508

@pytest.mark.parametrize('change',['manifest','checkpoint','case_membership','runtime','missing_receipt'])
def test_mutable_best400_or_mixed_runtime_components_are_rejected(change):
    rows,face,plate=components()
    if change=='manifest':face['manifest_sha256']='0'*64
    elif change=='checkpoint':face['checkpoint_sha256']='0'*64
    elif change=='case_membership':face['case_sample_ids'][0]='test'
    elif change=='runtime':plate['runtime_profile']['torch']='other'
    else:face.pop('manifest_sha256')
    with pytest.raises(ValueError):tool.compose_baseline(face,plate,rows)

def test_positive_canonical_case_requires_correct_sameclass_prediction_from_both_models():
    rows=[row(0),row(1),row(0,'license_plate')];base=result(rows);candidate=result(rows)
    candidate['predictions']=[p for p in candidate['predictions'] if p['sample_id']!='face-0']
    cases=tool.canonical_cases(rows,base,candidate,.1,.1)
    assert cases[0]['source_id']=='00001' and cases[1]['label']=='license_plate'
    candidate['predictions']=[p for p in candidate['predictions'] if p['class']!='license_plate']
    with pytest.raises(ValueError,match='common_positive'):tool.canonical_cases(rows,base,candidate,.1,.1)

def test_no_pixel_or_TEST_action_without_explicit_operator_authorization(monkeypatch):
    monkeypatch.setattr(tool,'protocol',lambda:pytest.fail('Must reject before protocol or pixels'))
    with pytest.raises(ValueError,match='authorization'):tool.evaluate_VAL('baseline',False)
    with pytest.raises(ValueError,match='authorization'):tool.run_comparison('fresh',False)

def test_failed_numeric_gate_prevents_any_TEST_plan_read_or_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(tool,'DEST',tmp_path);(tmp_path/'protocol.json').write_text('{}',encoding='utf-8')
    value={'protocol_sha256':tool.sha256(tmp_path/'protocol.json')};(tmp_path/'winner-freeze.json').write_text(json.dumps(value),encoding='utf-8')
    monkeypatch.setattr(tool,'protocol',lambda:{})
    def reject(_):raise ValueError('strict_face_or_plate_failed')
    monkeypatch.setattr(tool,'numeric_gates',reject)
    with pytest.raises(ValueError,match='strict_face'):tool.run_comparison('fresh',True)
    assert not (tmp_path/'fresh-attempt.json').exists()

def test_comparison_plan_uses_exact_frozen_model_artifacts_and_declared_benchmark(tmp_path,monkeypatch):
    monkeypatch.setattr(tool,'DEST',tmp_path);(tmp_path/'winner-freeze.json').write_text('{}',encoding='utf-8')
    frozen={'models':{'baseline':{'manifest':'data/synthetic/base.json','manifest_sha256':'a'*64},
                       'winner':{'manifest':'data/synthetic/winner.json','dataset_release_sha256':'b'*64}}}
    protocol={'benchmarks':{'fresh':{'sha256':'c'*64,'samples':200,'role':'fresh-only'},'regression':{'sha256':'d'*64,'samples':1400,'role':'regression-only'}}}
    plan=tool.comparison_plan(protocol,frozen,'fresh')
    assert plan['models']['candidate']=='data/synthetic/winner.json' and plan['expected_samples']==200
    assert plan['raw_rtol']==.001 and plan['raw_atol']==.0001 and plan['test_used_for_fitting_or_selection'] is False

def numeric_fixture(tmp_path,monkeypatch):
    from scripts import evaluate_public_refinement as strict
    monkeypatch.setattr(tool,'DEST',tmp_path);calls=[]
    # Isolate this wrapper test from the independently tested numeric verifier;
    # no fake successful numeric report or model is registered/published.
    monkeypatch.setattr(strict,'strict_proof',lambda manifest,proof,**kwargs:calls.append((proof.name,kwargs)))
    def write(path,value):path.write_text(json.dumps(value),encoding='utf-8');return tool.sha256(path)
    cases=[{'label':'face','sha256':'a'*64},{'label':'license_plate','sha256':'b'*64}]
    frozen={'positive_VAL_cases_sha256':write(tmp_path/'positive-VAL-cases.json',cases),'validation_predictions_sha256':{},'models':{}}
    for role in ['baseline','rescue_v8']:
        frozen['validation_predictions_sha256'][role]=write(tmp_path/(role+'-VAL508.json'),{})
    for role in ['baseline','winner']:
        folder=tmp_path/role;folder.mkdir();manifest=folder/'model.json';provenance=folder/'provenance.json';checkpoint=folder/'checkpoint.bin'
        checkpoint.write_bytes(b'unit contract metadata only, not learned weights')
        frozen['models'][role]={'manifest':str(manifest),'manifest_sha256':write(manifest,{}),'provenance_sha256':write(provenance,{}),
            'checkpoint':str(checkpoint),'checkpoint_sha256':tool.sha256(checkpoint),'origin_manifest':str(manifest),
            'origin_manifest_sha256':tool.sha256(manifest),'origin_provenance':str(provenance),'origin_provenance_sha256':tool.sha256(provenance)}
        for label,name in [('face','verification-final.json'),('license_plate','verification-plate.json')]:
            write(folder/name,{'app_backend':'onnxruntime-cpu','app_backend_version':'1.31.1',
                'detection_postprocess_parity':{ref:{'application_class_counts':{label:1}} for ref in ['ort','torch']}})
    return frozen,calls

def test_both_models_require_both_real_class_strict_verifier_calls(tmp_path,monkeypatch):
    frozen,calls=numeric_fixture(tmp_path,monkeypatch);tool.numeric_gates(frozen)
    assert [name for name,_ in calls]==['verification-final.json','verification-plate.json']*2
    assert all(kwargs['positive'] is True and kwargs['checkpoint_sha'] for _,kwargs in calls)

@pytest.mark.parametrize('failure',['missing_plate','plate_class_empty','wrong_CPU_version'])
def test_plate_or_runtime_gate_cannot_be_hidden_by_passing_face(tmp_path,monkeypatch,failure):
    frozen,_=numeric_fixture(tmp_path,monkeypatch);path=tmp_path/'winner/verification-plate.json'
    if failure=='missing_plate':path.unlink()
    else:
        value=json.loads(path.read_text(encoding='utf-8'))
        if failure=='plate_class_empty':value['detection_postprocess_parity']['torch']['application_class_counts']={}
        else:value['app_backend_version']='1.20.1'
        path.write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises((ValueError,FileNotFoundError)):tool.numeric_gates(frozen)
