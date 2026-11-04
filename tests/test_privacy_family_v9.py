"""V9 selection and closure contracts; synthetic metadata only, no learned weights."""
import hashlib,json
from pathlib import Path
import pytest
from scripts import evaluate_privacy_family_v9 as f
from scripts import compare_privacy_family_v9_parallel as parallel

def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf-8');return f.sha256(p)

@pytest.fixture
def context(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'OLD_DEST',tmp_path/'old');monkeypatch.setattr(f,'DEST',tmp_path/'new')
    stop=tmp_path/'stop.json';pin=put(stop,{'status':'stopped_before_V8_evaluation_for_preTEST_V9_revision','root_authorized':True,'exact_identity_absent':True,'own_children_before_stop':0,'phase_count':0,'actual_VAL508_evaluations':0,'winner_freezes':0,'actual_TEST_predictions':0,'models_weights_and_learning_sources_changed':False})
    monkeypatch.setattr(f,'STOP',stop);monkeypatch.setattr(f,'STOP_SHA',pin)
    return tmp_path

@pytest.mark.parametrize('name',['fresh-attempt.json','regression-attempt.json','fresh-comparison.json','regression-comparison.json','parallel-fresh-v8','parallel-regression-v8'])
def test_original_attempt_blocks_revision(context,name):
    put(f.OLD_DEST/name,{})
    with pytest.raises(ValueError,match='prior_V8_TEST'):f.no_prior_TEST()

@pytest.mark.parametrize('name',['fresh-attempt.json','regression-attempt.json','fresh-comparison.json','parallel-fresh-v9'])
def test_new_attempt_closes_selection_only(context,name):
    put(f.DEST/name,{})
    f.no_prior_TEST() # Existing active V9 worker must still recheck protocol.
    with pytest.raises(ValueError,match='selection_closed'):f.before_selection()


def test_changed_stop_receipt_blocks_before_images(context):
    f.STOP.write_text('{}')
    with pytest.raises(ValueError,match='stop_receipt_changed'):f.no_prior_TEST()

@pytest.mark.parametrize('key,value',[('phase_count',1),('actual_TEST_predictions',1),('exact_identity_absent',False),('root_authorized',False)])
def test_stop_receipt_requires_zero_prior_work(context,monkeypatch,key,value):
    record=f.read(f.STOP);record[key]=value;monkeypatch.setattr(f,'STOP_SHA',put(f.STOP,record))
    with pytest.raises(ValueError,match='pre_V8_phase'):f.no_prior_TEST()


def candidate(name,minrec=.9,precision=.5,large=1.,feasible=False):
    return {'name':name,'training_complete':True,'validation':{'min_recall':minrec,'macro_precision':precision,'large':large,'both_recall95':feasible}}

@pytest.fixture
def scoring(monkeypatch):
    monkeypatch.setattr(f,'calibrate',lambda x:{'selected':x})
    monkeypatch.setattr(f,'large_recall',lambda x:x['large'])


def test_exact_ties_have_predeclared_family_order(scoring):
    rows=[candidate(n) for n in reversed(f.FAMILIES)]
    assert f.family_winner(rows,1.)=='legacy_v4'
    rows[2]['validation']['macro_precision']=.4
    assert f.family_winner(rows,1.)=='rescue_v8'


def test_feasible_precision_primary_and_largeface_gate(scoring):
    rows=[candidate('legacy_v4',.96,.5,feasible=True),candidate('rescue_v8',.97,.6,feasible=True),candidate('fixed_final12',.99,.9,large=.98,feasible=True)]
    assert f.family_winner(rows,1.)=='rescue_v8'


def test_no_feasible_prioritizes_min_recall(scoring):
    rows=[candidate('legacy_v4',.8,.99),candidate('rescue_v8',.91,.2),candidate('fixed_final12',.9,.9)]
    assert f.family_winner(rows,1.)=='rescue_v8'

@pytest.mark.parametrize('mutation',['missing','duplicate','incomplete'])
def test_all_three_complete_families_required(scoring,mutation):
    rows=[candidate(n) for n in f.FAMILIES]
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows[2]=rows[1]
    else:rows[2]['training_complete']=False
    with pytest.raises(ValueError,match='three_complete'):f.family_winner(rows,1.)


def test_parallel_uses_new_namespace_and_old_unchanged_math():
    assert parallel.family is f
    assert parallel.PROTOCOL.name=='PRIVACY-FAMILY-V9-PARALLEL-EXECUTION-PROTOCOL.json'
    assert 'scripts/evaluate_privacy_family_v9.py' in parallel.SOURCES
    assert f.PIN['fresh']=='8230874168c8b2d4d8de09c5c11cb3197f641c0ef6a28999ef77e3ea2465c9a7'
    assert f.PIN['regression']=='03dbb7a8e60b0316e4c06508bcb76175097f594a29d0a25d40a2129998ae4c8a'


def test_no_model_or_image_read_without_evaluation_authorization(monkeypatch):
    monkeypatch.setattr(f,'protocol',lambda:pytest.fail('should reject before data'))
    with pytest.raises(ValueError,match='authorization'):f.evaluate_VAL('fixed_final12',False)


@pytest.fixture
def final_export(tmp_path,monkeypatch):
    folder=tmp_path/'final';folder.mkdir();rescue=tmp_path/'rescue';rescue.mkdir()
    monkeypatch.setattr(f,'FINAL12',folder);monkeypatch.setattr(f,'RESCUE',rescue)
    monkeypatch.setattr(f,'completed_rescue',lambda:{})
    plan=tmp_path/'docs/reports/PRIVACY-RESCUE-V8-FINAL12-EXPORT-PLAN.json';monkeypatch.setattr(f,'ROOT',tmp_path)
    put(rescue/'plan.json',{'approved':True});put(plan,{'export_source_sha256':'a'*64})
    (folder/'best.pt').write_bytes(b'SYNTHETIC checkpoint');(folder/'model.onnx').write_bytes(b'SYNTHETIC graph')
    ck=f.sha256(folder/'best.pt');onnx=f.sha256(folder/'model.onnx')
    put(folder/'model.json',{'sha256':onnx,'trained_checkpoint_sha256':ck,'training_complete':True})
    freeze={'manifest_sha256':f.sha256(folder/'model.json'),'checkpoint_sha256':ck,'onnx_sha256':onnx,'export_plan_sha256':f.sha256(plan),'export_source_sha256':'a'*64,'epoch':12,'fixed_final12_EMA':True,'TEST_images':0,'training_complete':True,'confidence':.01}
    digest=put(folder/'candidate-freeze.json',freeze)
    receipt=dict(freeze,status='completed',EMA=True,source_plan_sha256=f.sha256(rescue/'plan.json'),candidate_freeze_sha256=digest,legacy_raw_prefix_matches_frozen_v4=True)
    put(folder/'export-receipt.json',receipt)
    return folder,receipt


def test_exact_fixed_final12_export_is_admitted(final_export):
    assert f.completed_final12()['training_complete'] is True

@pytest.mark.parametrize('key,value',[('epoch',11),('EMA',False),('training_complete',False),('TEST_images',1),('source_plan_sha256','0'*64),('export_plan_sha256','0'*64),('checkpoint_sha256','0'*64),('candidate_freeze_sha256','0'*64),('legacy_raw_prefix_matches_frozen_v4',False)])
def test_fixed_final12_incomplete_or_tampered_receipt_rejected(final_export,key,value):
    folder,receipt=final_export;receipt[key]=value;put(folder/'export-receipt.json',receipt)
    with pytest.raises(ValueError):f.completed_final12()


def test_candidate_change_cannot_pass_preTEST_numeric_gate(monkeypatch):
    monkeypatch.setattr(f,'no_prior_TEST',lambda:None)
    monkeypatch.setattr(f,'candidate_artifacts',lambda:{'fixed_final12':{'checkpoint_sha256':'new'}})
    with pytest.raises(ValueError,match='candidate_changed'):f.numeric_gates({'candidate_artifacts':{}})



def test_prepare_only_reads_metadata_and_refuses_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'DEST',tmp_path/'private');monkeypatch.setattr(f,'PUBLIC',tmp_path/'public.json')
    monkeypatch.setattr(f,'before_selection',lambda:None);monkeypatch.setattr(f,'immutable_parents',lambda:None)
    monkeypatch.setattr(f,'validation',lambda:pytest.fail('preparation must not read VAL rows/images'))
    monkeypatch.setattr(f,'read_image',lambda *a:pytest.fail('no pixels during metadata prepare'))
    monkeypatch.setattr(f,'read',lambda p:{'export_source_sha256':'a'*64})
    # Real workspace temp outputs, synthetic metadata source SHA only.
    monkeypatch.setattr(f,'sha256',lambda p:'a'*64)
    value=f.prepare()
    assert value['families']==list(f.FAMILIES) and value['benchmarks']['regression']['samples']==1400
    assert value['family_aliases']=={'rescue_v8_AP_selected':'rescue_v8'}
    assert value['extra_gradient_updates']==0
    with pytest.raises(ValueError,match='already_exists'):f.prepare()



def test_retention_adds_only_two_explicit_optional_roots():
    from scripts import freeze_cycle2_retention as retention
    assert len(retention.POLICIES)==29
    for name in ('data/training/redaction-rescue-v8-final12','data/training/privacy-family-v9-evaluation'):
        assert name in retention.OPTIONAL_ROOTS
        assert retention.POLICIES[name][2]==retention.PRIVACY_CAP

@pytest.mark.parametrize('field',['image_sha256','sample_id','bbox_xywh_px','image_path'])
def test_V9_control_with_private_fields_is_not_exempted(tmp_path,field):
    from scripts import freeze_cycle2_retention as retention
    p=tmp_path/'freeze.json';put(p,{'format':'privacy-family-v9-winner-freeze',field:'SYNTHETIC private reference'})
    assert retention.minimal_control_provenance(p) is False
    put(p,{'format':'privacy-family-v9-winner-freeze','models':{'winner':{'manifest':'data/models/model.json','checkpoint':'data/models/best.pt'}}})
    assert retention.minimal_control_provenance(p) is True
