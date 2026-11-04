"""V9 selection and closure contracts; synthetic metadata only, no learned weights."""
import hashlib,json
from pathlib import Path
import pytest
from scripts import evaluate_privacy_family_v10 as f
from scripts import compare_privacy_family_v10_parallel as parallel

def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf-8');return f.sha256(p)

@pytest.fixture
def context(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'revision_lineage',lambda metadata_only=False:{})
    monkeypatch.setattr(f,'OLD_V9_DEST',tmp_path/'oldv9')
    monkeypatch.setattr(f,'OLD_DEST',tmp_path/'old');monkeypatch.setattr(f,'DEST',tmp_path/'new')
    stop=tmp_path/'stop.json';pin=put(stop,{'status':'stopped_before_V8_evaluation_for_preTEST_V9_revision','root_authorized':True,'exact_identity_absent':True,'own_children_before_stop':0,'phase_count':0,'actual_VAL508_evaluations':0,'winner_freezes':0,'actual_TEST_predictions':0,'models_weights_and_learning_sources_changed':False})
    monkeypatch.setattr(f,'STOP',stop);monkeypatch.setattr(f,'STOP_SHA',pin)
    return tmp_path

@pytest.mark.parametrize('name',['fresh-attempt.json','regression-attempt.json','fresh-comparison.json','regression-comparison.json','parallel-fresh-v8','parallel-regression-v8'])
def test_original_attempt_blocks_revision(context,name):
    put(f.OLD_DEST/name,{})
    with pytest.raises(ValueError,match='prior_V8_TEST'):f.no_prior_TEST()

@pytest.mark.parametrize('name',['fresh-attempt.json','regression-attempt.json','fresh-comparison.json','parallel-fresh-v10'])
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
    assert parallel.PROTOCOL.name=='PRIVACY-FAMILY-V10-PARALLEL-EXECUTION-PROTOCOL.json'
    assert 'scripts/evaluate_privacy_family_v10.py' in parallel.SOURCES
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
    plan=tmp_path/'docs/reports/PRIVACY-RESCUE-V8-FINAL12-RETRY1-EXPORT-PLAN.json';monkeypatch.setattr(f,'ROOT',tmp_path)
    put(rescue/'plan.json',{'approved':True})
    audit=tmp_path/'membership.json';audit_sha=put(audit,{'passed':True,'selected_training_and_validation_rows':3508,'parent_assets':3508,'unused_plate_TRAIN_rows_excluded':1993,'complete_identity_set_sha256':'b'*64,'development_sha256':f.PIN['development']})
    put(plan,{'export_source_sha256':'a'*64,'original_trainer_subset':f.TRAINER_SUBSET,'provenance_membership_audit':{'path':str(audit),'sha256':audit_sha}})
    (folder/'best.pt').write_bytes(b'SYNTHETIC checkpoint');(folder/'model.onnx').write_bytes(b'SYNTHETIC graph')
    ck=f.sha256(folder/'best.pt');onnx=f.sha256(folder/'model.onnx')
    put(folder/'model.json',{'sha256':onnx,'trained_checkpoint_sha256':ck,'training_complete':True})
    freeze={'manifest_sha256':f.sha256(folder/'model.json'),'checkpoint_sha256':ck,'onnx_sha256':onnx,'export_plan_sha256':f.sha256(plan),'export_source_sha256':'a'*64,'epoch':12,'fixed_final12_EMA':True,'TEST_images':0,'training_complete':True,'confidence':.01}
    digest=put(folder/'candidate-freeze.json',freeze)
    receipt=dict(freeze,status='completed',EMA=True,source_plan_sha256=f.sha256(rescue/'plan.json'),candidate_freeze_sha256=digest,legacy_raw_prefix_matches_frozen_v4=True)
    receipt.update(original_trainer_subset=f.TRAINER_SUBSET,same_original_final12_EMA=True,parent_export_policy_sha256='bca93d5d7163f4746c436ba2da4c277fe8b0356ac7cdfc8d0e4b91d857ad3daf',membership_audit_sha256=audit_sha,provenance_membership_sha256='b'*64)
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
    monkeypatch.setattr(f,'revision_lineage',lambda metadata_only=False:{})
    monkeypatch.setattr(f,'before_selection',lambda:None);monkeypatch.setattr(f,'immutable_parents',lambda:None)
    monkeypatch.setattr(f,'validation',lambda:pytest.fail('preparation must not read VAL rows/images'))
    monkeypatch.setattr(f,'read_image',lambda *a:pytest.fail('no pixels during metadata prepare'))
    monkeypatch.setattr(f,'read',lambda p:{'export_source_sha256':'a'*64,'original_trainer_subset':f.TRAINER_SUBSET})
    # Real workspace temp outputs, synthetic metadata source SHA only.
    monkeypatch.setattr(f,'sha256',lambda p:'a'*64)
    value=f.prepare()
    assert value['families']==list(f.FAMILIES) and value['benchmarks']['regression']['samples']==1400
    assert value['family_aliases']=={'rescue_v8_AP_selected':'rescue_v8'}
    assert value['extra_gradient_updates']==0
    with pytest.raises(ValueError,match='already_exists'):f.prepare()



def test_retention_has_exact31_and_two_explicit_new_optional_roots():
    from scripts import freeze_cycle2_retention as retention
    assert len(retention.POLICIES)==31
    for name in ('data/training/redaction-rescue-v8-final12-retry1','data/training/privacy-family-v10-evaluation'):
        assert name in retention.OPTIONAL_ROOTS
        assert retention.POLICIES[name][2]==retention.PRIVACY_CAP

@pytest.mark.parametrize('field',['image_sha256','sample_id','bbox_xywh_px','image_path'])
def test_V9_control_with_private_fields_is_not_exempted(tmp_path,field):
    from scripts import freeze_cycle2_retention as retention
    p=tmp_path/'freeze.json';put(p,{'format':'privacy-family-v10-winner-freeze',field:'SYNTHETIC private reference'})
    assert retention.minimal_control_provenance(p) is False
    put(p,{'format':'privacy-family-v10-winner-freeze','models':{'winner':{'manifest':'data/models/model.json','checkpoint':'data/models/best.pt'}}})
    assert retention.minimal_control_provenance(p) is True


@pytest.fixture
def lineage(tmp_path,monkeypatch):
    paths=[]
    for index in range(9):
        path=tmp_path/('input'+str(index)+('.pt' if index==0 else '.json'));path.write_bytes(b'SYNTHETIC')
        paths.append(path)
    failed=tmp_path/'failed.json';failure_sha=put(failed,{'status':'failed','sources_unchanged':True,'own_child_running_after_cleanup':False,'parent_artifacts_preserved_verified':False})
    stopped=tmp_path/'stopped.json';stop_sha=put(stopped,{'status':'stopped_naturally_for_failed_ROOT_fixed12_export','root_authorized':True,'exact_identity_absent':True,'phase_count':0,'owned_evaluation_children_created':0,'owned_CPU_export_operations':0,'actual_native_VAL508_evaluations':0,'winner_freezes':0,'actual_TEST_predictions':0})
    audit=tmp_path/'audit.json';audit_sha=put(audit,{'failed_execution_sha256':failure_sha,'all_original_inputs_unchanged':True,'original_input_artifacts_verified':9,'input_sha256':{str(p):f.sha256(p) for p in paths}})
    for name,path,digest in [('FAILED_EXPORT',failed,failure_sha),('STOP_V9',stopped,stop_sha),('PRESERVATION',audit,audit_sha)]:
        monkeypatch.setattr(f,name,path);monkeypatch.setattr(f,name+'_SHA',digest)
    return paths


def test_failed_wrapper_false_is_preserved_and_separate_audit_supplies_evidence(lineage):
    result=f.revision_lineage()
    assert result['old_failed_wrapper_parent_verification_was_not_reached'] is True
    assert f.read(f.FAILED_EXPORT)['parent_artifacts_preserved_verified'] is False


def test_metadata_lineage_does_not_read_weight_bytes(lineage,monkeypatch):
    real=f.sha256
    monkeypatch.setattr(f,'sha256',lambda path:pytest.fail('weights read during metadata prepare') if Path(path).suffix=='.pt' else real(path))
    f.revision_lineage(metadata_only=True)


def test_changed_parent_rejects_actual_admission(lineage):
    lineage[0].write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='parent_changed'):f.revision_lineage()


def test_V9_attempt_blocks_even_with_valid_revision_evidence(context,monkeypatch):
    put(f.OLD_V9_DEST/'fresh-attempt.json',{'status':'failed'})
    with pytest.raises(ValueError,match='prior_V9_TEST'):f.no_prior_TEST()


def test_original_trainer_predicate_is_exact_faceTRAIN3000_plus_ALLVAL508(tmp_path,monkeypatch):
    from scripts import redaction_rescue_data as data
    rows=[]
    for split,label,count in [('train','face',3000),('train','license_plate',1993),('validation','face',400),('validation','license_plate',108)]:
        for i in range(count):
            rows.append({'sample_id':split+'-'+label+'-'+str(i),'split':split,'evaluation_class':label,'known_label_coverage':[label],'human_privacy_reviewed':False,'not_for_deployment':True,'expires_at':'2099-01-01T00:00:00+00:00','image':{'width':64,'height':64},'objects':[{'class':label,'bbox_xywh_px':[0,0,16,16]}]})
    path=tmp_path/'synthetic-dev.json';put(path,rows)
    monkeypatch.setattr(data,'DEVELOPMENT',path)
    monkeypatch.setattr(data,'read_image',lambda *a:pytest.fail('metadata predicate must not decode pixels'))
    train,val,_=data.load_development()
    assert len(rows)==5501 and len(train)==3000 and len(val)==508
    assert all(r['evaluation_class']=='face' for r in train)
    assert sum(r['evaluation_class']=='face' for r in val)==400
    assert sum(r['evaluation_class']=='license_plate' for r in val)==108
    assert len(train+val)==3508


@pytest.mark.parametrize('subset',[{'train_face':3000,'validation_face':400,'validation_plate':108,'total':5501},{'train_face':3000,'validation_face':400,'validation_plate':107,'total':3507}])
def test_no_broad_or_incomplete_provenance_subset(final_export,subset):
    folder,receipt=final_export;receipt['original_trainer_subset']=subset;put(folder/'export-receipt.json',receipt)
    with pytest.raises(ValueError,match='exact_original_training_subset'):f.completed_final12()


@pytest.mark.parametrize('name',['export_redaction_final12_retry1.py','finalize_privacy_family_v10_results.py','privacy_family_v10_authorized_queue.py'])
def test_V10_known_writers_are_detected(tmp_path,name):
    from scripts import cycle2_research_writer_guard as guard
    from types import SimpleNamespace
    p=SimpleNamespace(info={'pid':123,'ppid':0,'create_time':456.,'cmdline':[str(tmp_path/name)],'cwd':str(tmp_path)})
    assert guard.inspect_writers(tmp_path,[p],999)['known_writers'][0]['scripts']==[name]

@pytest.mark.parametrize('phase',['baseline-VAL508','rescue-VAL508-import','fixed12-VAL508','winner-freeze','strict-baseline-face','strict-baseline-license_plate','strict-winner-face','strict-winner-license_plate','fresh','regression'])
def test_V10_exact_phase_logs_and_no_ROOT_CPU_export_phase(phase):
    from scripts import freeze_cycle2_retention as retention
    assert retention.valid_log_source('.cache/privacy-family-v10-operator/'+phase+'.log')
    assert not retention.valid_log_source('.cache/privacy-family-v10-operator/CPU-fixed12-export.log')
    assert not retention.valid_log_source('.cache/privacy-family-v10-operator/'+phase+'-receipt.json')
