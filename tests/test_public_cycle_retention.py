"""Synthetic-only retention preparation; never scans real research assets."""
import hashlib
import json
from pathlib import Path
import pytest
from scripts import freeze_public_cycle_retention as f
from scripts import research_retention as r

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.fixture
def scope(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'STATE',tmp_path/'data/research-retention')
    state=tmp_path/'data/research-retention'
    plan=state/'old-plan.json';plan_sha=put(plan,{'schema_version':1,'entries':[]})
    inventory=state/'old-inventory.json'
    inventory_sha=put(inventory,{'entries':{},'control_plans':{'old':{'status':'registered','path':'data/research-retention/old-plan.json','sha256':plan_sha}}})
    monkeypatch.setattr(f,'PINS',{'old-plan.json':plan_sha,'old-inventory.json':inventory_sha})
    roots={'data/research/mwpd-v6':('mwpd_source',['data/research/mwpd-v6/source.json'],'2026-11-11'),
           'data/training/pothole-mwpd-v6':('pothole_mixed_model',['data/research/mwpd-v6/source.json','data/research/rdd2020/source.json'],'2026-11-10')}
    monkeypatch.setattr(f,'POLICIES',roots)
    put(tmp_path/'data/research/mwpd-v6/source.json',{'expires_at':'2026-11-11T10:00:00Z'})
    put(tmp_path/'data/research/rdd2020/source.json',{'expires_at':'2026-11-10T10:00:00Z'})
    source=tmp_path/'data/research/mwpd-v6/photo.jpg';source.write_bytes(b'SYNTHETIC-NOT-PHOTO')
    model=tmp_path/'data/training/pothole-mwpd-v6';model.mkdir(parents=True)
    (model/'best.pt').write_bytes(b'PRESERVED-SYNTHETIC-WEIGHTS')
    put(model/'per-image.json',{'sample_id':'synthetic','predictions':[]})
    put(model/'provenance.json',{'format':'training-provenance-v1','assets':[]})
    receipt=state/'writer-stop.json'
    receipt_sha=put(receipt,{'cycle':'public-refinement-v6','writers_stopped':True,'confirmed_at':'2026-11-04T00:00:00Z','roots':sorted(roots)})
    return tmp_path,receipt.relative_to(tmp_path).as_posix(),receipt_sha,source

def test_audit_never_walks_or_hashes_active_assets(scope,monkeypatch):
    root,_,_,_=scope
    def refuse(*args,**kwargs):raise AssertionError('active asset scan forbidden')
    monkeypatch.setattr(f,'list_files',refuse)
    _,result=f.audit_immutable()
    assert result['historical_inventories_verified']==1 and result['historical_plans_verified']==1
    assert not (root/'data/research-retention/public-cycle-inventory.json').exists()

def test_manual_freeze_inherits_earliest_proof_preserves_weights_and_history(scope):
    root,receipt,pin,source=scope
    before={name:(root/(f.STATE+name)).read_bytes() for name in f.PINS}
    value=f.freeze(receipt,pin)
    inventory=json.loads((root/'data/research-retention/public-cycle-inventory.json').read_text())
    assert value['registered_files']==2 and value['source_files_deleted']==0
    assert value['deadline_date_counts']=={'2026-11-11':1,'2026-11-10':1}
    assert source.exists() and (root/'data/training/pothole-mwpd-v6/best.pt').exists()
    assert all((root/(f.STATE+name)).read_bytes()==blob for name,blob in before.items())
    assert all(not item['path'].endswith(('best.pt','provenance.json')) for item in inventory['entries'].values())
    public=json.dumps(value)
    assert 'photo.jpg' not in public and 'synthetic' not in public and 'source_path' not in public
    with pytest.raises(ValueError,match='new_public_cycle_outputs'):f.freeze(receipt,pin)

def test_writer_stop_required_before_any_asset_scan(scope,monkeypatch):
    root,receipt,_,_=scope
    value=json.loads((root/receipt).read_text());value['writers_stopped']=False;pin=put(root/receipt,value)
    monkeypatch.setattr(f,'list_files',lambda *args:pytest.fail('scanned while writers active'))
    with pytest.raises(ValueError,match='writer_stop_handoff'):f.freeze(receipt,pin)

def test_changed_handoff_digest_refused_before_scan(scope,monkeypatch):
    _,receipt,_,_=scope
    monkeypatch.setattr(f,'list_files',lambda *args:pytest.fail('scanned without pinned handoff'))
    with pytest.raises(ValueError,match='receipt_digest'):f.freeze(receipt,'0'*64)

def test_handoff_cannot_add_or_omit_roots(scope):
    root,receipt,_,_=scope
    value=json.loads((root/receipt).read_text());value['roots']=['data/recorded-drive-validation'];pin=put(root/receipt,value)
    with pytest.raises(ValueError,match='writer_stop_handoff'):f.freeze(receipt,pin)

def test_unknown_roots_and_tags_cannot_include_user_recordings(scope):
    _,receipt,pin,_=scope
    with pytest.raises(ValueError,match='exact_public_cycle_roots'):f.validate_policies({'데이터':('recordings',[], '2026-11-11')})
    changed=dict(f.POLICIES);name=next(iter(changed));changed[name]=('recordings',[], '2026-11-11')
    with pytest.raises(ValueError,match='fixed_scope_tag'):f.validate_policies(changed)

def test_immutable_history_tampering_refused(scope):
    root,receipt,pin,_=scope
    (root/'data/research-retention/old-plan.json').write_text('{}')
    with pytest.raises(ValueError,match='historical_control'):f.freeze(receipt,pin)

def test_later_source_expiry_cannot_extend_mixed_deadline(scope):
    root,receipt,pin,_=scope
    put(root/'data/research/mwpd-v6/source.json',{'expires_at':'2099-01-01T00:00:00Z'})
    with pytest.raises(ValueError,match='expiry_date_ceiling'):f.freeze(receipt,pin)

def test_missing_expiry_never_uses_mtime_or_current_date(scope):
    root,receipt,pin,_=scope
    put(root/'data/research/mwpd-v6/source.json',{'downloaded_at':'2026-11-04T00:00:00Z'})
    with pytest.raises(ValueError,match='original_expiry'):f.freeze(receipt,pin)

def test_changed_asset_during_freeze_refused_without_inventory(scope,monkeypatch):
    root,receipt,pin,source=scope
    original=r.file_snapshot;changed=False
    def mutate(path):
        nonlocal changed
        result=original(path)
        if path==source and not changed:
            changed=True;source.write_bytes(b'SYNTHETIC-CHANGED')
        return result
    monkeypatch.setattr(r,'file_snapshot',mutate)
    with pytest.raises(ValueError,match='public_cycle_file_changed'):f.freeze(receipt,pin)
    assert not (root/'data/research-retention/public-cycle-inventory.json').exists()

def test_junction_escape_is_refused(scope,monkeypatch):
    root,receipt,pin,source=scope
    original=Path.is_junction
    monkeypatch.setattr(Path,'is_junction',lambda self:self==source.parent or original(self))
    with pytest.raises(ValueError,match='links_not_allowed'):f.freeze(receipt,pin)

def test_absent_optional_negative_cohort_is_skipped_without_expiry_invention(scope,monkeypatch):
    root,receipt,_,_=scope
    name='data/training/research-pothole-clean-v6'
    policies=dict(f.POLICIES);policies[name]=('pothole_clean_evaluation',['data/research/rdd2020/source.json',name+'/release.json'],'2026-11-10')
    monkeypatch.setattr(f,'POLICIES',policies)
    value=json.loads((root/receipt).read_text());value['roots']=sorted(policies);pin=put(root/receipt,value)
    result=f.freeze(receipt,pin)
    assert result['declared_collection_roots']==3 and result['explicit_collection_roots']==2 and result['absent_optional_roots']==1
    assert not (root/name).exists()

def test_present_but_incomplete_optional_cohort_cannot_be_silently_skipped(scope,monkeypatch):
    root,receipt,_,_=scope
    name='data/training/research-pothole-clean-v6';(root/name).mkdir()
    policies=dict(f.POLICIES);policies[name]=('pothole_clean_evaluation',['data/research/rdd2020/source.json',name+'/release.json'],'2026-11-10')
    monkeypatch.setattr(f,'POLICIES',policies)
    value=json.loads((root/receipt).read_text());value['roots']=sorted(policies);pin=put(root/receipt,value)
    with pytest.raises(ValueError,match='file_missing'):f.freeze(receipt,pin)
    assert not (root/'data/research-retention/public-cycle-inventory.json').exists()

@pytest.mark.parametrize('missing_frame',[False,True])
def test_soak_registers_only_frame_outputs_and_inherits_all_old_proofs(scope,monkeypatch,missing_frame):
    root,receipt,_,_=scope
    name=f.SOAK;directory=root/name;(directory/'analysis').mkdir(parents=True)
    proofs=[]
    for number,hour in enumerate((12,11,10)):
        evidence=f'data/training/old-model-{number}/provenance.json'
        put(root/evidence,{'assets':[{'expires_at':f'2026-11-10T{hour:02d}:00:00Z'}]});proofs.append(evidence)
    put(directory/'summary.json',{'source_expires_at':'2026-11-10T09:00:00Z','sample_id':'force-preserved-summary','saved_PNGs':1,'inferred_frames':1})
    proofs.append(name+'/summary.json')
    put(directory/'plan.json',{'path':'force-preserved-origin-plan'})
    (directory/'executed-probe-v1.py').write_bytes(b'SYNTHETIC EXECUTION PROVENANCE')
    (directory/'generated-geometry.png').write_bytes(b'SYNTHETIC INPUT PRESERVED')
    put(directory/'memory-samples.json',[{'memory_bytes':100}]);put(directory/'progress.json',{'frames_completed':1})
    frame=directory/'analysis/000000000.redacted.png';frame.write_bytes(b'SYNTHETIC OUTPUT')
    (directory/'analysis/unclassified.png').write_bytes(b'KEEP UNCLASSIFIED')
    put(directory/'analysis/analysis.json',{'source_path':'SYNTHETIC','observations':[]})
    policies=dict(f.POLICIES);policies[name]=('synthetic_soak',proofs,'2026-11-10');monkeypatch.setattr(f,'POLICIES',policies)
    value=json.loads((root/receipt).read_text());value['roots']=sorted(policies);pin=put(root/receipt,value)
    if missing_frame:
        frame.unlink()
        with pytest.raises(ValueError,match='soak_completed_output_count'):f.freeze(receipt,pin)
        assert not (root/'data/research-retention/public-cycle-inventory.json').exists()
        return
    before={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
    result=f.freeze(receipt,pin)
    inventory=json.loads((root/'data/research-retention/public-cycle-inventory.json').read_text())
    soak=[item for item in inventory['entries'].values() if item['path'].startswith(name+'/')]
    assert {item['path'] for item in soak}=={name+'/analysis/000000000.redacted.png',name+'/analysis/analysis.json'}
    assert {item['expires_at'] for item in soak}=={'2026-11-10T09:00:00Z'}
    assert all((directory/key).read_bytes()==content for key,content in before.items())
    assert frame.exists() and result['source_files_deleted']==0

def test_three_explicit_RTK_folders_use_original_acquisition_and_own_evidence(scope,monkeypatch):
    root,receipt,_,_=scope
    source='data/public/rtk/acquisition.json';put(root/source,{'expires_at':'2027-02-01T12:00:00Z'})
    policies=dict(f.POLICIES)
    for name,hour in ((f.SCREEN,13),(f.SCREEN_FOLLOWUP,11),(f.SCREEN_FINAL,10)):
        directory=root/name;directory.mkdir(parents=True)
        (directory/'view.png').write_bytes(b'SYNTHETIC SCREENSHOT')
        put(directory/'private-evidence.json',{'expires_at':f'2027-02-01T{hour:02d}:00:00Z'})
        policies[name]=('rtk_screenshot',[source,name+'/private-evidence.json'],'2027-02-01')
    monkeypatch.setattr(f,'POLICIES',policies)
    value=json.loads((root/receipt).read_text());value['roots']=sorted(policies);pin=put(root/receipt,value)
    f.freeze(receipt,pin)
    inventory=json.loads((root/'data/research-retention/public-cycle-inventory.json').read_text())
    screens={item['path']:item for item in inventory['entries'].values() if item['root']=='data/demo-reference'}
    assert screens[f.SCREEN+'/view.png']['expires_at']=='2027-02-01T12:00:00Z'
    assert screens[f.SCREEN_FOLLOWUP+'/view.png']['expires_at']=='2027-02-01T11:00:00Z'
    assert screens[f.SCREEN_FINAL+'/view.png']['expires_at']=='2027-02-01T10:00:00Z'

@pytest.mark.parametrize('name',['data/cpu-inference-benchmark','data/gpu-inference-probe','data/gpu-model-sets/final-v6'])
def test_performance_case_material_inherits_caps_but_keeps_model_and_execution_proofs(scope,monkeypatch,name):
    root,receipt,_,_=scope;directory=root/name;directory.mkdir(parents=True)
    proof='data/research/rdd2020/source.json'
    put(directory/'private-plan.json',{'cases':[{'path':'SYNTHETIC-ONLY','expires_at':'2026-11-09T00:00:00Z'}]})
    put(directory/'private-checks.json',{'image_sha256':'a'*64,'passed':True})
    (directory/'blob.npy').write_bytes(b'SYNTHETIC-TENSOR-NOT-NUMPY')
    put(directory/'aggregate.json',{'passed':True});put(directory/'receipt.json',{'source_sha256':'b'*64})
    put(directory/'staging.json',{'format':'staging-cuda-model-set-v1','models':{'lane':{'manifest':{'path':'MODEL-ONLY'}}}})
    put(directory/'registered.json',{'format':'registered-cuda-model-set-v1','models':{'lane':{'manifest':{'path':'MODEL-ONLY'}}}})
    put(directory/'model.json',{'format':'onnx','weights':'weights.onnx'})
    (directory/'weights.onnx').write_bytes(b'SYNTHETIC-MODEL-NOT-ONNX')
    (directory/'execution-source.py').write_bytes(b'SOURCE PROVENANCE')
    (directory/'junit.xml').write_text('<testsuites><testsuite name="SYNTHETIC" tests="1"><testcase name="contract"/></testsuite></testsuites>')
    put(directory/'operator-profiles/trace.json',[{'name':'node','dur':1,'args':{'shape':[1,3,64,64]}}])
    run=directory/'pipeline-v1/run-0';run.mkdir(parents=True)
    (run/'000000000.redacted.png').write_bytes(b'SYNTHETIC-PIXELS')
    put(run/'analysis.json',{'source_path':'SYNTHETIC','local_material':{'expires_at':'2026-11-05T00:00:00Z'}})
    policies=dict(f.POLICIES);policies[name]=('performance_diagnostic',[proof],'2026-11-10');monkeypatch.setattr(f,'POLICIES',policies)
    value=json.loads((root/receipt).read_text());value['roots']=sorted(policies);pin=put(root/receipt,value)
    f.freeze(receipt,pin)
    inventory=json.loads((root/'data/research-retention/public-cycle-inventory.json').read_text())
    entries={item['path']:item for item in inventory['entries'].values() if item['path'].startswith(name+'/')}
    assert entries[name+'/private-plan.json']['expires_at']=='2026-11-09T00:00:00Z'
    assert entries[name+'/private-checks.json']['expires_at']=='2026-11-10T10:00:00Z'
    assert entries[name+'/blob.npy']['expires_at']=='2026-11-10T10:00:00Z'
    assert entries[name+'/pipeline-v1/run-0/analysis.json']['expires_at']=='2026-11-05T00:00:00Z'
    assert entries[name+'/pipeline-v1/run-0/000000000.redacted.png']['expires_at']=='2026-11-05T00:00:00Z'
    assert len(entries)==5
    assert all((directory/file).exists() for file in ['aggregate.json','receipt.json','staging.json','registered.json','weights.onnx','execution-source.py','junit.xml','operator-profiles/trace.json'])

def test_model_descriptor_with_cases_is_not_misclassified_as_minimal_provenance(scope):
    root,_,_,_=scope
    path=root/'data/cpu-inference-benchmark/descriptor.json'
    put(path,{'format':'registered-cuda-model-set-v1','models':{'lane':{'cases':[{'path':'SYNTHETIC'}]}}})
    assert not f.performance_provenance(path)

@pytest.mark.parametrize('name',['data/cpu-inference-benchmark-other','data/gpu-inference-probe-other','data/gpu-model-sets','data/gpu-model-sets/final-v7','data/gpu-model-sets/final-v6-other'])
def test_performance_allowlist_never_expands_to_siblings_or_parent_namespace(scope,name):
    with pytest.raises(ValueError,match='unknown_research_root'):r.roots([name])

def test_final_handoff_requires_all_seventeen_roots_including_latest_RTK(tmp_path,monkeypatch):
    assert len(f.POLICIES)==17
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'STATE',tmp_path/'data/research-retention')
    path=tmp_path/'data/research-retention/writer-stop.json'
    value={'cycle':'public-refinement-v6','writers_stopped':True,'confirmed_at':'2026-11-04T00:00:00Z','roots':sorted(set(f.POLICIES)-{f.SCREEN_FINAL})}
    pin=put(path,value)
    with pytest.raises(ValueError,match='writer_stop_handoff'):f.stopped_receipt('data/research-retention/writer-stop.json',pin)
    value['roots']=sorted(f.POLICIES);pin=put(path,value)
    assert f.stopped_receipt('data/research-retention/writer-stop.json',pin)==path
