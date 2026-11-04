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
