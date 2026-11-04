"""New-cycle preparation on synthetic fixtures; no real research inventory changes."""
import hashlib,json
from pathlib import Path
import pytest
from scripts import freeze_cycle2_retention as f
from scripts import research_retention as r


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def case(tmp_path,monkeypatch):
    assert tmp_path.is_relative_to(r.ROOT)
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'STATE',tmp_path/'data/research-retention')
    state=r.STATE;plan=state/'historical-plan.json';plan_pin=put(plan,{'schema_version':1,'entries':[]})
    inventory=state/'historical-inventory.json';inventory_pin=put(inventory,{'entries':{},'control_plans':{
        'original':{'status':'registered','path':f.STATE+plan.name,'sha256':plan_pin}}})
    monkeypatch.setattr(f,'PINS',{plan.name:plan_pin,inventory.name:inventory_pin})
    proof='data/public/a2d2-cycle2-v5/source.json';pin=put(tmp_path/proof,{'expires_at':'2026-01-08T00:00:00Z'})
    policies={'data/public/a2d2-cycle2-v5':('lane_source',[proof],'2026-01-08T00:00:00Z'),
        'data/gpu-model-sets/io-overlap-v1':('performance',[proof],'2026-01-08T00:00:00Z')}
    monkeypatch.setattr(f,'POLICIES',policies);monkeypatch.setattr(f,'PROOF_PINS',{proof:pin})
    image=tmp_path/'data/public/a2d2-cycle2-v5/synthetic.png';image.write_bytes(b'SYNTHETIC')
    performance=tmp_path/'data/gpu-model-sets/io-overlap-v1/run';performance.mkdir(parents=True)
    (performance/'frame.png').write_bytes(b'SYNTHETIC-DERIVED')
    put(performance/'analysis.json',{'expires_at':'2026-01-02T00:00:00Z','observations':[{'image_path':'synthetic.png'}]})
    put(performance.parent/'aggregate.json',{'format':'aggregate','fps':11.4})
    receipt=state/'writer-stop.json';receipt_pin=put(receipt,{'cycle':f.CYCLE,'writers_stopped':True,'confirmed_at':'2026-01-01T00:00:00Z','roots':sorted(policies)})
    return tmp_path,receipt.relative_to(tmp_path).as_posix(),receipt_pin,image


def test_default_audit_only_reads_pinned_controls(case,monkeypatch):
    root,_,_,_=case
    monkeypatch.setattr(f.old,'list_files',lambda *a:pytest.fail('active asset scan'))
    assert f.audit_immutable()[1]['historical_inventories_verified']==1
    assert not (r.STATE/'cycle2-inventory.json').exists()


@pytest.mark.parametrize('mutation',['active','roots','digest'])
def test_stop_receipt_guards_before_any_asset_list(case,monkeypatch,mutation):
    root,receipt,pin,_=case
    value=r.read(root/receipt)
    if mutation=='active':value['writers_stopped']=False;pin=put(root/receipt,value)
    if mutation=='roots':value['roots']=['data'];pin=put(root/receipt,value)
    if mutation=='digest':pin='0'*64
    monkeypatch.setattr(f.old,'list_files',lambda *a:pytest.fail('scanned without handoff'))
    with pytest.raises(ValueError):f.freeze(receipt,pin)


def test_separate_inventory_preserves_history_and_shorter_local_cap(case):
    root,receipt,pin,image=case;before={name:(r.STATE/name).read_bytes() for name in f.PINS}
    result=f.freeze(receipt,pin);inventory=r.read(r.STATE/'cycle2-inventory.json')
    assert result['registered_files']==3 and result['source_files_deleted']==0 and image.exists()
    assert all((r.STATE/name).read_bytes()==blob for name,blob in before.items())
    dates={item['expires_at'] for item in inventory['entries'].values()}
    assert dates=={'2026-01-08T00:00:00Z','2026-01-02T00:00:00Z'}
    assert (root/'data/gpu-model-sets/io-overlap-v1/aggregate.json').exists()
    public=json.dumps(result);assert 'synthetic.png' not in public and 'image_path' not in public
    with pytest.raises(ValueError,match='new_cycle2_outputs'):f.freeze(receipt,pin)


def test_source_proof_extension_or_hash_change_refused(case):
    root,receipt,pin,_=case;proof=next(iter(f.PROOF_PINS))
    put(root/proof,{'expires_at':'2099-01-01T00:00:00Z'})
    with pytest.raises(ValueError,match='original_source_proof_changed'):f.freeze(receipt,pin)
    assert not (r.STATE/'cycle2-inventory.json').exists()


def test_immutable_old_inventory_change_refused_before_scan(case,monkeypatch):
    _,receipt,pin,_=case;(r.STATE/'historical-inventory.json').write_text('{}')
    monkeypatch.setattr(f.old,'list_files',lambda *a:pytest.fail('scanned with changed history'))
    with pytest.raises(ValueError,match='historical_control_snapshot_changed'):f.freeze(receipt,pin)


def test_unknown_model_bin_is_preserved_and_not_claimed_covered(case):
    root,receipt,pin,_=case;path=root/'data/gpu-model-sets/io-overlap-v1/model.bin';path.write_bytes(b'WEIGHTS')
    result=f.freeze(receipt,pin);inventory=r.read(r.STATE/'cycle2-inventory.json')
    assert path.read_bytes()==b'WEIGHTS' and result['unclassified_requires_operator_review']
    assert all(not row['path'].endswith('model.bin') for row in inventory['entries'].values())


def test_log_migration_preparation_uses_exact_files_only_and_never_moves(case):
    root,receipt,pin,_=case;source=root/'.cache/cycle2-lane.log';source.parent.mkdir()
    source.write_text('{"image_sha256":"SYNTHETIC"}')
    aggregate=root/'.cache/cycle2-numbers.log';aggregate.write_text('epoch=1 loss=.5')
    entries=[{'source':path.relative_to(root).as_posix(),'sha256':r.file_snapshot(path)[0]} for path in (source,aggregate)]
    manifest=r.STATE/'exact-log-list.json';digest=put(manifest,{'format':'cycle2-exact-log-list-v1','logs':entries})
    output=r.STATE/'log-migration-plan.json'
    result=f.prepare_log_migration(receipt,pin,manifest.relative_to(root).as_posix(),digest,output.relative_to(root).as_posix())
    assert result['planned_private_log_moves']==1 and result['aggregate_logs_preserved']==1 and result['moves_executed'] is False
    assert source.exists() and aggregate.exists() and not (root/'data/training/cycle2-logs').exists()


@pytest.mark.parametrize('name',['.cache/other.log','.cache/cycle2/nested.log','data/cycle2-lane.log','.cache/cycle2-weights.bin'])
def test_log_list_does_not_authorize_arbitrary_namespace(case,name):
    root,receipt,pin,_=case;manifest=r.STATE/'bad-log-list.json'
    digest=put(manifest,{'format':'cycle2-exact-log-list-v1','logs':[{'source':name,'sha256':'a'*64}]})
    with pytest.raises(ValueError,match='exact_cycle2_cache_log_required'):
        f.prepare_log_migration(receipt,pin,manifest.relative_to(root).as_posix(),digest,(r.STATE/'must-not-exist.json').relative_to(root).as_posix())


def test_changed_asset_before_final_snapshot_creates_no_inventory(case,monkeypatch):
    _,receipt,pin,image=case;real=r.file_snapshot;changed=False
    def snapshot(path):
        nonlocal changed
        value=real(path)
        if path==image and not changed:changed=True;image.write_bytes(b'CHANGED-SYNTHETIC')
        return value
    monkeypatch.setattr(r,'file_snapshot',snapshot)
    with pytest.raises(ValueError,match='asset_or_proof_changed'):f.freeze(receipt,pin)
    assert not (r.STATE/'cycle2-inventory.json').exists() and not (r.STATE/'cycle2-frozen-plan.json').exists()


def test_failed_original_release_is_explicit_optional_not_a_missing_required_root(case,monkeypatch):
    root,receipt,_,_=case;name='data/training/research-pothole-rdd2022-blind-v7'
    policies={**f.POLICIES,name:('blind_release',['not-read-when-absent.json'],'2026-01-08T00:00:00Z')}
    monkeypatch.setattr(f,'POLICIES',policies)
    value=r.read(root/receipt);value['roots']=sorted(policies);pin=put(root/receipt,value)
    result=f.freeze(receipt,pin)
    assert result['declared_roots']==3 and result['explicit_roots']==2 and result['absent_optional_roots']==1
