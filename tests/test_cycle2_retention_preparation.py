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
    monkeypatch.setattr(f,'PROTECTED_CONTROLS',{})
    monkeypatch.setattr(f.writer_guard,'inspect_writers',lambda root:{'known_writers':[]})
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


def test_exact_expired_prototype_npys_keep_original_cap_and_preserve_models_context_aggregate(case,monkeypatch):
    root,receipt,_,_=case;folder=root/f.PROTOTYPE;folder.mkdir(parents=True)
    context=root/f.PROTOTYPE_CONTEXT
    digest=put(context,{'format':'redaction-rescue-synthetic-artifact-context-v1',
        'synthetic_tensor_temporary_expiry':'2026-01-02T00:00:00Z','legacy_source_earliest_expiry':'2026-01-08T00:00:00Z'})
    for name in f.PROTOTYPE_NPY:(folder/name).write_bytes(b'SYNTHETIC-NEVER-LOADED')
    (folder/'prototype.onnx').write_bytes(b'PRESERVED-MODEL')
    put(folder/'aggregate.json',{'raw_difference':0,'synthetic':True})
    put(folder/'provenance.json',{'format':'training-provenance-v1','assets':[]})
    policies={**f.POLICIES,f.PROTOTYPE:('synthetic_prototype',[f.PROTOTYPE_CONTEXT],'2026-01-02T00:00:00Z')}
    monkeypatch.setattr(f,'POLICIES',policies);monkeypatch.setattr(f,'PROOF_PINS',{**f.PROOF_PINS,f.PROTOTYPE_CONTEXT:digest})
    value=r.read(root/receipt);value['roots']=sorted(policies);pin=put(root/receipt,value)
    import numpy as np
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('expired prototype tensor reused'))
    result=f.freeze(receipt,pin);inventory=r.read(r.STATE/'cycle2-inventory.json')
    actual=[item for item in inventory['entries'].values() if item['path'].startswith(f.PROTOTYPE+'/')]
    assert len(actual)==3 and {item['expires_at'] for item in actual}=={'2026-01-02T00:00:00Z'}
    assert {Path(item['path']).name for item in actual}==f.PROTOTYPE_NPY
    assert context.exists() and (folder/'prototype.onnx').exists() and (folder/'aggregate.json').exists()
    assert result['source_files_deleted']==0 and all((folder/name).exists() for name in f.PROTOTYPE_NPY)
    assert f.category(folder/'unrelated-model.npy','synthetic_prototype') is None
    assert f.category(folder/'nested/synthetic-input.npy','synthetic_prototype') is None


def test_prototype_expiry_aliases_are_not_generic_and_no_now_fallback():
    value={'synthetic_tensor_temporary_expiry':'2026-01-02T00:00:00Z','legacy_source_earliest_expiry':'2026-01-08T00:00:00Z'}
    assert r.evidence_expiries(value)==[]
    assert min(r.evidence_expiries(value|{'format':'redaction-rescue-synthetic-artifact-context-v1'}))==r.clock('2026-01-02T00:00:00Z')
    with pytest.raises(KeyError):r.evidence_expiries({'format':'redaction-rescue-synthetic-artifact-context-v1'})


def test_family_minimal_control_preserved_only_without_individual_data(case):
    root,_,_,_=case;path=root/'control.json'
    put(path,{'format':'pothole-family-v9-validation-freeze-v1','model':{'path':'data/training/experiment/model.json','sha256':'a'*64}})
    assert f.minimal_control_provenance(path)
    put(path,{'format':'pothole-family-v9-validation-freeze-v1','image_sha256':'a'*64})
    assert not f.minimal_control_provenance(path)
    put(path,{'format':'pothole-family-one-shot-comparison-v1','path':'data/public/frame.jpg'})
    assert not f.minimal_control_provenance(path)
    put(path,{'format':'pothole-family-one-shot-comparison-v1','unexpected_reference':'data/public/frame.jpg'})
    assert not f.minimal_control_provenance(path)


def test_gpu_device_samples_and_numeric_supervisor_logs_are_not_personal(case):
    root,_,_,_=case;path=root/'device-samples.json'
    put(path,{'format':'small-v9-sequential-restart-supervision-v1','device_memory':{'samples':[{'receive_utc':'2026-01-01T00:00:00Z','used_bytes':123}]},'sources_unchanged':True})
    with pytest.raises(r.PreservedResearchFile):r.eligible(path,'identifiable_manifest')


@pytest.mark.parametrize('name,cap',[
    ('data/training/pothole-resolution-v10','2026-11-10T17:14:46.862846Z'),
    ('data/training/pothole-family-v10',f.BLIND_CAP),
    ('data/training/privacy-family-v8-evaluation',f.PRIVACY_CAP)])
def test_new_family_scopes_are_explicit_with_original_cap_and_no_parent_authority(name,cap):
    assert f.POLICIES[name][0]=='family_evaluation'
    assert f.POLICIES[name][2]==cap
    assert 'data/training' not in f.POLICIES
    assert 'data' not in f.POLICIES
    assert all('/' in proof for proof in f.POLICIES[name][1])


@pytest.mark.parametrize('format',[
    'pothole-family-v10-CPU-validation-freeze-v1',
    'pothole-family-v10-one-shot-comparison-v1',
    'lane-cycle2-retry1-completed-CPU-artifact-freeze-v1',
    'lane-cycle2-retry1-one-shot-comparison-v1'])
def test_v10_control_keeps_model_proof_but_never_individual_metadata(case,format):
    root,_,_,_=case;path=root/'new-control.json'
    value={'format':format,'manifest':{'path':'data/training/experiment/model.json','sha256':'a'*64}}
    put(path,value);assert f.minimal_control_provenance(path)
    put(path,value|{'image_sha256':'a'*64});assert not f.minimal_control_provenance(path)
    put(path,value|{'predictions':[{'bbox':[0,0,1,1]}]});assert not f.minimal_control_provenance(path)


def test_domain_v11_only_exact_archive_and_partial_names_are_classified(case):
    root,_,_,_=case;folder=root/f.DOMAIN_V11
    for name in f.DOMAIN_V11_ARCHIVES:
        assert f.category(folder/name,'domain_v11_source')=='source_archive'
        assert f.category(folder/'nested'/name,'domain_v11_source') is None
    for name in ('other.download','model.onnx.download','model.bin','other.zip','archive.part'):
        assert f.category(folder/name,'domain_v11_source') is None
    assert f.DOMAIN_V11 in r.KNOWN_ROOTS and 'data/research' not in r.KNOWN_ROOTS


def test_domain_v11_partial_uses_fixed_source_and_plan_minimum_without_deleting(case,monkeypatch):
    root,receipt,_,_=case;folder=root/f.DOMAIN_V11;folder.mkdir(parents=True)
    partial=folder/'RDD2022_released_through_CRDDC2022.zip.download';partial.write_bytes(b'PK-SYNTHETIC-PARTIAL')
    proof=folder/'transfer-plan.json';put(proof,{'format':'rdd2022-domain-v11-full-transfer-plan-v1','file_id':38030910,
        'archive_bytes':13_264_172_619,'publisher_md5':'b62bd51d2ffcfaa76c60f234f0cc2bb3','source_expires_at':'2026-11-10T17:07:24.286159Z',
        'expires_at':'2026-01-03T00:00:00Z'})
    # Reuse the fixture's pinned ordinary source JSON; no real archive scanning.
    first=next(iter(f.POLICIES.values()))[1][0]
    policies={**f.POLICIES,f.DOMAIN_V11:('domain_v11_source',[first,proof.relative_to(root).as_posix()],'2026-01-08T00:00:00Z')}
    monkeypatch.setattr(f,'POLICIES',policies)
    value=r.read(root/receipt);value['roots']=sorted(policies);pin=put(root/receipt,value)
    result=f.freeze(receipt,pin);inventory=r.read(r.STATE/'cycle2-inventory.json')
    rows=[entry for entry in inventory['entries'].values() if entry['path'].endswith('.zip.download')]
    assert len(rows)==1 and rows[0]['expires_at']=='2026-01-03T00:00:00Z'
    assert partial.read_bytes()==b'PK-SYNTHETIC-PARTIAL' and result['source_files_deleted']==0


def test_domain_v11_direct_register_eligibility_requires_exact_basename_sidecar_and_no_weight(case):
    root,_,_,_=case;folder=root/f.DOMAIN_V11;folder.mkdir(parents=True)
    unknown=folder/'other.download';unknown.write_bytes(b'SYNTHETIC')
    with pytest.raises(ValueError,match='exact_domain_v11'):r.eligible(unknown,'source_archive')
    partial=folder/(r.DOMAIN_ARCHIVE_NAME+'.download');partial.write_bytes(b'SYNTHETIC')
    with pytest.raises(ValueError,match='file_missing'):r.eligible(partial,'source_archive')
    put(folder/'transfer-plan.json',{'format':'rdd2022-domain-v11-full-transfer-plan-v1','file_id':38030910,
        'archive_bytes':13_264_172_619,'publisher_md5':'WRONG','source_expires_at':f.BLIND_CAP})
    with pytest.raises(ValueError,match='fixed_domain_v11'):r.eligible(partial,'source_archive')


def test_source_acquisition_control_has_no_individual_image_references(case):
    root,_,_,_=case;path=root/'acquisition-plan.json';format='rdd2022-domain-v11-bounded-acquisition-plan-v1'
    value={'format':format,'parent_expiry_evidence':{'path':'data/research/rdd2022-blind-v7/source.json','sha256':'a'*64}}
    put(path,value);assert f.minimal_control_provenance(path,{format})
    put(path,value|{'image_sha256':'a'*64});assert not f.minimal_control_provenance(path,{format})


def test_operator_logs_receive_collision_safe_prefix_and_keep_inputs(case):
    root,receipt,pin,_=case;entries=[]
    for namespace in ('privacy-family-v8-operator','privacy-family-v8-operator-retry1'):
        path=root/'.cache'/namespace/'strict-winner-face.log';path.parent.mkdir(parents=True)
        path.write_text('{"image_sha256":"SYNTHETIC"}')
        entries.append({'source':path.relative_to(root).as_posix(),'sha256':r.file_snapshot(path)[0]})
    manifest=r.STATE/'logs.json';digest=put(manifest,{'format':'cycle2-exact-log-list-v1','logs':entries})
    output=r.STATE/'moves.json';f.prepare_log_migration(receipt,pin,manifest.relative_to(root).as_posix(),digest,output.relative_to(root).as_posix())
    moves=r.read(output)['moves'];assert len({x['to_log'] for x in moves})==2
    assert all((root/x['from_log']).exists() for x in moves)
    assert all(not (root/x['to_log']).exists() for x in moves)


def test_pinned_nonpersonal_log_never_moved_and_changes_refused(case,monkeypatch):
    root,receipt,pin,_=case;path=root/'.cache/cycle2-rescue-v8-preflight.log';path.parent.mkdir();path.write_text('error_type=ModuleNotFoundError')
    digest=r.file_snapshot(path)[0];monkeypatch.setattr(f,'PROTECTED_CONTROLS',{path.relative_to(root).as_posix():digest})
    manifest=r.STATE/'logs.json';sha=put(manifest,{'format':'cycle2-exact-log-list-v1','logs':[{'source':path.relative_to(root).as_posix(),'sha256':digest}]})
    out=r.STATE/'moves.json';result=f.prepare_log_migration(receipt,pin,manifest.relative_to(root).as_posix(),sha,out.relative_to(root).as_posix())
    assert result['planned_private_log_moves']==0 and path.exists()
    path.write_text('changed')
    with pytest.raises(ValueError,match='control_changed'):f.preparation_guard()


def test_new_writer_blocks_log_plan_before_publication(case,monkeypatch):
    root,receipt,pin,_=case;path=root/'.cache/cycle2-private.log';path.parent.mkdir();path.write_text('{"image_sha256":"SYNTHETIC"}')
    manifest=r.STATE/'logs.json';digest=put(manifest,{'format':'cycle2-exact-log-list-v1','logs':[{'source':path.relative_to(root).as_posix(),'sha256':r.file_snapshot(path)[0]}]})
    count=0
    def guard(root):
        nonlocal count
        count+=1
        if count==2:raise ValueError('known_project_research_writer_active')
    monkeypatch.setattr(f.writer_guard,'require_stopped',guard)
    out=r.STATE/'moves.json'
    with pytest.raises(ValueError,match='writer_active'):f.prepare_log_migration(receipt,pin,manifest.relative_to(root).as_posix(),digest,out.relative_to(root).as_posix())
    assert count==2 and not out.exists() and path.exists()
