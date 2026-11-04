"""Synthetic narrow official-ZIP range cache retention; no actual asset scans."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import pytest
from scripts import research_retention as r


@pytest.fixture
def case(tmp_path,monkeypatch):
    assert tmp_path.resolve().is_relative_to(r.ROOT)
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'STATE',tmp_path/'data/research-retention')
    folder=tmp_path/r.RANGE_ROOT;folder.mkdir(parents=True)
    source=folder.parent/'source.json';source.write_text(json.dumps({'expires_at':'2026-01-08T00:00:00Z'}))
    def put(name='0-7.bin',data=b'SYNTHETI'):
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        path.with_suffix('.sha256').write_text(hashlib.sha256(data).hexdigest(),encoding='ascii');return path
    return tmp_path,folder,source,put


def test_only_exact_range_namespace_with_correct_sidecar_is_admitted(case):
    root,folder,_,put=case;path=put();r.eligible(path,'source_range')
    for other in [root/'data/training/model/0-7.bin',folder/'nested/0-7.bin',folder/'weights.bin',folder/'00-7.bin']:
        other.parent.mkdir(parents=True,exist_ok=True);other.write_bytes(b'SYNTHETI')
        other.with_suffix('.sha256').write_text(hashlib.sha256(other.read_bytes()).hexdigest())
        with pytest.raises(ValueError):r.eligible(other,'source_range')
    with pytest.raises(ValueError):r.eligible(path,'source_archive')
    with pytest.raises(ValueError):r.roots(['data/research','data/gpu-model-sets'])


@pytest.mark.parametrize('name,data',[
    ('8-7.bin',b''),('0-4194304.bin',b'x'),('13264172619-13264172619.bin',b'x'),('0-8.bin',b'wrong'),
])
def test_archive_range_block_and_exact_length_bounds(case,name,data):
    _,_,_,put=case
    with pytest.raises(ValueError,match='bounds_or_size'):r.eligible(put(name,data),'source_range')


def test_missing_or_changed_sidecar_and_changed_source_fail_closed(case):
    _,_,_,put=case;path=put();proof=path.with_suffix('.sha256')
    proof.unlink()
    with pytest.raises(ValueError):r.eligible(path,'source_range')
    proof.write_text('f'*64)
    with pytest.raises(ValueError,match='sidecar_mismatch'):r.eligible(path,'source_range')
    proof.write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    with pytest.raises(ValueError,match='bounded_range_hash_sidecar'):r.eligible(path,'source_range')


def test_registered_exact_range_dryrun_then_apply_preserves_model_binary_and_sidecar(case):
    root,folder,source,put=case;path=put();untouched=folder/'model.bin';untouched.write_bytes(b'WEIGHTS')
    relative=lambda p:p.relative_to(root).as_posix()
    plan=r.STATE/'range-plan.json';plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({'schema_version':1,'roots':['data/research/rdd2022-blind-v7'],'entries':[
        {'path':relative(path),'kind':'source_range','sha256':r.file_snapshot(path)[0],
         'evidence':[{'path':relative(source),'sha256':r.file_snapshot(source)[0]}]}]}))
    inventory=r.STATE/'range-inventory.json';r.register(relative(plan),relative(inventory))
    before=inventory.read_bytes();now=datetime(2026,1,9,tzinfo=timezone.utc)
    dry=r.purge(relative(inventory),now=now)
    assert dry['counts']['would_delete']==1 and inventory.read_bytes()==before and path.exists()
    actual=r.purge(relative(inventory),now=now,apply=True)
    assert actual['counts']['deleted']==1 and not path.exists()
    assert untouched.read_bytes()==b'WEIGHTS' and path.with_suffix('.sha256').exists()


def test_links_refused_before_reading_range_payload(case,monkeypatch):
    _,_,_,put=case;path=put();real=Path.is_symlink
    monkeypatch.setattr(Path,'is_symlink',lambda p:p==path or real(p))
    with pytest.raises(ValueError,match='links_not_allowed'):r.eligible(path,'source_range')


def test_valid_archive_last_byte_and_maximum_block(case):
    _,_,_,put=case
    end=r.RANGE_ARCHIVE_BYTES-1;r.eligible(put(f'{end}-{end}.bin',b'x'),'source_range')
    r.eligible(put(f'0-{r.RANGE_MAX_BYTES-1}.bin',b'x'*r.RANGE_MAX_BYTES),'source_range')
