"""Synthetic, workspace-contained safety checks for the new research release."""
import hashlib,json,tempfile
from pathlib import Path
import pytest
from scripts import prepare_a2d2_improvement as prep

@pytest.fixture
def dataset(tmp_path,monkeypatch):
    # pytest's configured temp base is project-local; no real research data used.
    monkeypatch.setattr(prep,'ROOT',tmp_path)
    monkeypatch.setattr(prep,'BASE',tmp_path/'data/public/a2d2-improvement-v2')
    old=tmp_path/'old.json';old.write_text(json.dumps([{'recording':x} for x in ['20180807_145028','20180810_142822','20180925_101535','20181008_095521','20181108_084007','20181204_135952']]))
    monkeypatch.setattr(prep,'OLD',old)
    def listing(prefix):
        tag='_label_' if '/label/' in prefix else '_camera_'
        return [{'key':prefix+'frame'+tag+str(i).zfill(4)+'.png','bytes':4,'etag':'0'*32} for i in range(120)]
    monkeypatch.setattr(prep.source,'listing',listing)
    return tmp_path

def test_pretraining_plan_is_stable_date_disjoint_and_counts(dataset):
    plan=prep.freeze_plan();before=(prep.BASE/'pretraining-plan.json').read_bytes()
    assert sum(x['split']=='train' for x in plan['new_pairs'])==400
    assert sum(x['split']=='test' for x in plan['new_pairs'])==100
    assert not {x[:8] for x in plan['fresh_test_recordings']}&{x[:8] for x in plan['additional_training_recordings']}
    assert prep.freeze_plan()==plan
    assert (prep.BASE/'pretraining-plan.json').read_bytes()==before
    prep.OLD.write_text('[]')
    with pytest.raises(ValueError,match='old_release_changed'):prep.freeze_plan()

def test_publisher_digest_mismatch_writes_nothing(dataset,monkeypatch):
    monkeypatch.setattr(prep.source,'get',lambda url:b'abcd')
    item={'split':'test','image':{'key':'safe/image.png','bytes':4,'etag':'0'*32}}
    with pytest.raises(ValueError,match='md5_mismatch'):prep.acquire(item,'image')
    assert not list(dataset.rglob('*.png'))

def test_fresh_test_download_is_separate_and_existing_bytes_rechecked(dataset,monkeypatch):
    raw=b'abcd';monkeypatch.setattr(prep.source,'get',lambda url:raw)
    item={'split':'test','image':{'key':'safe/image.png','bytes':4,'etag':hashlib.md5(raw).hexdigest()}}
    path=prep.acquire(item,'image')
    assert path.is_relative_to(prep.BASE/'quarantined-test')
    path.write_bytes(b'bad!')
    with pytest.raises(ValueError,match='md5_mismatch'):prep.acquire(item,'image')

def test_publisher_object_path_cannot_escape_dataset(dataset,monkeypatch):
    item={'split':'train','image':{'key':'../../../../outside.png','bytes':4,'etag':'0'*32}}
    with pytest.raises(ValueError,match='outside_dataset'):prep.acquire(item,'image')
