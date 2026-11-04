import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest
from scripts import acquire_rdd2022_blind_api as tool

def fixture_attempt(tmp_path,monkeypatch):
    source=tmp_path/'source.json'
    source.write_text(json.dumps({'expires_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()}))
    (tmp_path/'selection-protocol.json').write_text('{}')
    monkeypatch.setattr(tool.selection,'scoped_output',lambda _:tmp_path)
    def digest(path):
        if Path(path)==Path(tool.selection.__file__):return tool.SELECTION_SHA
        if Path(path).name=='selection-protocol.json':return tool.PROTOCOL_SHA
        return 'a'*64
    monkeypatch.setattr(tool,'sha256',digest)
    return tmp_path/'official-api-transport-attempt.json'

def test_transport_only_override_records_frozen_source_and_expiry(tmp_path,monkeypatch):
    receipt=fixture_attempt(tmp_path,monkeypatch);observed=[]
    monkeypatch.setattr(tool.selection,'prepare',lambda:observed.append(tool.selection.URL))
    monkeypatch.setattr(tool.selection,'URL',tool.selection.URL)
    tool.run();record=json.loads(receipt.read_text())
    assert observed==[tool.OFFICIAL_API]
    assert record['status']=='completed' and record['model_inferences']==0
    assert record['selection_source_sha256']==tool.SELECTION_SHA
    assert record['source_expires_at']==json.loads((tmp_path/'source.json').read_text())['expires_at']
    with pytest.raises(ValueError,match='already_exists'):tool.run()

def test_failed_transport_is_evidence_not_success(tmp_path,monkeypatch):
    receipt=fixture_attempt(tmp_path,monkeypatch)
    def fail():raise RuntimeError('private signed URL must not enter receipt')
    monkeypatch.setattr(tool.selection,'prepare',fail)
    monkeypatch.setattr(tool.selection,'URL',tool.selection.URL)
    with pytest.raises(RuntimeError):tool.run()
    raw=receipt.read_text();record=json.loads(raw)
    assert record['status']=='failed' and record['error_type']=='RuntimeError'
    assert 'private signed' not in raw

def test_frozen_source_mismatch_rejected_before_network(tmp_path,monkeypatch):
    receipt=fixture_attempt(tmp_path,monkeypatch)
    monkeypatch.setattr(tool,'sha256',lambda _: '0'*64)
    monkeypatch.setattr(tool.selection,'prepare',lambda:pytest.fail('must not acquire'))
    with pytest.raises(ValueError,match='source_changed'):tool.run()
    assert not receipt.exists()

def test_expired_original_cannot_gain_new_ttl(tmp_path,monkeypatch):
    receipt=fixture_attempt(tmp_path,monkeypatch)
    (tmp_path/'source.json').write_text(json.dumps({'expires_at':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()}))
    monkeypatch.setattr(tool.selection,'prepare',lambda:pytest.fail('must not acquire'))
    with pytest.raises(ValueError,match='source_expired'):tool.run()
    assert not receipt.exists()
