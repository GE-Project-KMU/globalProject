"""Synthetic full-run/deadline/quality contracts; never reads user recordings."""
from datetime import datetime,timedelta,timezone
import json
import numpy as np
import pytest
from test_recorded_mcap import folder,write
from agent.paths import sha256
from agent.recorded_mcap import RecordedMcapReader
from scripts.validate_recorded_mcap import DeadlineGuard,ProcessingDeadlineExceeded,MAX_AUTHORIZED,permission_guard,guarded_hash,Quality,validate_one,aggregate,private_output_path,public_output_path,control_metadata_unchanged,PRIVATE_ROOT

NOW=datetime.fromisoformat('2026-11-04T12:30:00+00:00')
def row(path):return {'source_path':str(path),'sha256':sha256(path),'size_bytes':path.stat().st_size}
def guard():return DeadlineGuard(MAX_AUTHORIZED,now=lambda:NOW)
def receipt():return {'format':'recorded-drive-validation-permission-v1','scope':'this recording validation only','source':'explicit user reply in this chat',
    'authorized_until':'2026-11-05T23:59:00+09:00','authorized_until_utc':'2026-11-05T14:59:00+00:00',
    **{key:False for key in ['original_capture_date_changed','base_retention_policy_changed','automatic_delete_authorized','automatic_retention_registration','external_upload_authorized','training_authorized_by_this_exception','raw_pixel_outputs_planned']}}

def test_future_deadline_cannot_expand_authorization():
    with pytest.raises(ValueError,match='exceeds'):DeadlineGuard(MAX_AUTHORIZED+timedelta(seconds=1),now=lambda:NOW)
@pytest.mark.parametrize('offset',[0,1])
def test_at_or_after_cutoff_is_rejected(offset):
    with pytest.raises(ProcessingDeadlineExceeded):DeadlineGuard(MAX_AUTHORIZED,now=lambda:MAX_AUTHORIZED+timedelta(seconds=offset))
def test_naive_clocks_rejected():
    with pytest.raises(ValueError):DeadlineGuard(MAX_AUTHORIZED.replace(tzinfo=None),now=lambda:NOW)
    with pytest.raises(ValueError):DeadlineGuard(MAX_AUTHORIZED,now=lambda:NOW.replace(tzinfo=None))

def test_valid_receipt_and_scope_mismatch_fail_closed(folder):
    path=folder/'permission.json';path.write_text(json.dumps(receipt()),encoding='utf-8');g,value=permission_guard(path,now=lambda:NOW);assert g.deadline==MAX_AUTHORIZED
    value['external_upload_authorized']=True;path.write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='scope_mismatch'):permission_guard(path,now=lambda:NOW)
    value=receipt();value['authorized_until_utc']='2026-11-05T15:00:00+00:00';path.write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='disagreement'):permission_guard(path,now=lambda:NOW)

def test_full_memory_decode_before_after_hash_quality_eof(folder):
    path=folder/'nonpersonal.svo2';write(path);r=validate_one(row(path),guard())
    assert r['full_contract_passed'] and r['source_checksum_unchanged'] and r['eof']
    assert r['frames']==r['video_messages']==r['decoder_frames']==3
    assert r['integrity']['fully_consumed'] and r['timestamp_mismatches']==0
    assert r['quality']['left']['frames']==3 and r['quality']['left']['flags']['std_lt_2']==3
    assert sorted(x.name for x in folder.iterdir())==['nonpersonal.svo2']

def test_wrong_intake_hash_prevents_decoder_creation(folder):
    path=folder/'original.svo2';write(path);source=row(path);source['sha256']='0'*64
    def forbidden(*a,**kw):raise AssertionError('decoder_must_not_open')
    result=validate_one(source,guard(),reader_factory=forbidden)
    assert result['status']=='failed' and 'checksum' in result['reason'] and result['frames']==0

def test_permission_expiry_before_open_stops_without_hash_or_decoder(folder):
    path=folder/'original.svo2';write(path);source=row(path);clock=[NOW];g=DeadlineGuard(MAX_AUTHORIZED,now=lambda:clock[0]);clock[0]=MAX_AUTHORIZED
    def forbidden(*a,**kw):raise AssertionError('decoder_must_not_open')
    result=validate_one(source,g,reader_factory=forbidden)
    assert result['status']=='deadline_stopped' and 'sha256_before' not in result and result['frames']==0

def test_expiry_inside_reader_closes_without_returning_pixels(folder):
    path=folder/'original.svo2';write(path);clock=[NOW];g=DeadlineGuard(MAX_AUTHORIZED,now=lambda:clock[0])
    reader=RecordedMcapReader(path,processing_guard=g);clock[0]=MAX_AUTHORIZED
    with pytest.raises(ProcessingDeadlineExceeded):reader.read()
    assert reader.closed and reader.stream.closed

def test_hash_checks_deadline_between_chunks(folder):
    path=folder/'small.bin';path.write_bytes(b'nonpersonal synthetic bytes');calls=[0]
    def cutoff():
        calls[0]+=1
        if calls[0]==3:raise ProcessingDeadlineExceeded('stop')
    with pytest.raises(ProcessingDeadlineExceeded):guarded_hash(path,cutoff)
    assert calls[0]==3

def test_post_decode_source_change_is_detected(folder):
    path=folder/'mutable-synthetic.svo2';write(path);source=row(path)
    class MutatingSyntheticReader(RecordedMcapReader):
        def __exit__(self,*args):
            super().__exit__(*args)
            with path.open('ab') as stream:stream.write(b'INTENTIONAL SYNTHETIC MUTATION')
    result=validate_one(source,guard(),reader_factory=MutatingSyntheticReader)
    assert result['frames']==3 and result['status']=='full_contract_failed' and not result['source_checksum_unchanged']

def test_quality_flags_do_not_invent_invalid_labels():
    q=Quality();q.add(np.zeros((8,8,3),np.uint8));q.add(np.full((8,8,3),255,np.uint8));s=q.summary()
    assert s['frames']==2 and s['flags']['mean_lt_5']==s['flags']['mean_gt_250']==1
    assert s['heuristics_only_no_GT_no_auto_exclusion']

def test_public_aggregate_excludes_original_identifiers_clocks_and_hashes(folder):
    path=folder/'SENSITIVE-SOURCE-NAME.svo2';write(path);result=validate_one(row(path),guard())
    private={'results':[result],'complete':True,'source_files_expected':1,'reader_sha256':'CODEHASH','validator_sha256':'CODEHASH2','processing_guard_checks':123,'control_metadata_unchanged':True}
    public=aggregate(private);text=json.dumps(public)
    assert 'SENSITIVE' not in text and result['expected_sha256'] not in text and str(result['log_time_min_ns']) not in text
    assert public['status']=='full_decode_verified' and public['decoded_frames']==3

@pytest.mark.parametrize('path',['docs/reports/private.json','data/agent/private.json','data/recorded-drive-validation/../escape.json','data/recorded-drive-validation'])
def test_private_output_root_restriction(path):
    with pytest.raises(ValueError,match='private_report'):private_output_path(path)
def test_private_output_safe_child_and_existing_rejection(folder,monkeypatch):
    from scripts import validate_recorded_mcap as module
    monkeypatch.setattr(module,'PRIVATE_ROOT',folder)
    path=private_output_path(folder/'synthetic-unused-output.json');assert path.is_relative_to(folder)
    existing=folder/'existing.json';existing.write_text('{}')
    with pytest.raises(ValueError,match='fresh_output'):private_output_path(existing)
@pytest.mark.parametrize('path',['data/recorded-drive-validation/public.json','데이터/public.json','docs/other.json'])
def test_public_output_is_aggregate_directory_only(path):
    with pytest.raises(ValueError,match='public_aggregate'):public_output_path(path)
def test_pinned_control_metadata_change_cannot_remain_success(folder):
    permission=folder/'p.json';inventory=folder/'i.json';permission.write_text('{}');inventory.write_text('{}')
    report={'permission_sha256':sha256(permission),'source_inventory_sha256':sha256(inventory)}
    assert control_metadata_unchanged(permission,inventory,report)
    inventory.write_text('{"mutation":true}')
    assert not control_metadata_unchanged(permission,inventory,report)
    path=folder/'synthetic.svo2';write(path);result=validate_one(row(path),guard())
    private={'results':[result],'complete':True,'source_files_expected':1,'reader_sha256':'CODE','validator_sha256':'CODE2','processing_guard_checks':1,'control_metadata_unchanged':False}
    assert aggregate(private)['status']=='incomplete_or_failed'
