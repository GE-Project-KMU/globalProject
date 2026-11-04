import hashlib,json,uuid
from datetime import datetime,timedelta,timezone
import pytest
from agent.dataset import validate_research_release
from scripts.build_rdd2022_us100_offline import select,materialize_and_validate

def test_UUID_order_is_fixed_per_class_and_output_never_mutates_candidate_labels():
    rows=[{'sample_id':f'{i:04d}','objects':[{'class':'pothole'}] if i%2 else []} for i in range(120,0,-1)]
    chosen=select(rows)
    assert [r['sample_id'] for r in chosen[:50]]==[f'{i:04d}' for i in range(1,100,2)]
    assert [r['sample_id'] for r in chosen[50:]]==[f'{i:04d}' for i in range(2,101,2)]
    chosen[0]['objects'][0]['class']='changed'
    assert all(r['objects']==[{'class':'pothole'}] for r in rows if r['objects'])

def test_short_class_support_fails_without_changing_target():
    with pytest.raises(ValueError,match='balanced_support'):select([{'sample_id':str(i),'objects':[{}]} for i in range(100)])

def test_duplicate_original_identity_is_not_silently_dropped_or_replaced():
    with pytest.raises(ValueError,match='unique_candidate'):select([{'sample_id':'same','objects':[]}]*100)

def local_row(tmp_path):
    release=tmp_path/'new-release';evidence=tmp_path/'license.json'
    evidence.write_text(json.dumps({'research_allowed':True,'license_id':'CC-BY-4.0','source_url':'https://figshare.com/articles/dataset/21431547/1','license_url':'https://creativecommons.org/licenses/by/4.0/'}),encoding='utf-8')
    identifier=str(uuid.uuid4());raw=b'synthetic fixture bytes, not a personal image';expiry=datetime.now(timezone.utc)+timedelta(hours=1)
    row={'sample_id':identifier,'source_id':str(uuid.uuid4()),'source_hash':hashlib.sha256(raw).hexdigest(),'leakage_group_id':'synthetic-fixture-only',
         'path':str(release/'source'/(identifier+'.jpg')),'sha256':hashlib.sha256(raw).hexdigest(),'annotation_version':'synthetic-test','split':'test',
         'privacy_status':'restricted_research','qc_status':'publisher_annotations_machine_checked','release_scope':'offline_research','human_privacy_reviewed':False,
         'not_for_deployment':True,'license_evidence_path':str(evidence),'license_evidence_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest(),
         'allowed_use':'Synthetic contract fixture only','expires_at':expiry.isoformat()}
    return release,row,raw,expiry

def test_actual_file_materialization_precedes_real_research_validator(tmp_path):
    release,row,raw,expiry=local_row(tmp_path)
    assert not validate_research_release([row])['valid']
    materialize_and_validate([row],{row['sample_id']:raw},release,expiry)
    assert validate_research_release([row])['valid']
    assert (release/'source'/(row['sample_id']+'.jpg')).read_bytes()==raw
    assert not (release/'incomplete.json').exists()

def test_failed_real_validation_retains_incomplete_bytes_with_original_expiry_no_release(tmp_path):
    release,row,raw,expiry=local_row(tmp_path);row['license_evidence_sha256']='0'*64
    with pytest.raises(ValueError,match='research_release_validation_failed'):materialize_and_validate([row],{row['sample_id']:raw},release,expiry)
    receipt=json.loads((release/'incomplete.json').read_text(encoding='utf-8'))
    assert receipt['expires_at']==expiry.isoformat() and receipt['release_published'] is False
    assert (release/'source'/(row['sample_id']+'.jpg')).read_bytes()==raw
    assert not (release/'release.json').exists()
    with pytest.raises(ValueError,match='new_US100'):materialize_and_validate([row],{row['sample_id']:raw},release,expiry)
