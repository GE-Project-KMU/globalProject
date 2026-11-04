import pytest
from scripts.cycle2_public_data import cross_split_clean
from scripts.nextcycle_val_audit import overlap,face_size,match_groups

def row(name,split,h,v):return {'id':name,'split':split,'h':h,'v':f'{v:016x}'}

def test_new_cross_split_exact_removes_all_members():
 rows=[row('a','train','same',0),row('b','test','same',65535),row('c','train','other',2**63)]
 assert [r['id'] for r in cross_split_clean(rows,'h','v',0)]==['c']

def test_new_cross_split_transitive_chain_removes_all_members():
 rows=[row('a','train','a',0),row('b','train','b',1),row('c','test','c',3),row('safe','validation','d',2**64-1)]
 assert [r['id'] for r in cross_split_clean(rows,'h','v',1)]==['safe']

def test_same_split_near_members_retained():
 rows=[row('a','train','a',0),row('b','train','b',1)]
 assert cross_split_clean(rows,'h','v',1)==rows

def test_validation_test_cross_match_is_excluded():
 assert cross_split_clean([row('a','validation','x',4),row('b','test','x',4)],'h','v',0)==[]

def test_iou_matching_boundary_exact_half():
 assert overlap([0,0,10,10],[0,0,20,10])==.5
 assert overlap([0,0,10,10],[10,0,10,10])==0
 assert overlap([0,0,0,0],[0,0,0,0])==0

@pytest.mark.parametrize('size,expected',[(15.99,'<16'),(16,'16..31'),(32,'32..63'),(64,'>=64')])
def test_face_original_area_bins(size,expected):assert face_size([0,0,size,size])==expected

def test_greedy_duplicate_does_not_cover_a_second_gt():
 truth=[{'sample_id':'s','class':'face','bbox_xywh_px':[0,0,10,10]},{'sample_id':'s','class':'face','bbox_xywh_px':[30,0,10,10]}]
 preds=[dict(truth[0],confidence=.9),dict(truth[0],confidence=.8)]
 result,detail=match_groups(truth,preds,{'s':{}})
 assert result['<16']=={'GT':2,'TP':1,'FN':1,'recall':.5}
 assert detail==[{'sample_id':'s','GT':2,'FN':1}]
from scripts.train_privacy_p2_cycle2 import checkpoint_is_better,training_image_weight
from scripts import train_lane_cycle2 as lane

@pytest.mark.parametrize('metric,passes,best,overall,expected',[(.8,False,-1,.7,True),(.8,False,.7,.7,False),(.6,True,-1,.9,True),(.8,True,.8,.8,False),(.9,True,.8,.8,True),(.5,True,.8,.8,False)])
def test_large_face_gate_selection_and_earliest_tie(metric,passes,best,overall,expected):
 assert checkpoint_is_better(metric,passes,best,overall) is expected

def test_sampling_changes_image_probability_without_dropping_GT():
 row={'image':{'width':1920,'height':1920},'objects':[{'class':'face','bbox_xywh_px':[0,0,10,10]},{'class':'face','bbox_xywh_px':[0,0,100,100]}]}
 assert training_image_weight(row)==2.
 assert len(row['objects'])==2
 row['objects']=row['objects'][1:]
 assert training_image_weight(row)==1.

def test_no_face_row_cannot_enter_weighted_sampler():
 with pytest.raises(ValueError,match='face_training_labels_required'):training_image_weight({'image':{'width':10,'height':10},'objects':[]})

def test_lane_validation_selection_is_recording_macro_not_pixel_weighted(monkeypatch):
 rows=[{'recording':'a'}]*100+[{'recording':'b'}]*2
 def score(model,items,size,torch):
  ious=[.9,.6] if len(items)==100 else [.1,.6]
  return None,[{'paint_threshold':t,'paint_iou':i,'confusion':{'tp':int(i*1000),'fp':int((1-i)*1000),'fn':0,'tn':0,'ignored':0}} for t,i in zip([.2,.5],ious)]
 monkeypatch.setattr(lane,'original_validation_scores',score)
 selected,scores=lane.validation_scores(None,rows,1024,None)
 assert selected['paint_threshold']==.5
 assert selected['paint_iou']==.6
 assert scores[0]['paint_iou']==.5
 assert selected['recording_IoUs']==[.6,.6]
from scripts.probe_cycle2_gpu_memory import memory_allowed,interval_summary,LIMIT

@pytest.mark.parametrize('used,before,after,ok',[(LIMIT,0,0,True),(LIMIT+1,0,0,False),(100,0,1,False),(-1,0,0,False),(True,0,0,False)])
def test_shared_global_memory_budget_and_even_hidden_OOM(used,before,after,ok):assert memory_allowed(used,before,after) is ok

def test_real_memory_sampling_period_not_assumed_sleep_duration():
 result=interval_summary([{'elapsed_ns':0,'used_bytes':1},{'elapsed_ns':25_000_000,'used_bytes':2},{'elapsed_ns':100_000_000,'used_bytes':3}])
 assert result['observed_interval_ms']=={'min':25.,'median':50.,'max':75.}
 assert result['requested_sleep_ms']==20.
 assert result['sleep_not_guaranteed_sampling_period'] is True
 assert result['max_observed_global_bytes']==3

def test_sampling_clock_must_be_monotonic():
 with pytest.raises(ValueError):interval_summary([{'elapsed_ns':10,'used_bytes':1},{'elapsed_ns':0,'used_bytes':2}])
import json,uuid,hashlib,tempfile
from datetime import datetime,timezone,timedelta
from pathlib import Path
from scripts.probe_cycle2_gpu_memory import check_probe_release

@pytest.fixture
def synthetic_research_row():
 root=Path(__file__).resolve().parents[1];folder=root/'.cache/cycle2-release-policy-tests';folder.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=folder) as temp:
  temp=Path(temp);image=temp/'synthetic-bytes.bin';image.write_bytes(b'synthetic checksum fixture; not a source photo')
  evidence=temp/'synthetic-license.json';evidence.write_text(json.dumps({'research_allowed':True,'license_id':'synthetic-test-only','source_url':'https://example.invalid/test','license_url':'https://example.invalid/license'}))
  hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  now=datetime(2026,11,5,tzinfo=timezone.utc)
  row={'sample_id':str(uuid.uuid4()),'source_id':'synthetic','source_hash':hashfile(image),'leakage_group_id':'synthetic-only','path':image.relative_to(root).as_posix(),'sha256':hashfile(image),'annotation_version':'synthetic-test-only','split':'train','privacy_status':'restricted_research','qc_status':'publisher_annotations_machine_checked','release_scope':'offline_research','human_privacy_reviewed':False,'not_for_deployment':True,'license_evidence_path':evidence.relative_to(root).as_posix(),'license_evidence_sha256':hashfile(evidence),'allowed_use':'synthetic policy unit test','expires_at':(now+timedelta(days=1)).isoformat()}
  yield row,now,image,evidence,hashfile

def test_probe_rejects_expired_metadata_before_cuda_or_decode(synthetic_research_row):
 row,now,*_=synthetic_research_row;row['expires_at']=now.isoformat()
 with pytest.raises(ValueError,match='before_GPU_or_pixel_decode'):check_probe_release([row],now=now)

def test_probe_current_valid_research_metadata_allowed(synthetic_research_row):
 row,now,*_=synthetic_research_row
 assert check_probe_release([row],now=now)['valid'] is True

def test_probe_denied_license_cannot_initialize_processing(synthetic_research_row):
 row,now,image,evidence,hashfile=synthetic_research_row;policy=json.loads(evidence.read_text());policy['research_allowed']=False;evidence.write_text(json.dumps(policy));row['license_evidence_sha256']=hashfile(evidence)
 with pytest.raises(ValueError):check_probe_release([row],now=now)

def test_probe_corrupt_source_checksum_is_rejected(synthetic_research_row):
 row,now,image,evidence,hashfile=synthetic_research_row;image.write_bytes(b'changed')
 with pytest.raises(ValueError):check_probe_release([row],now=now)

def test_probe_fake_human_approval_cannot_replace_research_scope(synthetic_research_row):
 row,now,*_=synthetic_research_row;row['privacy_status']='approved'
 with pytest.raises(ValueError):check_probe_release([row],now=now)
