import copy,hashlib,json,random,uuid
from pathlib import Path
import numpy as np,pytest
from scripts import finalize_privacy_family_v10_results as m
from scripts.compare_detector_models import score
from scripts.compare_privacy_parallel import assemble_results

@pytest.mark.parametrize('seed',range(10))
def test_indexed_AP_exact_oracle_stable_ties_duplicate_GT_and_all_thresholds(seed):
 rng=random.Random(seed);truth=[];predictions=[]
 for sid in ['a','b','c']:
  for label in m.CLASSES:
   for i in range(4):truth.append({'sample_id':sid,'class':label,'bbox_xywh_px':[i%2*10,0,10,10]})
 for i in range(70):
  predictions.append({'sample_id':rng.choice(['a','b','c','unknown']),'class':rng.choice(m.CLASSES),'confidence':rng.choice([.01,.03,.1,.35,.5,.9]),'bbox_xywh_px':[rng.choice([0,3,10,50]),0,10,10]})
 confidences=[.01,.03,.05,.1,.2,.35,.5,1.]
 found=m.indexed_scores(truth,predictions,confidences)
 for c in confidences:assert m.same(found[c],score(truth,predictions,m.CLASSES,c))

def bundle(role='fresh'):
 n=m.COUNTS[role];order=[str(uuid.UUID(int=i+1)) for i in range(n)];truth=[{'sample_id':order[0],'class':'face','bbox_xywh_px':[0,0,10,10]},{'sample_id':order[1],'class':'license_plate','bbox_xywh_px':[0,0,10,10]}]
 # A tied face FP BEFORE TP is deliberate AP/order coverage.
 found=[{'sample_id':order[0],'class':'face','bbox_xywh_px':[50,0,10,10],'confidence':.8},{**truth[0],'confidence':.8},{**truth[1],'confidence':.6}]
 plan={'benchmark_sha256':'b'*64,'role':'Synthetic contract only','expected_samples':n}
 execution={**plan,'parent_plan_sha256':'p'*64,'comparison_role':role,'parallel_protocol_sha256':'c'*64,'execution_models':{}}
 workers={};metadata={};pred={};timings={}
 for name in ['baseline','candidate']:
  metadata[name]={'manifest_sha256':('a' if name=='baseline' else 'd')*64,'weights_sha256':'e'*64,'operating_confidence':.35 if name=='baseline' else .9,'architecture':'two YOLOX-Nano specialists global plus four tiles','input_size':[1920,1920],'backend':'onnxruntime-cpu','backend_version':'1.31.1'}
  predictions=copy.deepcopy(found);counts=[2,1]+[0]*(n-2);times=[.1+i*.00001 for i in range(n)]
  workers[name]={'synthetic_contract':False,'name':name,'execution_plan_sha256':'x'*64,'model_manifest_sha256':metadata[name]['manifest_sha256'],'weights_sha256':'e'*64,'benchmark_sha256':'b'*64,'sample_order':order,'prediction_counts':counts,'predictions':predictions,'timings':times,'images':n,'backend':'onnxruntime-cpu','version':'1.31.1','providers':['CPUExecutionProvider'],'intra_threads':2,'inter_threads':1,'CV_threads':1}
  pred[name]=predictions;timings[name]=times
 execution['execution_models']=metadata;result=assemble_results(truth,pred,m.CLASSES,metadata,timings)
 report={'samples':n,'plan_sha256':'p'*64,'benchmark_sha256':'b'*64,'scope':plan['role'],'results':result,'test_used_for_selection':False,'known_class_conditional':True,'independent_physical_identity_or_route_verified':False,'project_acceptance_passed':False,'map50_change':result['candidate']['ap_metrics']['map50']-result['baseline']['ap_metrics']['map50']}
 attempt={'status':'completed','writer':'two_process_CPU','samples':n,'selection_allowed':False,'plan_sha256':'p'*64,'parallel_protocol_sha256':'c'*64}
 receipt={'status':'completed','parent_plan_sha256':'p'*64,'execution_plan_sha256':'x'*64,'parallel_protocol_sha256':'c'*64,'model_processes':2,'GPU_operations':0,'model_or_threshold_selection_allowed':False,'timing_is_shared_parallel_context':True,'wall_seconds':1.}
 return [role,plan,attempt,execution,receipt,workers,{'truth':truth,'predictions':pred},report,'p'*64,'x'*64]

@pytest.mark.parametrize('role',['fresh','regression'])
def test_completed_pair_whitelist_copies_original_values_no_reselection(role):
 args=bundle(role);result=m.validate_role(*args)
 assert result['images']==m.COUNTS[role] and result['candidate_both_class_recall95_goal_met'] is False
 assert result['models']['baseline']['classes']['face']['AP50']==.5
 assert result['models']['candidate']['classes']['face']['recall']==0.
 assert result['models']['candidate']['operating_confidence']==.9
 assert 'predictions' not in str(result) and 'sample_id' not in str(result)
 for name in ['baseline','candidate']:
  for label in m.CLASSES:
   assert result['models'][name]['classes'][label]['AP50']==args[7]['results'][name]['ap_metrics']['classes'][label]['ap50']

@pytest.mark.parametrize('mutation',['failed_attempt','incomplete_worker','runtime','pred_order','GT_change','metric_change','count_bool','selection','latency','execution_plan','execution_receipt','worker_SHA','benchmark_SHA','nonfinite_time'])
def test_each_binding_or_failure_refuses_success(mutation):
 a=bundle();plan,attempt,execution,receipt,workers,per_image,report=a[1:8]
 if mutation=='failed_attempt':attempt['status']='failed'
 elif mutation=='incomplete_worker':workers['candidate']['images']-=1
 elif mutation=='runtime':workers['candidate']['version']='1.20.2'
 elif mutation=='pred_order':workers['candidate']['predictions']=list(reversed(workers['candidate']['predictions']))
 elif mutation=='GT_change':per_image['truth'][0]['bbox_xywh_px'][0]=100
 elif mutation=='metric_change':report['results']['candidate']['operating_metrics']['classes']['face']['recall']=.1
 elif mutation=='count_bool':report['results']['baseline']['ap_metrics']['classes']['face']['tp']=True
 elif mutation=='selection':report['test_used_for_selection']=True
 elif mutation=='latency':report['results']['candidate']['latency_seconds']['median']=42.
 elif mutation=='execution_plan':execution['expected_samples']=1
 elif mutation=='execution_receipt':receipt['status']='failed'
 elif mutation=='worker_SHA':workers['baseline']['weights_sha256']='z'*64
 elif mutation=='benchmark_SHA':report['benchmark_sha256']='z'*64
 else:workers['candidate']['timings'][0]=float('nan')
 with pytest.raises((ValueError,KeyError)):m.validate_role(*a)

def queue():
 q={'status':'completed','operator_source_sha256':'a'*64,'CPU_execution_runtime':copy.deepcopy(m.RUNTIME),'automatic_retry_allowed':False,'user_recordings_used':False}
 phases={name:{'phase':name,'status':'completed','child_exit_code':0,'own_child_running_after_cleanup':False,'CPU_execution_runtime':copy.deepcopy(m.RUNTIME),'automatic_retry_allowed':False,'root_authorized':True} for name in m.PHASES}
 for key in ['baseline-VAL508','fixed12-VAL508']:phases[key].update(observer_child_stopped=True,observer_thread_stopped=True,device_memory={'over_budget_observed':False})
 return q,phases

@pytest.mark.parametrize('change',['phase_failure','missing_phase','queue_failure','unclean_child','environment','GPU_observer'])
def test_active_failure_or_environment_change_cannot_be_hidden_by_completed_benchmark(change):
 q,p=queue()
 if change=='phase_failure':p['strict-winner-face']['status']='failed'
 elif change=='missing_phase':p.pop('fresh')
 elif change=='queue_failure':q['status']='failed'
 elif change=='unclean_child':p['regression']['own_child_running_after_cleanup']=True
 elif change=='environment':p['fresh']['CPU_execution_runtime']['OpenCV']='changed'
 else:p['baseline-VAL508']['device_memory']['over_budget_observed']=True
 with pytest.raises(ValueError):m.validate_queue(q,p,'a'*64)

def test_completed_active_retry_does_not_erase_or_reinterpret_historical_failure():
 q,p=queue();m.validate_queue(q,p,'a'*64)
 historical={'status':'failed_supervision'}
 assert historical['status']=='failed_supervision'

def report():
 value={'format':'privacy-family-v10-final-results-v1','selected_family':'legacy_v4','selected_confidence':.9,'cohorts':{r:m.validate_role(*bundle(r)) for r in m.COUNTS},'limits':['Synthetic fixture, no performance claim.'],'revision':copy.deepcopy(m.REVISION),'candidate_artifact_sha256':{name:{key:'a'*64 for key in ['manifest_sha256','checkpoint_sha256','weights_sha256','provenance_sha256']} for name in ['baseline','rescue_v8','fixed_final12']},
  'research_only':True,'production_approved':False,'human_privacy_reviewed':False,'physical_person_or_route_independence_verified':False,'TEST_used_for_selection':False,'runtime':copy.deepcopy(m.RUNTIME)}
 for k in ['family_protocol_sha256','family_source_sha256','parallel_protocol_sha256','parallel_worker_source_sha256','winner_freeze_sha256','publisher_source_sha256','operator_control_sha256']:value[k]='a'*64
 value['artifact_sha256']={name:{key:'a'*64 for key in ['manifest_sha256','weights_sha256','checkpoint_sha256','origin_manifest_sha256','provenance_sha256','dataset_release_sha256']} for name in ['baseline','winner']}
 value['canonical_default_strict_evidence_sha256']={name:{label:'a'*64 for label in m.CLASSES} for name in ['baseline','winner']}
 value['worker_result_sha256']={role:{name:'a'*64 for name in ['baseline','candidate']} for role in m.COUNTS}
 return value

def test_idempotent_JSON_and_MD_bytes_no_timestamp_and_existing_mismatch_refused(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path);folder=tmp_path/'docs/reports';folder.mkdir(parents=True);path=folder/'fixture.json';value=report()
 a=m.publish_pair(value,path);before=(path.read_bytes(),path.with_suffix('.md').read_bytes())
 b=m.publish_pair(value,path)
 assert a['json_sha256']==b['json_sha256'] and a['MD_sha256']==b['MD_sha256'] and b['status']=='unchanged'
 assert before==(path.read_bytes(),path.with_suffix('.md').read_bytes())
 value['selected_confidence']=.1
 with pytest.raises(ValueError,match='differ'):m.publish_pair(value,path)

@pytest.mark.parametrize('key,value',[('image_sha256','a'*64),('sample_id','private'),('source_member','private'),('bbox_xywh_px',[1,2,3,4]),('predictions',[]),('path','data/private.jpg'),('extra','C:\\private\\a.png'),('extra','raw.npy'),('extra','weights.safetensors')])
def test_public_privacy_guard_rejects_individual_records_and_artifacts(tmp_path,monkeypatch,key,value):
 monkeypatch.setattr(m,'ROOT',tmp_path);folder=tmp_path/'docs/reports';folder.mkdir(parents=True);r=report();r[key]=value
 with pytest.raises(ValueError):m.publish_pair(r,folder/'fixture.json')
 assert not (folder/'fixture.json').exists()

def test_control_reader_blocks_pixel_NPY_and_detects_digest_changes(tmp_path):
 reader=m.Controls();p=tmp_path/'control.json';p.write_text('{"status":"synthetic"}',encoding='utf-8');reader.read(p);p.write_text('{}',encoding='utf-8')
 with pytest.raises(ValueError,match='SHA_mismatch'):reader.close()
 for ext in ['.jpg','.png','.npy','.npz']:
  pixel=tmp_path/('forbidden'+ext);pixel.write_bytes(b'not an actual image')
  with pytest.raises(ValueError,match='forbidden'):m.Controls().read(pixel)

def test_control_JSON_duplicates_and_nonfinite_values_are_refused(tmp_path):
 for i,raw in enumerate(['{"a":1,"a":2}','{"x":NaN}','{"x":1e9999}']):
  p=tmp_path/f'bad{i}.json';p.write_text(raw,encoding='utf-8')
  with pytest.raises(ValueError):m.Controls().read(p)

def test_unknown_public_field_and_UUID_hidden_in_limits_are_rejected():
 r=report();r['arbitrary_debug']='not allowed'
 with pytest.raises(ValueError,match='whitelist'):m.public_whitelist(r)
 r=report();r['limits'].append(str(uuid.uuid4()))
 with pytest.raises(ValueError,match='UUID'):m.public_whitelist(r)

def test_partial_publication_never_recreates_or_overwrites_a_success_pair(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path);folder=tmp_path/'docs/reports';folder.mkdir(parents=True);path=folder/'fixture.json';path.write_bytes(b'preserved partial proof')
 with pytest.raises(ValueError,match='partial'):m.publish_pair(report(),path)
 assert path.read_bytes()==b'preserved partial proof' and not path.with_suffix('.md').exists()

def test_public_hash_maps_are_hashes_only_not_hidden_identity_channels():
 r=report();r['artifact_sha256']={'baseline':{'path':'private'},'winner':{}}
 with pytest.raises(ValueError,match='whitelist'):m.public_whitelist(r)
 r=report();r['worker_result_sha256']={role:{name:'a'*64 for name in ['baseline','candidate']} for role in m.COUNTS};m.public_whitelist(r)
 r['worker_result_sha256']['fresh']['candidate']='not an aggregate hash'
 with pytest.raises(ValueError,match='SHA'):m.public_whitelist(r)

def test_no_authorization_stops_before_any_completed_control_or_real_result_read(monkeypatch):
 called=[];monkeypatch.setattr(m.sys,'argv',['publisher','--operator-source-sha256','a'*64])
 monkeypatch.setattr(m,'gather',lambda:called.append(True))
 assert m.main()==1 and called==[]

def test_MD_IO_failure_cannot_leave_a_success_JSON(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp_path);folder=tmp_path/'docs/reports';folder.mkdir(parents=True);path=folder/'fixture.json'
 original=Path.open
 def fail_MD(self,*args,**kwargs):
  if self==path.with_suffix('.md'):raise OSError('synthetic write error')
  return original(self,*args,**kwargs)
 monkeypatch.setattr(Path,'open',fail_MD)
 with pytest.raises(OSError):m.publish_pair(report(),path)
 assert not path.exists()

@pytest.mark.parametrize('kind',['absolute','dotdot','prefix_sibling','other_drive','UNC','drive_relative'])
def test_outside_paths_are_rejected_before_any_resolve_stat_or_shared_resolver(monkeypatch,kind):
 root=m.ROOT
 values={'absolute':str(root.parent/'outside-control.json'),'dotdot':'../outside-control.json',
  'prefix_sibling':str(root.parent/(root.name+'-sibling')/'control.json'),'other_drive':'Z:/outside/control.json',
  'UNC':'\\\\unrelated-server\\share\\control.json','drive_relative':'Z:control.json'}
 calls=[]
 def trap(*args,**kwargs):calls.append(True);raise AssertionError('external FS/resolver access forbidden')
 with monkeypatch.context() as patch:
  patch.setattr(m,'workspace_path',trap);patch.setattr(Path,'resolve',trap);patch.setattr(Path,'stat',trap);patch.setattr(m.os,'stat',trap)
  with pytest.raises(ValueError,match='lexically_refused'):m.Controls().path(values[kind])
 assert calls==[]

def test_string_only_normalization_allows_internal_dot_segments_without_IO(monkeypatch):
 def trap(*args,**kwargs):raise AssertionError('string-only normalization must not access filesystem')
 with monkeypatch.context() as patch:
  patch.setattr(Path,'resolve',trap);patch.setattr(Path,'stat',trap)
  value=m.lexical_control_path('data/synthetic/../control.json')
 assert value==m.os.path.normcase(m.os.path.normpath(str(m.ROOT/'data/control.json')))

def test_publication_destination_is_also_lexically_rejected_before_resolve(monkeypatch):
 value=report();outside=str(m.ROOT.parent/'outside-publication.json');calls=[]
 def trap(*args,**kwargs):calls.append(True);raise AssertionError('external resolver or stat forbidden')
 with monkeypatch.context() as patch:
  patch.setattr(m,'workspace_path',trap);patch.setattr(Path,'resolve',trap);patch.setattr(Path,'stat',trap)
  with pytest.raises(ValueError,match='lexically_refused'):m.publish_pair(value,outside)
 assert calls==[]


def test_fixed12_selected_family_is_publicly_distinct_and_revision_cannot_be_erased():
 r=report();r['selected_family']='fixed_final12';m.public_whitelist(r)
 assert '모든 VAL 이전' in m.markdown(r).decode('utf-8')
 r['revision']['development_informed_revision']=False
 with pytest.raises(ValueError,match='revision'):m.public_whitelist(r)


def test_original_AP_and_final12_hashes_are_both_required_even_if_unselected():
 r=report();r['candidate_artifact_sha256'].pop('rescue_v8')
 with pytest.raises(ValueError,match='candidate'):m.public_whitelist(r)

def test_V9_reuses_exact_immutable_V8_score_and_control_helpers():
 from scripts import finalize_privacy_family_results as parent
 assert m.indexed_scores is parent.indexed_scores and m.Controls is parent.Controls
 assert m.validate_role is parent.validate_role and m.lexical_control_path is parent.lexical_control_path
 assert m.PARENT_PUBLISHER_SHA=='12d9a6cee2aa9a32176f9802ca7b5e08d44f4072f64f000dd0914ff918ba6a8b'


def test_V10_correction_keeps_same_EMA_and_preserves_failed_parent_flag():
 r=report();m.public_whitelist(r)
 assert r['revision']['same_original_final12_EMA'] is True and r['revision']['new_training_updates']==0
 assert r['revision']['old_failed_parent_preservation_flag'] is False
 r['revision']['old_failed_parent_preservation_flag']=True
 with pytest.raises(ValueError,match='revision'):m.public_whitelist(r)
