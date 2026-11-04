import copy,json,os,subprocess,sys,time
import numpy as np
import pytest
from scripts import compare_privacy_family_v10_parallel as tool
from scripts.compare_detector_models import score
_ACTUAL_ADMIT=tool.admit

@pytest.fixture(autouse=True)
def synthetic_family_pin(monkeypatch):
    monkeypatch.setattr(tool,'family_sha',lambda:'SYNTHETIC-V10-PROTOCOL')

def test_shared_attempt_exclusive_flush_fsync_blocks_both_execution_orders(tmp_path,monkeypatch):
    path=tmp_path/'fresh-attempt.json';sync=[];real=os.fsync
    monkeypatch.setattr(tool.os,'fsync',lambda fd:(sync.append(fd),real(fd))[1])
    tool.claim_attempt(path,{'status':'started','writer':'parallel'})
    assert len(sync)==1 and json.loads(path.read_text(encoding='utf-8'))['writer']=='parallel'
    with pytest.raises(FileExistsError):path.open('x')  # Original serial barrier.
    other=tmp_path/'regression-attempt.json'
    with other.open('x',encoding='utf-8') as f:json.dump({'writer':'serial'},f)
    with pytest.raises(FileExistsError):tool.claim_attempt(other,{'writer':'parallel'})
    assert json.loads(other.read_text(encoding='utf-8'))['writer']=='serial'

def test_frame_prediction_order_and_partial_knownclass_ground_truth_remain_exact():
    found=[{'class':'face','confidence':.5,'bbox_xywh_px':[100,100,10,10]},
           {'class':'face','confidence':.5,'bbox_xywh_px':[0,0,10,10]},
           {'class':'license_plate','confidence':.9,'bbox_xywh_px':[0,0,10,10]}]
    selected=tool.ordered_frame(found,'s1',{'face'})
    assert [p['bbox_xywh_px'] for p in selected]==[[100,100,10,10],[0,0,10,10]]
    assert found[0].get('sample_id') is None
    rows=[{'sample_id':'s1','known_label_coverage':['face'],'objects':[{'class':'face','bbox_xywh_px':[0,0,10,10]},{'class':'license_plate','bbox_xywh_px':[1,1,2,2]}]}]
    assert tool.truth_rows(rows,['face','license_plate'])==[{'sample_id':'s1','class':'face','bbox_xywh_px':[0,0,10,10]}]
    rows[0]['ignore_bboxes_xywh_px']=[[0,0,1,1]]
    with pytest.raises(ValueError,match='ignore'):tool.truth_rows(rows,['face','license_plate'])

def payload_fixture():
    metadata={'manifest_sha256':'a'*64,'weights_sha256':'b'*64,'operating_confidence':.35,
              'architecture':'synthetic math contract only','input_size':[1920,1920],'backend':'onnxruntime-cpu','backend_version':'1.31.1'}
    execution={'execution_models':{'baseline':metadata},'benchmark_sha256':'c'*64}
    value={'synthetic_contract':False,'name':'baseline','execution_plan_sha256':'d'*64,'model_manifest_sha256':'a'*64,'weights_sha256':'b'*64,
           'benchmark_sha256':'c'*64,'sample_order':['s1','s2'],'prediction_counts':[2,0],
           'predictions':[{'sample_id':'s1','class':'face','confidence':.5,'bbox_xywh_px':[10,10,2,2]},
                          {'sample_id':'s1','class':'face','confidence':.5,'bbox_xywh_px':[20,20,2,2]}],
           'timings':[.1,.2],'images':2,'backend':'onnxruntime-cpu','version':'1.31.1','providers':['CPUExecutionProvider'],'intra_threads':2,'inter_threads':1,'CV_threads':1}
    return execution,value

def test_complete_worker_binding_and_order_contract():
    execution,value=payload_fixture();tool.validate_worker_output(value,'baseline',execution,'d'*64,['s1','s2'])

@pytest.mark.parametrize('change',['synthetic','model','plan','benchmark','version','provider','threads','CV','order','missing_row','count','cross_frame','timing'])
def test_worker_mismatch_or_partial_shard_cannot_publish(change):
    execution,value=payload_fixture()
    if change=='synthetic':value['synthetic_contract']=True
    elif change=='model':value['model_manifest_sha256']='0'*64
    elif change=='plan':value['execution_plan_sha256']='0'*64
    elif change=='benchmark':value['benchmark_sha256']='0'*64
    elif change=='version':value['version']='1.20.1'
    elif change=='provider':value['providers']=['CUDAExecutionProvider']
    elif change=='threads':value['intra_threads']=4
    elif change=='CV':value['CV_threads']=2
    elif change=='order':value['sample_order'].reverse()
    elif change=='missing_row':value['images']=1
    elif change=='count':value['prediction_counts']=[1,0]
    elif change=='cross_frame':value['predictions'][1]['sample_id']='s2'
    else:value['timings'][0]=float('nan')
    with pytest.raises(ValueError):tool.validate_worker_output(value,'baseline',execution,'d'*64,['s1','s2'])

def test_no_authorization_or_failed_numeric_admission_never_claims_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(tool,'admit',lambda role:pytest.fail('No admission without explicit authorization'))
    with pytest.raises(ValueError,match='authorization'):tool.run('fresh',False)
    monkeypatch.setattr(tool,'check_protocol',lambda:{'parent_benchmarks':{'fresh':{'sha256':'b'}}})
    monkeypatch.setattr(tool.family,'DEST',tmp_path)
    (tmp_path/'winner-freeze.json').write_text(json.dumps({'protocol_sha256':'SYNTHETIC-V10-PROTOCOL'}),encoding='utf-8')
    parent={'family_freeze_sha256':tool.sha256(tmp_path/'winner-freeze.json'),'benchmark_sha256':'b'}
    (tmp_path/'fresh-plan.json').write_text(json.dumps(parent),encoding='utf-8')
    monkeypatch.setattr(tool.family,'protocol',lambda:{'benchmarks':{'fresh':{'sha256':'b'}}})
    monkeypatch.setattr(tool.family,'comparison_plan',lambda *args:parent)
    def fail(_):raise ValueError('strict_plate_failed')
    monkeypatch.setattr(tool.family,'numeric_gates',fail)
    # Restore actual admission rather than the authorization-only spy.
    from importlib import reload
    actual=_ACTUAL_ADMIT
    # reload changes only this new module, not the frozen family module.
    monkeypatch.setattr(tool,'check_protocol',lambda:{'parent_benchmarks':{'fresh':{'sha256':'b'}}})
    with pytest.raises(ValueError,match='strict_plate'):actual('fresh')
    assert not (tmp_path/'fresh-attempt.json').exists()

def test_only_two_models_are_launched_and_failure_stops_owned_peer(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='two'):tool.launch({'baseline':[sys.executable,'-c','pass']},tmp_path,2)
    real=tool.subprocess.Popen;owned=[]
    def capture(*args,**kwargs):
        process=real(*args,**kwargs);owned.append(process);return process
    monkeypatch.setattr(tool.subprocess,'Popen',capture)
    start=time.monotonic()
    with pytest.raises(ValueError,match='worker_failed'):
        tool.launch({'baseline':[sys.executable,'-c','import time; time.sleep(30)'],
                     'candidate':[sys.executable,'-c','raise SystemExit(7)']},tmp_path,5)
    assert len(owned)==2 and all(process.poll() is not None for process in owned)
    assert time.monotonic()-start<15

def test_two_real_synthetic_processes_preserve_exact_serial_score_assembly(tmp_path):
    classes=['face','license_plate'];truth=[{'sample_id':'s1','class':'face','bbox_xywh_px':[0,0,10,10]},
        {'sample_id':'s2','class':'face','bbox_xywh_px':[0,0,10,10]}, {'sample_id':'s3','class':'license_plate','bbox_xywh_px':[0,0,20,10]}]
    preds={'baseline':[{'sample_id':'s1','class':'face','bbox_xywh_px':[100,100,10,10],'confidence':.5},
                       {'sample_id':'s1','class':'face','bbox_xywh_px':[0,0,10,10],'confidence':.5},
                       {'sample_id':'s2','class':'face','bbox_xywh_px':[0,0,10,10],'confidence':.5},
                       {'sample_id':'s3','class':'license_plate','bbox_xywh_px':[0,0,20,10],'confidence':.9}],
           'candidate':[dict(item,confidence=.6) for item in truth]}
    times={'baseline':[.1,.2,.3],'candidate':[.2,.3,.4]};metadata={name:{'manifest_sha256':name,'weights_sha256':name,'operating_confidence':.35,'architecture':'synthetic scheduler contract','input_size':[1920,1920],'backend':'onnxruntime-cpu','backend_version':'1.31.1'} for name in preds}
    metadata['candidate']['operating_confidence']=.65
    commands={}
    for name in preds:
        source=tmp_path/(name+'-fixture.json');source.write_text(json.dumps({'synthetic_contract':True,'predictions':preds[name],'timings':times[name]}),encoding='utf-8')
        commands[name]=[sys.executable,str(tool.ROOT/'scripts/compare_privacy_family_v10_parallel.py'),'contract-worker','--plan',str(source),'--output',str(tmp_path/(name+'.json'))]
    tool.launch(commands,tmp_path,30)
    returned={name:json.loads((tmp_path/(name+'.json')).read_text(encoding='utf-8')) for name in preds}
    actual=tool.assemble_results(truth,{n:r['predictions'] for n,r in returned.items()},classes,metadata,{n:r['timings'] for n,r in returned.items()})
    expected={name:{**metadata[name],'ap_metrics':score(truth,found,classes,.01),'operating_metrics':score(truth,found,classes,metadata[name]['operating_confidence']),
        'fixed_035_reference':score(truth,found,classes,.35),'latency_seconds':{'median':float(np.median(times[name])),'p95':float(np.percentile(times[name],95))}} for name,found in preds.items()}
    assert actual==expected
    assert actual['baseline']['ap_metrics']['classes']['face']['ap50']==pytest.approx(2/3)
    assert actual['candidate']['ap_metrics']['classes']['face']['ap50']==1
    assert actual['candidate']['operating_metrics']['classes']['face']['tp']==0
    assert actual['candidate']['fixed_035_reference']['classes']['face']['tp']==2
    assert all(returned[n]['predictions']==preds[n] and returned[n]['synthetic_contract'] is True for n in preds)

@pytest.mark.parametrize('changed',['parent','winner','protocol','benchmark','model','source','worker_source'])
def test_source_model_parent_and_benchmark_close_reject_any_drift(tmp_path,monkeypatch,changed):
    monkeypatch.setattr(tool.family,'DEST',tmp_path);monkeypatch.setattr(tool,'PROTOCOL',tmp_path/'execution-protocol.json')
    paths={name:tmp_path/(name+'.json') for name in ['parent','winner','protocol','benchmark','model']}
    paths['winner']=tmp_path/'winner-freeze.json';paths['protocol']=tool.PROTOCOL
    for path in paths.values():path.write_text('{}',encoding='utf-8')
    snapshot={'parent_plan':tool.relative(paths['parent']),'parent_plan_sha256':tool.sha256(paths['parent']),
              'family_freeze_sha256':tool.sha256(paths['winner']),'parallel_protocol_sha256':tool.sha256(paths['protocol']),
              'source_sha256':tool.source_pins(),'worker_source_sha256':tool.sha256(tool.Path(tool.__file__)),
              'benchmark_manifest':tool.relative(paths['benchmark']),'benchmark_sha256':tool.sha256(paths['benchmark']),
              'model_closure':{'baseline':{'model':{'path':tool.relative(paths['model']),'sha256':tool.sha256(paths['model'])}}}}
    tool.assert_closure(snapshot)
    if changed=='source':monkeypatch.setattr(tool,'source_pins',lambda:{'changed':'0'*64})
    elif changed=='worker_source':snapshot['worker_source_sha256']='0'*64
    else:paths[changed].write_text('{"changed":true}',encoding='utf-8')
    with pytest.raises(ValueError):tool.assert_closure(snapshot)

def test_changed_original_comparison_plan_cannot_reach_numeric_or_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(tool.family,'DEST',tmp_path);(tmp_path/'winner-freeze.json').write_text(json.dumps({'protocol_sha256':'SYNTHETIC-V10-PROTOCOL'}),encoding='utf-8')
    (tmp_path/'fresh-plan.json').write_text(json.dumps({'models':{'candidate':'unapproved.json'}}),encoding='utf-8')
    monkeypatch.setattr(tool,'check_protocol',lambda:{})
    monkeypatch.setattr(tool.family,'protocol',lambda:{})
    monkeypatch.setattr(tool.family,'comparison_plan',lambda *args:{'models':{'candidate':'frozen.json'}})
    monkeypatch.setattr(tool.family,'numeric_gates',lambda _:pytest.fail('Changed plan must fail first'))
    with pytest.raises(ValueError,match='exact_original'):tool.admit('fresh')
    assert not (tmp_path/'fresh-attempt.json').exists()

@pytest.mark.parametrize('change',['parent','failed_attempt','other_role','attempt_SHA'])
def test_direct_or_stale_worker_replay_cannot_bypass_one_shot_parent(tmp_path,monkeypatch,change):
    monkeypatch.setattr(tool.family,'DEST',tmp_path);monkeypatch.setattr(tool.os,'getppid',lambda:123)
    attempt=tmp_path/'fresh-attempt.json';value={'status':'started','writer':'two_process_CPU','plan_sha256':'a','parallel_protocol_sha256':'b','selection_allowed':False,'automatic_retry_allowed':False}
    attempt.write_text(json.dumps(value),encoding='utf-8')
    snapshot={'parent_pid':123,'comparison_role':'fresh','attempt_path':str(attempt),'attempt_sha256':tool.sha256(attempt),'parent_plan_sha256':'a','parallel_protocol_sha256':'b'}
    tool.assert_worker_attempt(snapshot)
    if change=='parent':snapshot['parent_pid']=124
    elif change=='other_role':snapshot['comparison_role']='regression'
    elif change=='attempt_SHA':snapshot['attempt_sha256']='0'*64
    else:
        value['status']='failed';attempt.write_text(json.dumps(value),encoding='utf-8');snapshot['attempt_sha256']=tool.sha256(attempt)
    with pytest.raises(ValueError):tool.assert_worker_attempt(snapshot)

def test_child_failure_preserves_shared_failed_attempt_and_prevents_auto_retry(tmp_path,monkeypatch):
    import agent.dataset as dataset
    monkeypatch.setattr(tool.family,'DEST',tmp_path);monkeypatch.setattr(tool,'PROTOCOL',tmp_path/'protocol.json')
    tool.PROTOCOL.write_text('{}',encoding='utf-8');(tmp_path/'winner-freeze.json').write_text('{}',encoding='utf-8')
    benchmark=tmp_path/'synthetic-metadata.json';benchmark.write_text(json.dumps([{'split':'test','sample_id':'s1','sha256':'a'*64,'leakage_group_id':'synthetic',
        'expires_at':'2099-01-01T00:00:00+00:00','known_label_coverage':['face'],'objects':[]}]),encoding='utf-8')
    plan={'benchmark_manifest':str(benchmark),'benchmark_sha256':tool.sha256(benchmark),'expected_samples':1,
          'models':{'baseline':'synthetic-base.json','candidate':'synthetic-candidate.json'}}
    parent=tmp_path/'fresh-plan.json';parent.write_text(json.dumps(plan),encoding='utf-8')
    execution={'worker_source_sha256':tool.sha256(tool.Path(tool.__file__)),'source_sha256':{},'worker_timeout_seconds':1}
    monkeypatch.setattr(tool,'admit',lambda _: (execution,{}, {},parent,plan));monkeypatch.setattr(tool,'check_protocol',lambda:execution)
    monkeypatch.setattr(dataset,'validate_research_release',lambda _: {'valid':True})
    monkeypatch.setattr(tool,'model_bindings',lambda *_: ({'baseline':{},'candidate':{}},{},['face','license_plate']))
    monkeypatch.setattr(tool,'assert_closure',lambda _:None)
    def fail(*_):raise ValueError('controlled_child_failure_no_model_or_pixel_execution')
    monkeypatch.setattr(tool,'launch',fail)
    with pytest.raises(ValueError,match='controlled_child'):tool.run('fresh',True)
    marker=json.loads((tmp_path/'fresh-attempt.json').read_text(encoding='utf-8'))
    assert marker['status']=='failed' and marker['writer']=='two_process_CPU' and marker['automatic_retry_allowed'] is False
    with pytest.raises(FileExistsError):tool.run('fresh',True)
    assert json.loads((tmp_path/'fresh-attempt.json').read_text(encoding='utf-8'))==marker
