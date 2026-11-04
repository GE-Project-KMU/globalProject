"""Synthetic small-frame contracts only; no trained models/GPU/source photographs."""
import copy,json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import pytest
from scripts import benchmark_final_cpu_pipeline as b

def future():return (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()

class Net:
    def __init__(self,value=.1):self.value=value;self.runtime_options={'fixture':True}
    def setInput(self,blob):self.blob=blob
    def forward(self):return np.asarray([self.value],np.float32)

class Model:
    backend='synthetic-unit-only';backend_version='fixture'
    def __init__(self,task,value=.1,post_delta=False):self.task=task;self.net=Net(value);self.closed=False;self.calls=0;self.post_delta=post_delta
    def predict(self,frame):
        assert not self.closed;self.calls+=1
        self.net.setInput(frame[0:1,0:1,:].astype(np.float32).reshape(1,3,1,1));self.net.forward()
        if self.task=='lane':
            mask=np.zeros(frame.shape[:2],np.uint8)
            if self.post_delta:mask[0,0]=1
            return mask
        return [{'class':'pothole' if self.task=='pothole' else 'face','confidence':.9,'bbox_xywh_px':[2.,2.,4.,4.]}]
    def close(self):self.closed=True

@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(b,'FRAMES',5);monkeypatch.setattr(b,'WIDTH',32);monkeypatch.setattr(b,'HEIGHT',32)
    return b.generated_frames()


def test_production_scope_fixed_and_no_GPU_profile():
    assert b.FRAMES==16 and (b.WIDTH,b.HEIGHT)==(1280,720)
    assert b.PLAN_SHA=='fda53437ca10b9fa7e0f87eb2b8160adcbde076c39c0fa2c80aa52104a5f443c'


def test_generated_inputs_are_deterministic_and_nonpersonal(tiny):
    assert all(np.array_equal(a,c) for a,c in zip(tiny,b.generated_frames()))
    assert all(f.dtype==np.uint8 for f in tiny)


def test_actual_pipeline_ABBA_with_fake_models_keeps_PNG_JSON_tracking_and_cleanup(tmp_path,tiny,monkeypatch):
    monkeypatch.setattr(b.pipeline,'frame_quality',lambda frame:{'assessable':False,'flags':{'synthetic_bad_quality':True}})
    output=tmp_path/'run';output.mkdir();created=[];profiles=[]
    def loader(parents,profile,**options):
        assert options=={'inference_backend':'cpu'}
        assert all(m.closed for group in created for m in group.values())
        models={task:Model(task) for task in b.g.TASKS};created.append(models.copy());profiles.append(profile);return models
    result=b.execute(tiny,output,{k:'unused-model.json' for k in b.g.TASKS},future(),loader=loader,require_real=False)
    assert profiles==['default','fast','fast','default']
    assert result['PNG_quality_tracking_lane_exact'] and result['raw_arrays_written']==0
    assert all(m.closed for group in created for m in group.values())
    assert all(r['warmup_calls']==5 and r['frames']==5 and r['io_profile']=='overlap' for r in result['runs'])
    assert all(r['processing_seconds']>0 and r['model_load_seconds']>=0 and r['teardown_seconds']>=0 for r in result['runs'])
    assert all((output/f'run-{i}'/'analysis.json').exists() for i in range(4))
    assert not list(output.rglob('*.npy')) and not list(output.rglob('*.npz'))
    assert result['raw_post_proof'][1]['comparison']=='fast_vs_default'
    assert all(r['actual_timed_model_predict_calls']=={k:5 for k in b.g.TASKS} for r in result['runs'])
    assert all(m.calls==(15 if index<2 else 10) for index,group in enumerate(created) for m in group.values())
    assert all(not row['quality']['assessable'] for row in json.loads((output/'run-0'/'analysis.json').read_text(encoding='utf-8'))['observations'])

@pytest.mark.parametrize('kind',['raw','post'])
def test_profile_mismatch_fails_closed_and_closes_every_session(tmp_path,tiny,kind):
    output=tmp_path/'run';output.mkdir();all_models=[]
    def loader(parents,profile,**options):
        models={task:Model(task,value=.2 if kind=='raw' and profile=='fast' else .1,post_delta=kind=='post' and profile=='fast') for task in b.g.TASKS};all_models.extend(models.values());return models
    with pytest.raises(ValueError,match='raw_tolerance|postprocess_exact'):
        b.execute(tiny,output,{k:'unused' for k in b.g.TASKS},future(),loader=loader,require_real=False)
    assert all(m.closed for m in all_models)
    assert not (output/'run-1').exists()


def test_event_ID_normalization_preserves_tracking_relationships():
    def result(ids):return {'observations':[{'frame_id':i,'pts_seconds':i/12,'quality':{},'lane':{},'potholes':[{'event_id':v,'confidence':.9}]} for i,v in enumerate(ids)]}
    assert b.normalize(result(['a','a','b']))==b.normalize(result(['x','x','y']))
    assert b.normalize(result(['a','a','b']))!=b.normalize(result(['x','y','y']))


def test_no_slot_authorization_reads_nothing(monkeypatch):
    monkeypatch.setattr(b,'policy',lambda:pytest.fail('no admission without authorization'))
    with pytest.raises(ValueError,match='isolated_CPU_slot'):b.run('unused','unused',False)


def test_wrong_final_plan_rejected_before_artifacts_or_models(monkeypatch):
    monkeypatch.setattr(b,'sha256',lambda *a:'0'*64);monkeypatch.setattr(b.g,'read',lambda *a:pytest.fail('wrong frozen plan'))
    with pytest.raises(ValueError,match='exact_final_modelset'):b.models_and_cap()


def test_expired_source_rejected_before_loader(tmp_path,tiny):
    with pytest.raises(ValueError):b.execute(tiny,tmp_path,{},'2000-01-01T00:00:00Z',loader=lambda *a,**k:pytest.fail('expired source'),require_real=False)


def test_cleanup_failure_does_not_replace_primary_exception():
    class Broken:
        def close(self):raise OSError('cleanup')
    primary=ValueError('processing')
    b.close_owned({'x':Broken()},set(),primary)
    assert str(primary)=='processing' and primary.__notes__


def test_metadata_prepare_no_models_or_frames(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'BASE',tmp_path);monkeypatch.setattr(b,'PUBLIC',tmp_path/'public.json')
    monkeypatch.setattr(b,'policy',lambda:{'scope':'synthetic-only'})
    monkeypatch.setattr(b,'generated_frames',lambda:pytest.fail('metadata only'))
    monkeypatch.setattr(b,'create_models',lambda *a:pytest.fail('metadata only'))
    execution=tmp_path/'execution.json';output=tmp_path/'out'
    b.prepare(execution,output)
    assert execution.exists() and not output.exists()
    with pytest.raises(ValueError):b.prepare(execution,output)
