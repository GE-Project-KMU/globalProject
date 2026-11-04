"""A failed restart must never launch another GPU training process."""
import json,tempfile
from pathlib import Path
import pytest
from agent.paths import ROOT,sha256
from scripts import supervise_pothole_small_retry as supervisor

@pytest.fixture
def restart(monkeypatch):
    cache=ROOT/'.cache/restart-guard-tests';cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        root=Path(directory);parent=root/'data/training/pothole-small-v9';out=root/'data/training/pothole-small-v9-retry1';parent.mkdir(parents=True);out.mkdir()
        source=root/'scripts/train_pothole_small.py';source.parent.mkdir();source.write_text('# synthetic never executed')
        def save(path,value):path.write_text(json.dumps(value),encoding='utf-8')
        plan={'source_sha256':sha256(source),'dependencies':{},'source_expires_at':'2099-01-01T00:00:00Z'}
        save(parent/'plan.json',plan)
        for name,value in [('release.json',[]),('preflight.json',{}),('GPU-memory-preflight.json',{'status':'passed','plan_sha256':sha256(parent/'plan.json')})]:save(parent/name,value)
        for p in parent.iterdir():(out/p.name).write_bytes(p.read_bytes())
        stopped=root/'stopped.json';save(stopped,{'own_child_running_after_cleanup':False,'own_child_pid':999999999,'own_child_creation_time':0})
        retry={'supervisor_sha256':sha256(Path(supervisor.__file__)),'copied_input_sha256':{p.name:sha256(p) for p in parent.iterdir()},'stopped_parent_receipts':{'stopped.json':sha256(stopped)}};save(out/'retry-plan.json',retry)
        monkeypatch.setattr(supervisor,'ROOT',root)
        from scripts import train_pothole_small
        monkeypatch.setattr(train_pothole_small,'DEPS',[])
        calls=[]
        def forbidden(*a,**k):calls.append(1);raise AssertionError('must not launch child')
        monkeypatch.setattr(supervisor.subprocess,'Popen',forbidden)
        yield root,out,retry,save,calls
        assert calls==[]

def test_changed_learning_input_prevents_observer_and_training(restart):
    _,out,_,_,_=restart;(out/'release.json').write_text('[{"changed":true}]')
    with pytest.raises(ValueError,match='restart_learning_inputs_changed'):supervisor.main()

def test_changed_parent_stop_evidence_prevents_training(restart):
    root,_,_,_,_=restart;(root/'stopped.json').write_text('{}')
    with pytest.raises(ValueError,match='parent_stop_evidence_changed'):supervisor.main()

def test_still_running_parent_receipt_prevents_training(restart):
    root,out,retry,save,_=restart
    value=json.loads((root/'stopped.json').read_text());value['own_child_running_after_cleanup']=True;save(root/'stopped.json',value)
    retry['stopped_parent_receipts']['stopped.json']=sha256(root/'stopped.json');save(out/'retry-plan.json',retry)
    with pytest.raises(ValueError,match='prior_training_not_stopped'):supervisor.main()

def test_busy_device_fails_before_popen_and_closes_observer(restart,monkeypatch):
    _,out,_,_,_=restart;closed=[]
    class Busy:
        def __init__(self,*a,**k):pass
        def start(self):return self
        def sample_rows(self):return [{'used_bytes':3*2**30}]
        def close(self):closed.append(True)
        def summary(self):return {'synthetic':True}
    from scripts import cycle2_device_memory
    monkeypatch.setattr(cycle2_device_memory,'DeviceMemoryObserver',Busy)
    with pytest.raises(SystemExit) as error:supervisor.main()
    assert error.value.code==1 and closed==[True]
    result=json.loads((out/'shared-training-supervisor.json').read_text())
    assert result['status']=='failed' and result['reason']=='ValueError:exclusive_GPU_slot_not_idle'
