import json,tempfile
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
from agent.paths import sha256
from scripts import supervise_cycle2_lane as lane
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def sandbox(monkeypatch):
 base=ROOT/'.cache/cycle2-lane-supervision-tests';base.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=base) as folder:
  root=Path(folder)
  for p in ['scripts','docs/reports','data/training/public-nextcycle','data/public/a2d2-cycle2-v5']:(root/p).mkdir(parents=True,exist_ok=True)
  source=root/'scripts/train_lane_cycle2.py';source.write_text('synthetic stub; never executed')
  observer=root/'scripts/cycle2_device_memory.py';observer.write_text('synthetic fingerprint only')
  plan=root/'docs/reports/LANE-CYCLE2-PLAN.json';plan.write_text(json.dumps({'training_code_sha256':sha256(source)}))
  protocol=root/'docs/reports/LANE-CYCLE2-SHARED-SUPERVISION-PLAN.json';protocol.write_text(json.dumps({'supervisor_source_sha256':sha256(Path(lane.__file__)),'observer_source_sha256':sha256(observer),'learning_plan_sha256':sha256(plan)}))
  dev=root/'data/public/a2d2-cycle2-v5/development.json';dev.write_text(json.dumps([{'expires_at':(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()}]))
  calls=[]
  class FakeObserver:
   def __init__(self,*a,**k):calls.append('observer_constructed')
   def start(self):calls.append('synthetic_observer_start');raise ValueError('synthetic_monitor_failure_no_GPU')
  monkeypatch.setattr(lane,'ROOT',root);monkeypatch.setattr(lane,'DeviceMemoryObserver',FakeObserver)
  def forbidden_child(*a,**k):calls.append('unexpected_child');raise AssertionError('no_child_expected_in_policy_tests')
  monkeypatch.setattr(lane.subprocess,'Popen',forbidden_child)
  yield root,calls,source,dev

def test_lane_observer_failure_never_launches_training_and_records_failure(sandbox):
 root,calls,source,dev=sandbox
 with pytest.raises(SystemExit):lane.supervise()
 assert calls==['observer_constructed','synthetic_observer_start']
 report=json.loads((root/'data/training/public-nextcycle/lane-shared-training-supervisor.json').read_text())
 assert report['status']=='failed_supervision'
 assert report['other_process_controlled'] is False
 assert report['sources_unchanged'] is True

def test_lane_expired_source_rejected_before_observer_or_child(sandbox):
 root,calls,source,dev=sandbox;dev.write_text(json.dumps([{'expires_at':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()}]))
 with pytest.raises(ValueError,match='expired'):lane.supervise()
 assert calls==[]

def test_lane_source_pin_mismatch_rejected_before_any_child(sandbox):
 root,calls,source,dev=sandbox;source.write_text('changed')
 with pytest.raises(ValueError,match='frozen_supervision'):lane.supervise()
 assert calls==[]

def test_lane_existing_output_cannot_automatically_resume_or_overwrite(sandbox):
 root,calls,source,dev=sandbox;(root/'data/training/lane-cycle2-v5').mkdir()
 with pytest.raises(ValueError,match='fresh_training'):lane.supervise()
 assert calls==[]
