import copy,json,tempfile
from pathlib import Path
import psutil,pytest
from scripts import supervise_redaction_rescue_v8 as rescue
ROOT=Path(__file__).resolve().parents[1]

def absent(pid):raise psutil.NoSuchProcess(pid)

@pytest.fixture
def receipt():
 root=ROOT/'.cache/rescue-guardian-tests';root.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=root) as temp:
  path=Path(temp)/'synthetic-prerequisite.json';data={'status':'completed','child_exit_code':0,'own_child_running_after_cleanup':False,'sources_unchanged':True,'own_child_pid':12345,'own_child_creation_time':1234567.,'bound_plan':'p'};path.write_text(json.dumps(data));yield path,data

def test_prior_successful_cleanup_with_matching_plan_and_absent_child(receipt):
 path,data=receipt
 assert rescue.completed_child_absent(path,plan_field='bound_plan',plan_sha='p',process_factory=absent)['child_identity_absent'] is True

@pytest.mark.parametrize('changes',[{'status':'running'},{'status':'failed'},{'child_exit_code':False},{'child_exit_code':1},{'own_child_running_after_cleanup':True},{'sources_unchanged':False},{'bound_plan':'different'},{'own_child_pid':True},{'own_child_creation_time':float('nan')}])
def test_incomplete_failed_unbound_or_invalid_prior_blocks_GPU(receipt,changes):
 path,data=receipt;data.update(changes);path.write_text(json.dumps(data))
 with pytest.raises(ValueError):rescue.completed_child_absent(path,plan_field='bound_plan',plan_sha='p',process_factory=absent)

def test_same_prior_identity_live_blocks_without_termination(receipt):
 path,data=receipt
 class Live:
  def create_time(self):return data['own_child_creation_time']
  def is_running(self):return True
 with pytest.raises(rescue.PrerequisiteNotReady):rescue.completed_child_absent(path,plan_field='bound_plan',plan_sha='p',process_factory=lambda _:Live())

def test_initial_idle_strictly_two_below2GiB():
 rescue.require_idle([{'used_bytes':1},{'used_bytes':2**30}])
 for samples in ([],[{'used_bytes':1}],[{'used_bytes':1},{'used_bytes':2*2**30}],[{'used_bytes':True},{'used_bytes':1}]):
  with pytest.raises(ValueError):rescue.require_idle(samples)

def valid_preflight():
 return {'plan_sha256':'plan','training_source_sha256':'source','legacy_state_preserved':True,'actual_training_experiment_started':False,'TEST_images':0,'TRAIN_timing_batches':8,'ephemeral_optimizer_steps':8,'GPU_peak_allocated_GiB':1.,'GPU_peak_reserved_GiB':2.}
def spec():return {'learning_plan_sha256':'plan','training_source_sha256':'source'}

def test_matching_separate_8batch_report_is_resource_only():assert rescue.validate_preflight_report(valid_preflight(),spec()) is True

@pytest.mark.parametrize('changes',[{'plan_sha256':'changed'},{'training_source_sha256':'changed'},{'legacy_state_preserved':False},{'actual_training_experiment_started':True},{'TEST_images':1},{'TEST_images':False},{'TRAIN_timing_batches':7},{'ephemeral_optimizer_steps':0},{'GPU_peak_allocated_GiB':float('nan')},{'GPU_peak_reserved_GiB':10.},{'GPU_peak_allocated_GiB':3.},{'GPU_peak_reserved_GiB':0.}])
def test_report_mismatch_or_missing_actual_resource_evidence_blocks_train(changes):
 report=valid_preflight();report.update(changes)
 with pytest.raises(ValueError):rescue.validate_preflight_report(report,spec())

def test_preflight_and_train_never_share_receipt_or_log_namespace():
 a=rescue.phase_paths('preflight');b=rescue.phase_paths('train')
 assert set(a.values()).isdisjoint(set(b.values()))

def test_existing_phase_artifact_blocks_before_any_device_query(monkeypatch):
 root=ROOT/'.cache/rescue-guardian-tests';root.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=root) as temp:
  temp=Path(temp);public=temp/'spec.json';public.write_text('{}');out=temp/'out';out.mkdir();(out/'external-preflight-supervisor.json').write_text('{}')
  monkeypatch.setattr(rescue,'OUT',out);monkeypatch.setattr(rescue,'PUBLIC_SPEC',public)
  monkeypatch.setattr(rescue,'validate_pins',lambda _:({},None))
  paths={'receipt':out/'external-preflight-supervisor.json','log':temp/'new.log','observer_log':temp/'observer.log','samples':temp/'samples.json'}
  monkeypatch.setattr(rescue,'phase_paths',lambda _:paths)
  calls=[];monkeypatch.setattr(rescue,'DeviceMemoryObserver',lambda *a,**k:calls.append('unexpected_device'))
  with pytest.raises(ValueError,match='no_overwrite'):rescue.supervise('preflight')
  assert calls==[]
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace

@pytest.fixture
def control_sandbox(monkeypatch):
 base=ROOT/'.cache/rescue-guardian-tests';base.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=base) as folder:
  root=Path(folder);out=root/'data/training/redaction-rescue-v8';out.mkdir(parents=True);(root/'docs/reports').mkdir(parents=True)
  public=root/'docs/reports/spec.json';public.write_text(json.dumps({'learning_plan_sha256':'plan','training_source_sha256':'source'}))
  monkeypatch.setattr(rescue,'ROOT',root);monkeypatch.setattr(rescue,'OUT',out);monkeypatch.setattr(rescue,'PUBLIC_SPEC',public)
  monkeypatch.setattr(rescue,'validate_pins',lambda *a,**k:({'epochs':12},datetime.now(timezone.utc)+timedelta(days=1)))
  calls=[]
  def no_child(*a,**k):calls.append('training_child');raise AssertionError('no_GPU_child_allowed')
  monkeypatch.setattr(rescue.subprocess,'Popen',no_child)
  yield root,out,calls

def test_prior_pending_blocks_observer_and_Popen_in_actual_control_flow(control_sandbox,monkeypatch):
 root,out,calls=control_sandbox
 def pending(*a,**k):raise rescue.PrerequisiteNotReady('synthetic_prior_running')
 monkeypatch.setattr(rescue,'prerequisites',pending)
 monkeypatch.setattr(rescue,'DeviceMemoryObserver',lambda *a,**k:calls.append('observer'))
 with pytest.raises(rescue.PrerequisiteNotReady):rescue.supervise('preflight')
 assert calls==[] and not (out/'external-preflight-supervisor.json').exists()

def test_train_without_completed_preflight_blocks_observer_and_Popen(control_sandbox,monkeypatch):
 root,out,calls=control_sandbox;monkeypatch.setattr(rescue,'prerequisites',lambda _: {})
 def absent_preflight(*a,**k):raise rescue.PrerequisiteNotReady('synthetic_no_preflight')
 monkeypatch.setattr(rescue,'prior_preflight',absent_preflight)
 monkeypatch.setattr(rescue,'DeviceMemoryObserver',lambda *a,**k:calls.append('observer'))
 with pytest.raises(rescue.PrerequisiteNotReady):rescue.supervise('train')
 assert calls==[]

def test_busy_device_writes_failure_and_never_spawns_training(control_sandbox,monkeypatch):
 root,out,calls=control_sandbox;monkeypatch.setattr(rescue,'prerequisites',lambda _: {})
 class Busy:
  owned=SimpleNamespace(process=SimpleNamespace(poll=lambda:0));thread=SimpleNamespace(is_alive=lambda:False)
  def __init__(self,*a,**k):calls.append('observer')
  def start(self):return self
  def raise_if_unsafe(self):pass
  def sample_rows(self):return [{'used_bytes':3*1024**3},{'used_bytes':3*1024**3}]
  def close(self):calls.append('observer_closed')
  def summary(self):return {'synthetic_only':True}
 monkeypatch.setattr(rescue,'DeviceMemoryObserver',Busy)
 with pytest.raises(SystemExit):rescue.supervise('preflight')
 assert calls==['observer','observer_closed']
 result=json.loads((out/'external-preflight-supervisor.json').read_text())
 assert result['status']=='failed_supervision' and result['other_process_controlled'] is False
