import json,tempfile
from pathlib import Path
import psutil,pytest
from scripts.supervise_lane_cycle2_retry1 import require_small_completed,require_idle_samples,PrerequisiteNotReady
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def receipt():
 base=ROOT/'.cache/lane-retry1-gate-tests';base.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=base) as temp:
  path=Path(temp)/'synthetic-receipt.json'
  value={'status':'completed','child_exit_code':0,'own_child_running_after_cleanup':False,'sources_unchanged':True,'own_child_pid':12345,'own_child_creation_time':1234567.,'plan_sha256':'expected'}
  path.write_text(json.dumps(value));yield path,value

def absent(pid):raise psutil.NoSuchProcess(pid)

def test_completed_cleaned_matching_run_and_absent_identity_allowed(receipt):
 path,data=receipt
 assert require_small_completed(path,expected_plan_sha='expected',process_factory=absent)['completed_and_child_identity_absent'] is True

@pytest.mark.parametrize('changes',[{'status':'running'},{'status':'starting'},{'status':'failed'},{'child_exit_code':1},{'child_exit_code':False},{'own_child_running_after_cleanup':True},{'sources_unchanged':False},{'own_child_pid':True},{'own_child_creation_time':float('nan')},{'own_child_creation_time':0}])
def test_no_training_start_from_incomplete_failed_or_invalid_receipt(receipt,changes):
 path,data=receipt;data.update(changes);path.write_text(json.dumps(data))
 with pytest.raises(ValueError):require_small_completed(path,expected_plan_sha='expected',process_factory=absent)

def test_wrong_small_experiment_binding_rejected(receipt):
 path,data=receipt
 with pytest.raises(ValueError,match='binding'):require_small_completed(path,expected_plan_sha='other',process_factory=absent)

def test_same_child_identity_still_running_must_wait(receipt):
 path,data=receipt
 class Live:
  def create_time(self):return data['own_child_creation_time']
  def is_running(self):return True
 with pytest.raises(PrerequisiteNotReady):require_small_completed(path,process_factory=lambda _:Live())

def test_reused_pid_is_not_mistaken_for_the_previous_training_child(receipt):
 path,data=receipt
 class Reused:
  def create_time(self):return data['own_child_creation_time']+100
  def is_running(self):return True
 assert require_small_completed(path,process_factory=lambda _:Reused())['completed_and_child_identity_absent'] is True

def test_two_recent_idle_samples_required():require_idle_samples([{'used_bytes':1024**3},{'used_bytes':1024**3}])

@pytest.mark.parametrize('samples',[[],[{'used_bytes':1}],[{'used_bytes':1},{'used_bytes':2*1024**3}],[{'used_bytes':3*1024**3},{'used_bytes':1}],[{'used_bytes':-1},{'used_bytes':1}],[{'used_bytes':True},{'used_bytes':1}]])
def test_not_idle_or_missing_measurement_cannot_start_training(samples):
 with pytest.raises(ValueError):require_idle_samples(samples)
