import os,subprocess,sys,time,uuid
from pathlib import Path
import pytest
from scripts.cycle2_device_memory import DeviceMemoryObserver,OwnedChild,parse_used_mib,LIMIT
from scripts.supervise_cycle2_training import own_log_has_oom
ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'.venv-training/python.exe' if os.name=='nt' else Path(sys.executable)
requires_python=pytest.mark.skipif(not PYTHON.is_file(),reason='optional workspace training interpreter missing on Windows')

@pytest.mark.parametrize('text,expected',[('3880',3880*1024**2),(' 9216\n',LIMIT),('0.000001',2)])
def test_device_units_and_conservative_fraction_roundup(text,expected):assert parse_used_mib(text)==expected

@pytest.mark.parametrize('text',['N/A','3880 MiB','3,880','-1','nan','1e4',''])
def test_invalid_or_missing_device_measurement_is_not_zero(text):
 with pytest.raises(ValueError):parse_used_mib(text)

@requires_python
def test_own_process_creation_time_mismatch_refuses_termination():
 child=subprocess.Popen([str(PYTHON),'-c','import time; time.sleep(30)'],cwd=ROOT)
 owned=OwnedChild(child)
 try:
  actual=owned.creation_time;owned.creation_time=actual+1
  assert owned.stop() is False
  assert child.poll() is None
  owned.creation_time=actual
  assert owned.stop() is True
  assert child.poll() is not None
 finally:
  owned.creation_time=actual
  if child.poll() is None:owned.stop()

@requires_python
def test_device_observer_fake_query_enforces_budget_and_closes_only_own_child():
 logs=ROOT/'.cache/cycle2-supervision-tests';logs.mkdir(parents=True,exist_ok=True)
 trigger=logs/(str(uuid.uuid4())+'.trigger')
 script="import time; from pathlib import Path; print(5000,flush=True); time.sleep(.05); print(5000,flush=True)\nwhile not Path("+repr(str(trigger))+").exists():time.sleep(.01)\nprint(9500,flush=True);time.sleep(30)"
 observer=DeviceMemoryObserver(logs/(str(uuid.uuid4())+'.log'),command=[str(PYTHON),'-u','-c',script])
 try:
  observer.start();trigger.write_text('synthetic-trigger');end=time.monotonic()+5
  while not observer.violation and time.monotonic()<end:time.sleep(.01)
  with pytest.raises(ValueError,match='over_9GiB'):observer.raise_if_unsafe()
  summary=observer.summary()
  assert summary['max_device_reported_used_bytes']==9500*1024**2
  assert summary['timestamps_are_receive_times_not_hardware_sample_times'] is True
  assert summary['observed_receive_interval_ms']['min']>0
 finally:
  observer.close()
  trigger.unlink(missing_ok=True)
 assert observer.owned.process.poll() is not None
 assert not observer.thread.is_alive()

@requires_python
def test_device_observer_parse_error_fails_closed_and_cleans_up():
 logs=ROOT/'.cache/cycle2-supervision-tests';logs.mkdir(parents=True,exist_ok=True)
 observer=DeviceMemoryObserver(logs/(str(uuid.uuid4())+'.log'),command=[str(PYTHON),'-u','-c',"import time; print('N/A',flush=True); time.sleep(30)"])
 with pytest.raises(ValueError,match='parse_or_runtime_failed'):observer.start()
 assert observer.owned.process.poll() is not None

@pytest.mark.parametrize('text',["RuntimeError: CUDA out of memory",'OOM RuntimeError is raised due to assignment','torch.OutOfMemoryError: x'])
def test_own_log_OOM_fallback_is_not_hidden(text):assert own_log_has_oom(text)

def test_regular_training_log_is_not_an_OOM():assert not own_log_has_oom('{"epoch":1,"loss":2.1}')
