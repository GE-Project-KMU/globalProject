"""Exclusive-GPU lane retry1 full-restart supervision; learning source stays frozen.

CLI is local and fixed-command. It never controls Tiny or any externally supplied
PID. Device sampling is a guard, not a hard VRAM quota or global OOM proof.
"""
import argparse,json,math,os,re,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256,workspace_path
from scripts.cycle2_device_memory import DeviceMemoryObserver,OwnedChild
OOM_SIGNATURES=['CUDA out of memory','CUDA error: out of memory','OutOfMemoryError','OOM RuntimeError']

def own_log_has_oom(text):return any(value.lower() in text.lower() for value in OOM_SIGNATURES)

class PrerequisiteNotReady(ValueError):pass

def require_small_completed(receipt_path=None,*,expected_plan_sha=None,process_factory=None):
 import psutil
 path=receipt_path or ROOT/'data/training/pothole-small-v9-retry1/shared-training-supervisor.json'
 if not path.exists():raise PrerequisiteNotReady('Small_retry1_completion_receipt_not_ready')
 data=json.loads(path.read_text(encoding='utf-8'))
 if data.get('status') in ('starting','running'):raise PrerequisiteNotReady('Small_retry1_still_running')
 if (data.get('status')!='completed' or type(data.get('child_exit_code')) is not int or data.get('child_exit_code')!=0 or data.get('own_child_running_after_cleanup') is not False or data.get('sources_unchanged') is not True):raise ValueError('Small_retry1_successful_cleanup_required')
 if expected_plan_sha is not None and data.get('plan_sha256')!=expected_plan_sha:raise ValueError('Small_retry1_plan_binding_mismatch')
 if type(data.get('own_child_pid')) is not int or data['own_child_pid']<=0 or type(data.get('own_child_creation_time')) not in (int,float) or not math.isfinite(data['own_child_creation_time']) or data['own_child_creation_time']<=0:raise ValueError('Small_retry1_child_identity_required')
 factory=process_factory or psutil.Process
 try:
  child=factory(data['own_child_pid'])
  if child.create_time()==data['own_child_creation_time'] and child.is_running():raise PrerequisiteNotReady('Small_retry1_child_identity_still_running')
 except psutil.NoSuchProcess:pass
 return {'receipt_sha256':sha256(path),'completed_and_child_identity_absent':True}

def require_idle_samples(samples):
 if len(samples)<2:raise ValueError('two_initial_device_samples_required')
 if any(type(v.get('used_bytes')) is not int or not 0<=v['used_bytes']<2*1024**3 for v in samples[-2:]):raise ValueError('exclusive_GPU_initial_device_memory_must_be_below_2GiB')

def supervise():
 from scripts.setup_training import environment
 source=ROOT/'scripts/train_lane_cycle2_retry1.py';learning_plan=ROOT/'docs/reports/LANE-CYCLE2-RETRY1-PLAN.json';protocol=ROOT/'docs/reports/LANE-CYCLE2-RETRY1-SUPERVISION-PLAN-V2.json'
 spec=json.loads(protocol.read_text());learning=json.loads(learning_plan.read_text());out=ROOT/'data/training/public-nextcycle/lane-retry1-training-supervisor.json';public=ROOT/'docs/reports/LANE-CYCLE2-RETRY1-SUPERVISION.json'
 if out.exists() or (ROOT/'data/training/lane-cycle2-v5-retry1').exists():raise ValueError('fresh_training_and_supervision_output_required')
 if spec['supervisor_source_sha256']!=sha256(Path(__file__)) or spec['observer_source_sha256']!=sha256(ROOT/'scripts/cycle2_device_memory.py') or spec['learning_plan_sha256']!=sha256(learning_plan) or learning['training_code_sha256']!=sha256(source):raise ValueError('frozen_supervision_learning_sources_required')
 development=ROOT/'data/public/a2d2-cycle2-v5/development.json';rows=json.loads(development.read_text());expiry=min(datetime.fromisoformat(r['expires_at'].replace('Z','+00:00')) for r in rows)
 if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc):raise ValueError('research_sources_expired')
 prerequisite=require_small_completed(expected_plan_sha=spec['prerequisite_Small_retry1_plan_sha256'])
 report={'format':'lane-exclusive-retry1-training-supervision-v1','status':'starting','learning_source_sha256':sha256(source),'learning_plan_sha256':sha256(learning_plan),'supervision_plan_sha256':sha256(protocol),'supervisor_source_sha256':sha256(Path(__file__)),'learning_objective_and_hyperparameters_changed':False,'runtime_backend_revision':'cudnn.benchmark=False','bitwise_training_equivalence_claimed':False,'restart_mode':'full_restart_not_resume','requires_exclusive_GPU_training':True,'global_OOM_zero_claimed':False,'OOM_detection_scope':'own child log only','owns_training_child_only':True,'other_process_controlled':False,'source_expiries_extended':False,'source_expiry':expiry.isoformat(),'prerequisite_small_completion_sha256':prerequisite['receipt_sha256'],'Small_retry1_completed_child_absent':True}
 observer=None;owned=None;log=None;start=time.monotonic();reason=None;oom=False
 try:
  observer=DeviceMemoryObserver(ROOT/'.cache/cycle2-lane-retry1-device-observer.log').start();observer.raise_if_unsafe();require_idle_samples(observer.sample_rows())
  if require_small_completed(expected_plan_sha=spec['prerequisite_Small_retry1_plan_sha256'])['receipt_sha256']!=prerequisite['receipt_sha256']:raise ValueError('Small_completion_receipt_changed_before_GPU_child')
  logpath=ROOT/'.cache/cycle2-lane-retry1-training.log';log=logpath.open('w',encoding='utf-8');env=environment();env['CUDA_VISIBLE_DEVICES']='0'
  process=subprocess.Popen([str(ROOT/'.venv-training/python.exe'),str(source),'train'],cwd=ROOT,env=env,stdout=log,stderr=log)
  try:owned=OwnedChild(process)
  except BaseException:
   # Popen's Windows process handle still binds our exact newly-created child,
   # so failed identity acquisition cannot leave it running unmonitored.
   if process.poll() is None:process.terminate();process.wait(timeout=10)
   raise
  report.update(status='running',own_child_pid=owned.pid,own_child_creation_time=owned.creation_time,parent_pid=owned.parent_pid);atomic_json(out,report)
  while process.poll() is None:
   observer.raise_if_unsafe()
   if datetime.now(timezone.utc)>=expiry:raise ValueError('research_source_deadline_reached')
   with logpath.open('rb') as reader:reader.seek(max(0,logpath.stat().st_size-524288));text=reader.read().decode('utf-8',errors='replace')
   if own_log_has_oom(text):oom=True;raise ValueError('own_child_OOM_signature_observed')
   time.sleep(.1)
  observer.raise_if_unsafe()
  with logpath.open('rb') as reader:reader.seek(max(0,logpath.stat().st_size-524288));final_text=reader.read().decode('utf-8',errors='replace')
  if own_log_has_oom(final_text):oom=True;raise ValueError('own_child_OOM_signature_at_exit')
  report['child_exit_code']=process.returncode;report['status']='completed' if process.returncode==0 else 'failed_child_exit'
 except BaseException as error:
  reason=type(error).__name__+':'+str(error);report['status']='failed_supervision'
  if owned:
   try:report['own_child_stopped_after_identity_check']=owned.stop()
   except BaseException as stop_error:report['termination_refused_or_failed']=type(stop_error).__name__
 finally:
  if observer:
   observer.close();report['device_memory']=observer.summary();atomic_json(out.with_name(out.stem+'-device-samples.json'),{'samples':observer.sample_rows()})
  if log:log.close()
  report['whole_supervised_wall_seconds']=time.monotonic()-start;report['failure_reason']=reason;report['own_OOM_signature_observed']=oom
  if owned:report['child_exit_code']=owned.process.poll();report['own_child_running_after_cleanup']=owned.process.poll() is None
  report['sources_unchanged']=sha256(source)==learning['training_code_sha256'] and sha256(learning_plan)==spec['learning_plan_sha256']
  if not report['sources_unchanged'] or report.get('own_child_running_after_cleanup'):report['status']='failed_supervision'
  atomic_json(out,report);safe={k:v for k,v in report.items() if k not in ['own_child_pid','own_child_creation_time','parent_pid']};atomic_json(public,safe)
 print(json.dumps({k:v for k,v in report.items() if k not in ['own_child_pid','own_child_creation_time','parent_pid']}),flush=True)
 if report['status']!='completed':raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['train-lane-retry1']);a=p.parse_args();supervise()
