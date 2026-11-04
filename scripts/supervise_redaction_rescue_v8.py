"""External, fixed-command, single-GPU Rescue resource/training supervision.

No trainer/factory/data mutation. Preflight is a separate process whose temporary
8 optimizer-step attempts are discarded; real training constructs fresh state.
"""
import argparse,json,math,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,workspace_path,sha256
from scripts.cycle2_device_memory import DeviceMemoryObserver,OwnedChild
from scripts.supervise_cycle2_training import own_log_has_oom
OUT=ROOT/'data/training/redaction-rescue-v8'
PUBLIC_SPEC=ROOT/'docs/reports/PRIVACY-RESCUE-V8-SUPERVISION-PLAN.json'

class PrerequisiteNotReady(ValueError):pass

def completed_child_absent(path,*,plan_field,plan_sha,process_factory=None):
 if not path.exists():raise PrerequisiteNotReady('required_prior_completion_receipt_missing')
 data=json.loads(path.read_text(encoding='utf-8'))
 if data.get('status') in ('starting','running'):raise PrerequisiteNotReady('required_prior_GPU_job_still_running')
 if (data.get('status')!='completed' or type(data.get('child_exit_code')) is not int or data['child_exit_code']!=0 or data.get('own_child_running_after_cleanup') is not False or data.get('sources_unchanged') is not True or data.get(plan_field)!=plan_sha):raise ValueError('successful_bound_prior_completion_and_cleanup_required')
 pid=data.get('own_child_pid');created=data.get('own_child_creation_time')
 if type(pid) is not int or pid<=0 or type(created) not in (int,float) or not math.isfinite(created) or created<=0:raise ValueError('prior_child_identity_invalid')
 try:
  process=(process_factory or psutil.Process)(pid)
  if process.create_time()==created and process.is_running():raise PrerequisiteNotReady('prior_child_identity_still_exists')
 except psutil.NoSuchProcess:pass
 return {'receipt_sha256':sha256(path),'child_identity_absent':True}

def require_idle(samples):
 if len(samples)<2 or any(type(v.get('used_bytes')) is not int or not 0<=v['used_bytes']<2*1024**3 for v in samples[-2:]):raise ValueError('two_device_samples_below2GiB_required_before_GPU_child')

def phase_paths(action):
 return {'receipt':OUT/f'external-{action}-supervisor.json','log':ROOT/f'.cache/cycle2-rescue-v8-{action}.log','observer_log':ROOT/f'.cache/cycle2-rescue-v8-{action}-observer.log','samples':OUT/f'external-{action}-device-samples.json'}

def validate_pins(spec,*,check_deadline=True):
 if spec['supervisor_source_sha256']!=sha256(Path(__file__)):raise ValueError('frozen_external_supervisor_required')
 for relative,digest in spec['input_sha256'].items():
  if sha256(workspace_path(relative,exists=True))!=digest:raise ValueError('rescue_input_source_plan_or_contract_changed')
 private=OUT/'plan.json';public=ROOT/'docs/reports/PRIVACY-RESCUE-V8-PLAN.json'
 if sha256(private)!=spec['learning_plan_sha256'] or sha256(public)!=spec['learning_plan_sha256']:raise ValueError('private_public_learning_plan_binding_failed')
 plan=json.loads(private.read_text(encoding='utf-8'));contract=json.loads((OUT/'cpu-contract.json').read_text(encoding='utf-8'))
 if plan['training_source_sha256']!=spec['training_source_sha256'] or contract.get('passed') is not True or contract.get('trainer_source_sha256')!=spec['training_source_sha256'] or contract.get('GPU_used') is not False:raise ValueError('matching_real_CPU_contract_required')
 expiry=datetime.fromisoformat(plan['source_expires_at'].replace('Z','+00:00'))
 if expiry.tzinfo is None:raise ValueError('timezone_aware_original_deadline_required')
 if check_deadline and expiry<=datetime.now(timezone.utc):raise ValueError('original_research_sources_expired')
 return plan,expiry

def prerequisites(spec):
 results={}
 for name,rule in spec['prerequisites'].items():
  results[name]=completed_child_absent(workspace_path(rule['receipt']),plan_field=rule['plan_field'],plan_sha=rule['plan_sha256'])
 return results

def validate_preflight_report(value,spec):
 if (value.get('plan_sha256')!=spec['learning_plan_sha256'] or value.get('training_source_sha256')!=spec['training_source_sha256'] or value.get('legacy_state_preserved') is not True or value.get('actual_training_experiment_started') is not False or type(value.get('TEST_images')) is not int or value.get('TEST_images')!=0 or type(value.get('TRAIN_timing_batches')) is not int or value.get('TRAIN_timing_batches')!=8 or type(value.get('ephemeral_optimizer_steps')) is not int or value.get('ephemeral_optimizer_steps')!=8):raise ValueError('matching_separate_fresh_GPU_preflight_required')
 for key in ['GPU_peak_allocated_GiB','GPU_peak_reserved_GiB']:
  number=value.get(key)
  if type(number) not in (int,float) or not math.isfinite(number) or not 0<number<=9:raise ValueError('finite_GPU_preflight_resource_proof_required')
 if value['GPU_peak_allocated_GiB']>value['GPU_peak_reserved_GiB']:raise ValueError('GPU_preflight_allocator_statistics_inconsistent')
 return True

def prior_preflight(spec):
 receipt=OUT/'external-preflight-supervisor.json';bound=completed_child_absent(receipt,plan_field='learning_plan_sha256',plan_sha=spec['learning_plan_sha256']);data=json.loads(receipt.read_text(encoding='utf-8'));report=OUT/'gpu-preflight.json'
 if data.get('phase')!='preflight' or data.get('trainer_report_sha256')!=sha256(report) or data.get('separate_preflight_model_discarded') is not True:raise ValueError('completed_preflight_child_and_report_binding_required')
 validate_preflight_report(json.loads(report.read_text(encoding='utf-8')),spec)
 return bound

def supervise(action):
 if action not in ('preflight','train'):raise ValueError('fixed_preflight_or_train_action_required')
 spec=json.loads(PUBLIC_SPEC.read_text(encoding='utf-8'));plan,expiry=validate_pins(spec);paths=phase_paths(action)
 if any(path.exists() for path in paths.values()):raise ValueError('new_phase_receipts_logs_only_no_overwrite_or_automatic_retry')
 if any((OUT/name).exists() for name in ['provenance.json','history.json','training-status.json','epoch-recovery.pt','model.json','model.onnx']):raise ValueError('fresh_training_experiment_required')
 if action=='preflight' and (OUT/'gpu-preflight.json').exists():raise ValueError('preflight_already_recorded')
 prior=prerequisites(spec)
 if action=='train':prior['separate_preflight']=prior_preflight(spec)
 report={'format':'rescue-v8-external-supervision-v1','phase':action,'status':'starting','learning_plan_sha256':spec['learning_plan_sha256'],'training_source_sha256':spec['training_source_sha256'],'supervision_plan_sha256':sha256(PUBLIC_SPEC),'supervisor_source_sha256':sha256(Path(__file__)),'prerequisite_receipt_sha256':{k:v['receipt_sha256'] for k,v in prior.items()},'requires_single_GPU_training_job':True,'training_math_changed':False,'owns_training_child_only':True,'other_process_controlled':False,'global_OOM_zero_claimed':False,'OOM_detection_scope':'own child log only','source_expiries_extended':False,'TEST_images':0,'preflight_optimizer_step_attempts_reported':8 if action=='preflight' else None,'preflight_is_real_training_experiment':False,'separate_preflight_model_discarded':action=='preflight','fresh_construct_for_train':action=='train'}
 observer=None;owned=None;log=None;reason=None;oom=False;start=time.monotonic()
 try:
  observer=DeviceMemoryObserver(paths['observer_log']).start();observer.raise_if_unsafe();require_idle(observer.sample_rows())
  refreshed=prerequisites(spec)
  if any(refreshed[k]['receipt_sha256']!=prior[k]['receipt_sha256'] for k in refreshed):raise ValueError('prior_completion_receipts_changed_before_child')
  if action=='train' and prior_preflight(spec)['receipt_sha256']!=prior['separate_preflight']['receipt_sha256']:raise ValueError('preflight_receipt_changed_before_train')
  from scripts.setup_training import environment
  trainer_action='gpu-preflight' if action=='preflight' else 'train';command=[str(ROOT/'.venv-training/python.exe'),str(ROOT/'scripts/train_redaction_rescue_v8.py'),trainer_action,'--gpu-authorized']
  log=paths['log'].open('x',encoding='utf-8');env=environment();env['CUDA_VISIBLE_DEVICES']='0'
  process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=log)
  try:owned=OwnedChild(process)
  except BaseException:
   if process.poll() is None:process.terminate();process.wait(timeout=10)
   raise
  report.update(status='running',own_child_pid=owned.pid,own_child_creation_time=owned.creation_time,parent_pid=owned.parent_pid);atomic_json(paths['receipt'],report)
  while process.poll() is None:
   observer.raise_if_unsafe()
   if datetime.now(timezone.utc)>=expiry:raise ValueError('original_source_deadline_reached')
   with paths['log'].open('rb') as reader:reader.seek(max(0,paths['log'].stat().st_size-524288));text=reader.read().decode('utf-8',errors='replace')
   if own_log_has_oom(text):oom=True;raise ValueError('own_child_OOM_signature_or_fallback')
   time.sleep(.1)
  observer.raise_if_unsafe()
  with paths['log'].open('rb') as reader:reader.seek(max(0,paths['log'].stat().st_size-524288));last=reader.read().decode('utf-8',errors='replace')
  if own_log_has_oom(last):oom=True;raise ValueError('own_child_OOM_signature_at_exit')
  if process.returncode!=0:raise ValueError('own_child_nonzero_exit')
  if action=='preflight':
   trainer_report=OUT/'gpu-preflight.json';validate_preflight_report(json.loads(trainer_report.read_text(encoding='utf-8')),spec)
  else:
   trainer_report=OUT/'training-status.json';status=json.loads(trainer_report.read_text(encoding='utf-8'))
   if status.get('status')!='completed' or status.get('epochs')!=plan['epochs'] or status.get('TEST_images')!=0:raise ValueError('complete_frozen_training_status_required')
   manifest=json.loads((OUT/'model.json').read_text(encoding='utf-8'))
   if manifest.get('training_complete') is not True or manifest.get('optimizer_updates',0)<1 or sha256(workspace_path(manifest['weights'],exists=True))!=manifest['sha256']:raise ValueError('actual_trained_artifact_required')
  report['trainer_report_sha256']=sha256(trainer_report);report['status']='completed'
 except BaseException as error:
  reason=type(error).__name__+':'+str(error);report['status']='failed_supervision'
  if owned:
   try:report['own_child_stopped_after_identity_check']=owned.stop()
   except BaseException as stop_error:report['termination_refused_or_failed']=type(stop_error).__name__
 finally:
  if observer:
   observer.close();report['device_memory']=observer.summary();report['observer_child_stopped']=observer.owned.process.poll() is not None;report['observer_thread_stopped']=not observer.thread.is_alive();atomic_json(paths['samples'],{'samples':observer.sample_rows()})
   if not report['observer_child_stopped'] or not report['observer_thread_stopped']:report['status']='failed_supervision'
  if log:log.close()
  report['wall_seconds']=time.monotonic()-start;report['failure_reason']=reason;report['own_OOM_signature_observed']=oom
  if owned:report['child_exit_code']=owned.process.poll();report['own_child_running_after_cleanup']=owned.process.poll() is None
  try:validate_pins(spec,check_deadline=False);report['sources_unchanged']=sha256(PUBLIC_SPEC)==report['supervision_plan_sha256']
  except (ValueError,FileNotFoundError,KeyError):report['sources_unchanged']=False
  if not report['sources_unchanged'] or report.get('own_child_running_after_cleanup'):report['status']='failed_supervision'
  atomic_json(paths['receipt'],report);safe={k:v for k,v in report.items() if k not in ('own_child_pid','own_child_creation_time','parent_pid')};atomic_json(ROOT/f'docs/reports/PRIVACY-RESCUE-V8-{action.upper()}-SUPERVISION.json',safe)
 print(json.dumps({k:v for k,v in report.items() if k not in ('own_child_pid','own_child_creation_time','parent_pid')}),flush=True)
 if report['status']!='completed':raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','train']);a=p.parse_args();supervise(a.action)
