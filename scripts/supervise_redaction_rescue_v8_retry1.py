"""Retry1 copy of the original external Rescue controller; learning math unchanged.

No trainer/factory/data mutation. Preflight is a separate process whose temporary
8 optimizer-step attempts are discarded; real training constructs fresh state.
"""
import argparse,hashlib,json,math,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,workspace_path,sha256
from scripts.cycle2_device_memory import DeviceMemoryObserver,OwnedChild
from scripts.supervise_cycle2_training import own_log_has_oom
OUT=ROOT/'data/training/redaction-rescue-v8'
PUBLIC_SPEC=ROOT/'docs/reports/PRIVACY-RESCUE-V8-RETRY1-SUPERVISION-PLAN.json'
PUBLIC_LEARNING=ROOT/'docs/reports/PRIVACY-RESCUE-V8-PLAN.json'
LEGACY_SPEC=ROOT/'docs/reports/PRIVACY-RESCUE-V8-SUPERVISION-PLAN.json'
LEGACY_SPEC_SHA='c5e1dcc65cdd48d11d46b79f0fe6caf0f045bf5628c8597f5356a1ca0a29b1ae'
LEGACY_SOURCE=ROOT/'scripts/supervise_redaction_rescue_v8.py'
ENV_ROOT=ROOT/'data/training/redaction-rescue-v8-env-repair'
TRAINING_PYTHON=ROOT/'.venv-training/python.exe'
ENV_NAMES=('before.json','after.json','import-success.json','requirements-snapshot.txt','install.log','install-compatible.log','publisher-wheel-metadata.json')
REQUIRED_IMPORTS=['agent.evaluate','agent.pipeline','yolox.utils','torch']
ADDED_PACKAGES={'av':'18.1.0','cryptography':'50.0.2','cffi':'2.1.1','pycparser':'3.12'}


def read_json(path):
 path=workspace_path(path,exists=True)
 if path.stat().st_size>8*1024**2:raise ValueError('bounded_retry_environment_JSON_required')
 return json.loads(path.read_text(encoding='utf-8'))


def exact_ref(ref,path):
 if ref!={'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path)}:raise ValueError('retry_evidence_path_or_hash_changed')


def failed_preflight_paths():
 return {'receipt':OUT/'external-preflight-supervisor.json','log':ROOT/'.cache/cycle2-rescue-v8-preflight.log',
  'samples':OUT/'external-preflight-device-samples.json','observer_log':ROOT/'.cache/cycle2-rescue-v8-preflight-observer.log'}


def failed_child_absent(data,process_factory=None):
 pid=data.get('own_child_pid');created=data.get('own_child_creation_time')
 if type(pid) is not int or pid<=0 or type(created) not in (int,float) or not math.isfinite(created) or created<=0:raise ValueError('failed_child_identity_invalid')
 try:
  process=(process_factory or psutil.Process)(pid)
  if process.create_time()==created and process.is_running():raise PrerequisiteNotReady('failed_preflight_child_identity_still_exists')
 except psutil.NoSuchProcess:pass
 return True


def validate_retry_context(spec,*,process_factory=None):
 if spec.get('format')!='rescue-v8-external-single-GPU-supervision-retry1-plan-v1':raise ValueError('explicit_retry1_supervision_plan_required')
 exact_ref(spec['parent_supervision_plan'],LEGACY_SPEC)
 if sha256(LEGACY_SPEC)!=LEGACY_SPEC_SHA:raise ValueError('original_supervision_plan_changed')
 parent=read_json(LEGACY_SPEC)
 if sha256(LEGACY_SOURCE)!=parent['supervisor_source_sha256']:raise ValueError('original_supervisor_source_changed')
 if spec['learning_plan_sha256']!=parent['learning_plan_sha256'] or spec['training_source_sha256']!=parent['training_source_sha256'] or spec['prerequisites']!=parent['prerequisites']:raise ValueError('retry_must_preserve_learning_and_prior_jobs')
 for key in ('initial_device_samples_each_strictly_below_bytes','initial_device_samples','device_reported_budget_bytes','requested_query_interval_ms','max_GPU_training_jobs','source_deadline','phases'):
  if spec.get(key)!=parent.get(key):raise ValueError('original_resource_and_expiry_limits_must_be_preserved')
 if any(spec['input_sha256'].get(path)!=digest for path,digest in parent['input_sha256'].items()):raise ValueError('original_input_pins_must_be_preserved')
 paths=failed_preflight_paths()
 if set(spec['failed_preflight'])!=set(paths):raise ValueError('complete_original_failure_evidence_required')
 for key,path in paths.items():exact_ref(spec['failed_preflight'][key],path)
 failed=read_json(paths['receipt'])
 if (failed.get('phase')!='preflight' or failed.get('status')!='failed_supervision' or type(failed.get('child_exit_code')) is not int or failed['child_exit_code']==0 or
     failed.get('own_child_running_after_cleanup') is not False or failed.get('sources_unchanged') is not True or failed.get('own_OOM_signature_observed') is not False or
     failed.get('observer_child_stopped') is not True or failed.get('observer_thread_stopped') is not True or failed.get('device_memory',{}).get('over_budget_observed') is not False or
     failed.get('supervision_plan_sha256')!=LEGACY_SPEC_SHA or failed.get('supervisor_source_sha256')!=parent['supervisor_source_sha256'] or
     failed.get('learning_plan_sha256')!=parent['learning_plan_sha256'] or failed.get('training_source_sha256')!=parent['training_source_sha256']):raise ValueError('bound_failed_nonOOM_cleaned_preflight_required')
 if paths['log'].stat().st_size>4*1024**2:raise ValueError('bounded_original_failure_log_required')
 text=paths['log'].read_text(encoding='utf-8',errors='replace')
 if own_log_has_oom(text) or 'ModuleNotFoundError' not in text:raise ValueError('environment_import_failure_without_OOM_required')
 failed_child_absent(failed,process_factory)
 evidence=spec['environment_evidence']
 if set(evidence)!=set(ENV_NAMES):raise ValueError('complete_environment_repair_evidence_required')
 for name in ENV_NAMES:exact_ref(evidence[name],ENV_ROOT/name)
 before=read_json(ENV_ROOT/'before.json');after=read_json(ENV_ROOT/'after.json');imports=read_json(ENV_ROOT/'import-success.json')
 if before.get('GPU_used') is not False or before.get('repair_training_SOURCE_changed') is not False or after.get('GPU_used') is not False or after.get('existing_package_versions_unchanged') is not True:raise ValueError('CPU_only_dependency_repair_required')
 if after.get('before_sha256')!=evidence['before.json']['sha256'] or after.get('requirements_snapshot')!=evidence['requirements-snapshot.txt'] or after.get('original_failure_sha256')!=before.get('preserved_sha256'):raise ValueError('environment_before_after_binding_required')
 for path in (paths['receipt'],paths['log'],ROOT/'scripts/train_redaction_rescue_v8.py'):
  if before['preserved_sha256'].get(path.relative_to(ROOT).as_posix())!=sha256(path):raise ValueError('original_failure_or_training_source_changed')
 old=before['packages'];new=after['packages']
 if any(new.get(name)!=version for name,version in old.items()) or {name:version for name,version in new.items() if name not in old}!=ADDED_PACKAGES or after.get('added_packages')!=ADDED_PACKAGES:raise ValueError('only_pinned_missing_dependencies_may_be_added')
 snapshot={}
 for line in (ENV_ROOT/'requirements-snapshot.txt').read_text(encoding='utf-8').splitlines():
  if not line or line.startswith('#'):continue
  name,separator,version=line.partition('==')
  if not separator or not name or not version or name in snapshot:raise ValueError('exact_pinned_environment_snapshot_required')
  snapshot[name]=version
 if snapshot!=new:raise ValueError('requirements_snapshot_must_match_after_packages')
 interpreter=after.get('interpreter',{})
 if interpreter.get('executable','').replace('\\','/')!=TRAINING_PYTHON.relative_to(ROOT).as_posix() or not interpreter.get('version','').startswith('3.11.'):raise ValueError('same_project_training_interpreter_required')
 if spec.get('training_python_sha256')!=sha256(TRAINING_PYTHON):raise ValueError('training_python_binary_changed')
 if (imports.get('passed') is not True or imports.get('required_imports')!=REQUIRED_IMPORTS or imports.get('GPU_used') is not False or imports.get('cuda_initialized') is not False or
     imports.get('CUDA_VISIBLE_DEVICES')!='' or imports.get('source_pixels_read') is not False or imports.get('training_run_started') is not False or
     imports.get('after_sha256')!=evidence['after.json']['sha256'] or imports.get('requirements_snapshot_sha256')!=evidence['requirements-snapshot.txt']['sha256'] or imports.get('trainer_source_sha256')!=spec['training_source_sha256']):raise ValueError('matching_CPU_import_diagnostic_required')
 if any(new.get(name)!=version for name,version in imports.get('module_versions',{}).items()) or not set(ADDED_PACKAGES)<=set(imports.get('module_versions',{})):raise ValueError('imported_dependency_versions_must_match_snapshot')
 for name,key in [('install.log','failed_install_log_preserved_sha256'),('install-compatible.log','compatible_install_log_sha256'),('publisher-wheel-metadata.json','publisher_wheel_metadata_sha256')]:
  if after.get(key)!=evidence[name]['sha256']:raise ValueError('dependency_install_evidence_binding_required')
 return {'failed_preflight_sha256':spec['failed_preflight']['receipt']['sha256'],'CPU_import_evidence_sha256':evidence['import-success.json']['sha256']}


def verify_current_environment(spec):
 """Separate CPU-only import process; timeout kills/waits only its own handle."""
 from scripts.setup_training import environment
 expected=read_json(ENV_ROOT/'after.json')['packages']
 code="""import importlib,importlib.metadata as metadata,json,os,sys
sys.path[:0]=[sys.argv[1],sys.argv[2]]
names=['agent.evaluate','agent.pipeline','yolox.utils','torch']
for name in names:importlib.import_module(name)
import torch
print(json.dumps({'packages':{dist.metadata['Name'].lower():dist.version for dist in metadata.distributions()},'required_imports':names,'passed':True,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'cuda_initialized':torch.cuda.is_initialized()}))
"""
 env=environment();env['CUDA_VISIBLE_DEVICES']=''
 result=subprocess.run([str(TRAINING_PYTHON),'-c',code,str(ROOT),str(ROOT/'.cache/vendor/YOLOX')],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
 if result.returncode!=0 or len(result.stdout)>1024**2 or len(result.stderr)>1024**2:raise ValueError('current_CPU_import_environment_probe_failed')
 try:value=json.loads(result.stdout.strip().splitlines()[-1])
 except (ValueError,IndexError):raise ValueError('current_CPU_import_environment_probe_invalid') from None
 if (value.get('passed') is not True or value.get('required_imports')!=REQUIRED_IMPORTS or value.get('packages')!=expected or
     value.get('CUDA_VISIBLE_DEVICES')!='' or value.get('cuda_initialized') is not False or sha256(TRAINING_PYTHON)!=spec['training_python_sha256']):raise ValueError('current_training_environment_differs_from_pinned_snapshot')
 return {'passed':True,'packages_sha256':hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest(),
  'CPU_only':True,'cuda_initialized':False,'CUDA_VISIBLE_DEVICES':'','timeout_seconds':60,
  'interpreter_sha256':spec['training_python_sha256'],'imports':REQUIRED_IMPORTS}

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
 return {'receipt':OUT/f'external-{action}-supervisor-retry1.json','log':ROOT/f'.cache/cycle2-rescue-v8-{action}-retry1.log','observer_log':ROOT/f'.cache/cycle2-rescue-v8-{action}-observer-retry1.log','samples':OUT/f'external-{action}-device-samples-retry1.json'}

def validate_pins(spec,*,check_deadline=True):
 if spec['supervisor_source_sha256']!=sha256(Path(__file__)):raise ValueError('frozen_external_supervisor_required')
 validate_retry_context(spec)
 for relative,digest in spec['input_sha256'].items():
  if sha256(workspace_path(relative,exists=True))!=digest:raise ValueError('rescue_input_source_plan_or_contract_changed')
 private=OUT/'plan.json';public=PUBLIC_LEARNING
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
 receipt=OUT/'external-preflight-supervisor-retry1.json';bound=completed_child_absent(receipt,plan_field='learning_plan_sha256',plan_sha=spec['learning_plan_sha256']);data=json.loads(receipt.read_text(encoding='utf-8'));report=OUT/'gpu-preflight.json'
 if data.get('format')!='rescue-v8-external-supervision-retry1-v1' or data.get('supervision_plan_sha256')!=sha256(PUBLIC_SPEC) or data.get('supervisor_source_sha256')!=sha256(Path(__file__)) or data.get('phase')!='preflight' or data.get('trainer_report_sha256')!=sha256(report) or data.get('separate_preflight_model_discarded') is not True or data.get('GPU_child_started') is not True or data.get('current_environment_verification',{}).get('passed') is not True or data.get('current_environment_verification',{}).get('cuda_initialized') is not False:raise ValueError('completed_preflight_child_and_report_binding_required')
 validate_preflight_report(json.loads(report.read_text(encoding='utf-8')),spec)
 return bound

def supervise(action,*,gpu_approved=False):
 if gpu_approved is not True:raise ValueError('explicit_root_GPU_phase_approval_required')
 if action not in ('preflight','train'):raise ValueError('fixed_preflight_or_train_action_required')
 spec=json.loads(PUBLIC_SPEC.read_text(encoding='utf-8'));plan,expiry=validate_pins(spec);paths=phase_paths(action)
 if any(path.exists() for path in paths.values()):raise ValueError('new_phase_receipts_logs_only_no_overwrite_or_automatic_retry')
 if any((OUT/name).exists() for name in ['provenance.json','history.json','training-status.json','epoch-recovery.pt','model.json','model.onnx']):raise ValueError('fresh_training_experiment_required')
 if action=='preflight' and (OUT/'gpu-preflight.json').exists():raise ValueError('preflight_already_recorded')
 prior=prerequisites(spec)
 if action=='train':prior['separate_preflight']=prior_preflight(spec)
 report={'format':'rescue-v8-external-supervision-retry1-v1','phase':action,'status':'starting','learning_plan_sha256':spec['learning_plan_sha256'],'training_source_sha256':spec['training_source_sha256'],'supervision_plan_sha256':sha256(PUBLIC_SPEC),'supervisor_source_sha256':sha256(Path(__file__)),'prerequisite_receipt_sha256':{k:v['receipt_sha256'] for k,v in prior.items()},'requires_single_GPU_training_job':True,'training_math_changed':False,'owns_training_child_only':True,'other_process_controlled':False,'global_OOM_zero_claimed':False,'OOM_detection_scope':'own child log only','source_expiries_extended':False,'TEST_images':0,'preflight_optimizer_step_attempts_reported':8 if action=='preflight' else None,'preflight_is_real_training_experiment':False,'separate_preflight_model_discarded':action=='preflight','fresh_construct_for_train':action=='train',
  'parent_failed_preflight_sha256':spec['failed_preflight']['receipt']['sha256'],'CPU_import_evidence_sha256':spec['environment_evidence']['import-success.json']['sha256'],'dependency_repair_only':True,'GPU_child_started':False}
 # Exclusive attempt claim precedes even the new CPU probe. A concurrent or
 # failed attempt cannot overwrite another owner or silently retry this phase.
 with paths['receipt'].open('x',encoding='utf-8') as claim:json.dump(report,claim)
 observer=None;owned=None;log=None;reason=None;oom=False;start=time.monotonic()
 try:
  report['current_environment_verification']=verify_current_environment(spec)
  if datetime.now(timezone.utc)>=expiry:raise ValueError('original_source_deadline_reached')
  observer=DeviceMemoryObserver(paths['observer_log']).start();observer.raise_if_unsafe();require_idle(observer.sample_rows())
  refreshed=prerequisites(spec)
  if any(refreshed[k]['receipt_sha256']!=prior[k]['receipt_sha256'] for k in refreshed):raise ValueError('prior_completion_receipts_changed_before_child')
  if action=='train' and prior_preflight(spec)['receipt_sha256']!=prior['separate_preflight']['receipt_sha256']:raise ValueError('preflight_receipt_changed_before_train')
  if datetime.now(timezone.utc)>=expiry:raise ValueError('original_source_deadline_reached')
  from scripts.setup_training import environment
  trainer_action='gpu-preflight' if action=='preflight' else 'train';command=[str(TRAINING_PYTHON),str(ROOT/'scripts/train_redaction_rescue_v8.py'),trainer_action,'--gpu-authorized']
  log=paths['log'].open('x',encoding='utf-8');env=environment();env['CUDA_VISIBLE_DEVICES']='0'
  process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=log)
  try:owned=OwnedChild(process)
  except BaseException:
   if process.poll() is None:process.terminate();process.wait(timeout=10)
   raise
  report.update(status='running',GPU_child_started=True,own_child_pid=owned.pid,own_child_creation_time=owned.creation_time,parent_pid=owned.parent_pid);atomic_json(paths['receipt'],report)
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
  atomic_json(paths['receipt'],report);safe={k:v for k,v in report.items() if k not in ('own_child_pid','own_child_creation_time','parent_pid')};atomic_json(ROOT/f'docs/reports/PRIVACY-RESCUE-V8-{action.upper()}-RETRY1-SUPERVISION.json',safe)
 print(json.dumps({k:v for k,v in report.items() if k not in ('own_child_pid','own_child_creation_time','parent_pid')}),flush=True)
 if report['status']!='completed':raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','train']);p.add_argument('--gpu-phase-approved',action='store_true');a=p.parse_args();supervise(a.action,gpu_approved=a.gpu_phase_approved)
