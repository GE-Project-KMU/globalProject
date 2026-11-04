"""Synthetic dependency/failure pins and CPU/GPU-controller mocks; no real GPU."""
import copy,json,subprocess
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import psutil,pytest
from scripts import supervise_redaction_rescue_v8_retry1 as c


def absent(pid):raise psutil.NoSuchProcess(pid)


@pytest.fixture
def evidence(tmp_path,monkeypatch):
 out=tmp_path/'out';out.mkdir();env=tmp_path/'environment';env.mkdir();root=c.ROOT
 monkeypatch.setattr(c,'OUT',out);monkeypatch.setattr(c,'ENV_ROOT',env)
 monkeypatch.setattr(c,'LEGACY_SPEC',tmp_path/'parent.json');monkeypatch.setattr(c,'LEGACY_SOURCE',tmp_path/'old-controller.py')
 monkeypatch.setattr(c,'PUBLIC_SPEC',tmp_path/'new-plan.json');monkeypatch.setattr(c,'PUBLIC_LEARNING',tmp_path/'learning-public.json')
 monkeypatch.setattr(c,'TRAINING_PYTHON',tmp_path/'python.exe');c.TRAINING_PYTHON.write_bytes(b'SYNTHETIC-PYTHON')
 monkeypatch.setattr(c.psutil,'Process',absent)
 def put(path,value):path.write_text(json.dumps(value));return {'path':path.relative_to(root).as_posix(),'sha256':c.sha256(path)}
 def ref(path):return {'path':path.relative_to(root).as_posix(),'sha256':c.sha256(path)}
 c.LEGACY_SOURCE.write_text('SYNTHETIC OLD CONTROLLER')
 source=c.sha256(root/'scripts/train_redaction_rescue_v8.py')
 learning={'epochs':12,'training_source_sha256':source,'source_expires_at':(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()}
 put(out/'plan.json',learning);put(c.PUBLIC_LEARNING,learning)
 put(out/'cpu-contract.json',{'passed':True,'trainer_source_sha256':source,'GPU_used':False})
 parent={'supervisor_source_sha256':c.sha256(c.LEGACY_SOURCE),'learning_plan_sha256':c.sha256(c.PUBLIC_LEARNING),
  'training_source_sha256':source,'input_sha256':{},'prerequisites':{},
  'initial_device_samples_each_strictly_below_bytes':2*1024**3,'initial_device_samples':2,'device_reported_budget_bytes':9*1024**3,
  'requested_query_interval_ms':100,'max_GPU_training_jobs':1,'source_deadline':learning['source_expires_at'],'phases':['preflight','train']}
 put(c.LEGACY_SPEC,parent);monkeypatch.setattr(c,'LEGACY_SPEC_SHA',c.sha256(c.LEGACY_SPEC))
 paths={key:tmp_path/(key+'.json' if key in ('receipt','samples') else key+'.log') for key in ('receipt','log','samples','observer_log')}
 monkeypatch.setattr(c,'failed_preflight_paths',lambda:paths)
 failed={'phase':'preflight','status':'failed_supervision','child_exit_code':1,'own_child_running_after_cleanup':False,
  'sources_unchanged':True,'own_OOM_signature_observed':False,'observer_child_stopped':True,'observer_thread_stopped':True,
  'device_memory':{'over_budget_observed':False},'supervision_plan_sha256':c.LEGACY_SPEC_SHA,
  'supervisor_source_sha256':parent['supervisor_source_sha256'],'learning_plan_sha256':parent['learning_plan_sha256'],
  'training_source_sha256':source,'own_child_pid':12345,'own_child_creation_time':1234567.}
 put(paths['receipt'],failed);put(paths['samples'],{'samples':[]});paths['log'].write_text('ModuleNotFoundError')
 paths['observer_log'].write_text('synthetic device observer')
 old={'numpy':'1.26.4','torch':'2.7.1+cu126'};packages=old|c.ADDED_PACKAGES
 preserved={path.relative_to(root).as_posix():c.sha256(path) for path in (paths['receipt'],paths['log'],root/'scripts/train_redaction_rescue_v8.py')}
 before={'packages':old,'preserved_sha256':preserved,'GPU_used':False,'repair_training_SOURCE_changed':False}
 put(env/'before.json',before)
 (env/'requirements-snapshot.txt').write_text('\n'.join(f'{name}=={version}' for name,version in sorted(packages.items())))
 (env/'install.log').write_text('SYNTHETIC FAILED INSTALL');(env/'install-compatible.log').write_text('SYNTHETIC COMPATIBLE INSTALL')
 put(env/'publisher-wheel-metadata.json',{'synthetic':True})
 after={'before_sha256':c.sha256(env/'before.json'),'requirements_snapshot':ref(env/'requirements-snapshot.txt'),
  'packages':packages,'added_packages':c.ADDED_PACKAGES,'existing_package_versions_unchanged':True,'GPU_used':False,
  'interpreter':{'executable':c.TRAINING_PYTHON.relative_to(root).as_posix(),'version':'3.11.9 SYNTHETIC'},'original_failure_sha256':preserved,
  'failed_install_log_preserved_sha256':c.sha256(env/'install.log'),'compatible_install_log_sha256':c.sha256(env/'install-compatible.log'),
  'publisher_wheel_metadata_sha256':c.sha256(env/'publisher-wheel-metadata.json')}
 put(env/'after.json',after)
 imports={'passed':True,'required_imports':c.REQUIRED_IMPORTS,'GPU_used':False,'cuda_initialized':False,'CUDA_VISIBLE_DEVICES':'',
  'source_pixels_read':False,'training_run_started':False,'after_sha256':c.sha256(env/'after.json'),
  'requirements_snapshot_sha256':c.sha256(env/'requirements-snapshot.txt'),'trainer_source_sha256':source,'module_versions':packages}
 put(env/'import-success.json',imports)
 spec=copy.deepcopy(parent);spec.update(format='rescue-v8-external-single-GPU-supervision-retry1-plan-v1',
  parent_supervision_plan=ref(c.LEGACY_SPEC),failed_preflight={key:ref(path) for key,path in paths.items()},
  environment_evidence={name:ref(env/name) for name in c.ENV_NAMES},training_python_sha256=c.sha256(c.TRAINING_PYTHON),
  supervisor_source_sha256=c.sha256(Path(c.__file__)))
 put(c.PUBLIC_SPEC,spec)
 return spec,paths,env,put,ref


def test_bound_original_failure_and_dependency_repair_are_CPU_metadata_only(evidence):
 spec,paths,env,_,_=evidence;before={path:path.read_bytes() for path in [*paths.values(),c.LEGACY_SPEC,c.LEGACY_SOURCE]}
 assert c.validate_retry_context(spec)['failed_preflight_sha256']==spec['failed_preflight']['receipt']['sha256']
 assert c.validate_pins(spec)[0]['epochs']==12
 assert all(path.read_bytes()==raw for path,raw in before.items())


@pytest.mark.parametrize('changes',[{'status':'completed'},{'own_child_running_after_cleanup':True},{'sources_unchanged':False},
 {'own_OOM_signature_observed':True},{'child_exit_code':0},{'observer_thread_stopped':False},{'own_child_pid':True}])
def test_invalid_original_failure_never_admits_GPU(evidence,changes):
 spec,paths,_,put,_=evidence;value=json.loads(paths['receipt'].read_text());value.update(changes)
 spec['failed_preflight']['receipt']=put(paths['receipt'],value)
 with pytest.raises(ValueError):c.validate_retry_context(spec)


def test_live_original_child_identity_blocks_without_stopping(evidence):
 spec,paths,_,_,_=evidence;value=json.loads(paths['receipt'].read_text())
 class Live:
  def create_time(self):return value['own_child_creation_time']
  def is_running(self):return True
 with pytest.raises(c.PrerequisiteNotReady):c.validate_retry_context(spec,process_factory=lambda _:Live())


@pytest.mark.parametrize('problem',['changed_old_source','changed_failure_log','package_upgrade','cuda','missing_import','wrong_after_hash','snapshot','resource_budget','python_binary'])
def test_retry_input_or_environment_tampering_is_rejected(evidence,problem):
 spec,paths,env,put,ref=evidence
 if problem=='changed_old_source':c.LEGACY_SOURCE.write_text('CHANGED')
 elif problem=='changed_failure_log':paths['log'].write_text('CHANGED')
 elif problem=='package_upgrade':
  value=json.loads((env/'after.json').read_text());value['packages']['numpy']='2.0';spec['environment_evidence']['after.json']=put(env/'after.json',value)
 elif problem in ('cuda','missing_import','wrong_after_hash'):
  value=json.loads((env/'import-success.json').read_text())
  if problem=='cuda':value['cuda_initialized']=True
  elif problem=='missing_import':value['required_imports']=['torch']
  else:value['after_sha256']='0'*64
  spec['environment_evidence']['import-success.json']=put(env/'import-success.json',value)
 elif problem=='snapshot':(env/'requirements-snapshot.txt').write_text('torch==CHANGED')
 elif problem=='resource_budget':spec['device_reported_budget_bytes']=10*1024**3
 else:c.TRAINING_PYTHON.write_bytes(b'CHANGED')
 with pytest.raises(ValueError):c.validate_retry_context(spec)


def current_probe_value(env):
 return {'packages':json.loads((env/'after.json').read_text())['packages'],'required_imports':c.REQUIRED_IMPORTS,
  'passed':True,'CUDA_VISIBLE_DEVICES':'','cuda_initialized':False}


def test_current_CPU_probe_uses_training_python_and_empty_CUDA_visibility(evidence,monkeypatch):
 spec,_,env,_,_=evidence;calls=[]
 def run(command,**kwargs):
  calls.append((command,kwargs));return SimpleNamespace(returncode=0,stdout=json.dumps(current_probe_value(env)),stderr='')
 monkeypatch.setattr(c.subprocess,'run',run)
 result=c.verify_current_environment(spec)
 assert result['passed'] and calls[0][0][0]==str(c.TRAINING_PYTHON)
 assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES']=='' and calls[0][1]['timeout']==60


@pytest.mark.parametrize('problem',['package_drift','cuda','nonzero','timeout'])
def test_actual_current_environment_must_match_pinned_after(evidence,monkeypatch,problem):
 spec,_,env,_,_=evidence
 def run(*args,**kwargs):
  if problem=='timeout':raise subprocess.TimeoutExpired('synthetic CPU probe',60)
  value=current_probe_value(env)
  if problem=='package_drift':value['packages']['numpy']='CHANGED'
  if problem=='cuda':value['cuda_initialized']=True
  return SimpleNamespace(returncode=1 if problem=='nonzero' else 0,stdout=json.dumps(value),stderr='')
 monkeypatch.setattr(c.subprocess,'run',run)
 with pytest.raises((ValueError,subprocess.TimeoutExpired)):c.verify_current_environment(spec)


def test_new_paths_are_disjoint_from_original_and_each_other(evidence):
 from scripts import supervise_redaction_rescue_v8 as old
 paths=c.phase_paths('preflight');train=c.phase_paths('train')
 assert set(paths.values()).isdisjoint(train.values()) and set(paths.values()).isdisjoint(old.phase_paths('preflight').values())
 assert all('retry1' in path.name for path in [*paths.values(),*train.values()])


def test_no_explicit_root_approval_never_queries_device_or_launches_child(monkeypatch):
 monkeypatch.setattr(c,'DeviceMemoryObserver',lambda *a:pytest.fail('device query'))
 with pytest.raises(ValueError,match='explicit_root_GPU_phase_approval'):c.supervise('preflight')


def test_failed_CPU_probe_claims_attempt_and_blocks_GPU_and_retry(evidence,tmp_path,monkeypatch):
 spec,_,_,put,_=evidence;root=tmp_path/'control-root';(root/'docs/reports').mkdir(parents=True)
 monkeypatch.setattr(c,'ROOT',root)
 expiry=datetime.now(timezone.utc)+timedelta(days=1)
 monkeypatch.setattr(c,'validate_pins',lambda *a,**k:({'epochs':12},expiry));monkeypatch.setattr(c,'prerequisites',lambda _: {})
 paths={key:tmp_path/(key+'-retry1.json') for key in ('receipt','log','observer_log','samples')}
 monkeypatch.setattr(c,'phase_paths',lambda _:paths)
 def fail(*args):raise ValueError('current_training_environment_differs_from_pinned_snapshot')
 monkeypatch.setattr(c,'verify_current_environment',fail)
 monkeypatch.setattr(c,'DeviceMemoryObserver',lambda *a:pytest.fail('device after failed CPU probe'))
 with pytest.raises(SystemExit):c.supervise('preflight',gpu_approved=True)
 value=json.loads(paths['receipt'].read_text());assert value['status']=='failed_supervision' and value['GPU_child_started'] is False
 with pytest.raises(ValueError,match='no_overwrite'):c.supervise('preflight',gpu_approved=True)


def test_train_requires_new_retry_preflight_and_exact_report(evidence):
 spec,_,_,put,_=evidence
 report={'plan_sha256':spec['learning_plan_sha256'],'training_source_sha256':spec['training_source_sha256'],
  'legacy_state_preserved':True,'actual_training_experiment_started':False,'TEST_images':0,
  'TRAIN_timing_batches':8,'ephemeral_optimizer_steps':8,'GPU_peak_allocated_GiB':1.,'GPU_peak_reserved_GiB':2.}
 put(c.OUT/'gpu-preflight.json',report)
 receipt={'format':'rescue-v8-external-supervision-retry1-v1','phase':'preflight','status':'completed','child_exit_code':0,
  'sources_unchanged':True,'own_child_running_after_cleanup':False,'own_child_pid':12345,'own_child_creation_time':1234567.,
  'learning_plan_sha256':spec['learning_plan_sha256'],'supervision_plan_sha256':c.sha256(c.PUBLIC_SPEC),
  'supervisor_source_sha256':c.sha256(Path(c.__file__)),'trainer_report_sha256':c.sha256(c.OUT/'gpu-preflight.json'),
  'separate_preflight_model_discarded':True,'GPU_child_started':True,'current_environment_verification':{'passed':True,'cuda_initialized':False}}
 path=c.OUT/'external-preflight-supervisor-retry1.json';put(path,receipt)
 assert c.prior_preflight(spec)['child_identity_absent'] is True
 report['ephemeral_optimizer_steps']=7;put(c.OUT/'gpu-preflight.json',report)
 with pytest.raises(ValueError,match='report_binding'):c.prior_preflight(spec)


def test_expiry_during_CPU_probe_never_starts_device_observer(evidence,tmp_path,monkeypatch):
 spec,_,_,_,_=evidence;root=tmp_path/'expiry-root';(root/'docs/reports').mkdir(parents=True)
 monkeypatch.setattr(c,'ROOT',root);monkeypatch.setattr(c,'validate_pins',lambda *a,**k:({'epochs':12},datetime.now(timezone.utc)-timedelta(seconds=1)))
 monkeypatch.setattr(c,'prerequisites',lambda _: {});monkeypatch.setattr(c,'verify_current_environment',lambda _: {'passed':True})
 paths={key:tmp_path/(key+'-expiry-retry1.json') for key in ('receipt','log','observer_log','samples')}
 monkeypatch.setattr(c,'phase_paths',lambda _:paths);monkeypatch.setattr(c,'DeviceMemoryObserver',lambda *a:pytest.fail('GPU observer after deadline'))
 with pytest.raises(SystemExit):c.supervise('preflight',gpu_approved=True)
 assert json.loads(paths['receipt'].read_text())['failure_reason']=='ValueError:original_source_deadline_reached'


@pytest.mark.parametrize('phase,trainer_action',[('preflight','gpu-preflight'),('train','train')])
def test_GPU_child_command_stays_frozen_and_training_never_resumes_preflight(evidence,tmp_path,monkeypatch,phase,trainer_action):
 spec,_,_,_,_=evidence;root=tmp_path/'command-root';(root/'docs/reports').mkdir(parents=True)
 monkeypatch.setattr(c,'ROOT',root);monkeypatch.setattr(c,'validate_pins',lambda *a,**k:({'epochs':12},datetime.now(timezone.utc)+timedelta(days=1)))
 monkeypatch.setattr(c,'prerequisites',lambda _: {});monkeypatch.setattr(c,'prior_preflight',lambda _: {'receipt_sha256':'pinned-new-preflight'})
 monkeypatch.setattr(c,'verify_current_environment',lambda _: {'passed':True})
 paths={key:tmp_path/(key+'-command-retry1.json') for key in ('receipt','log','observer_log','samples')}
 monkeypatch.setattr(c,'phase_paths',lambda _:paths)
 class Observer:
  owned=SimpleNamespace(process=SimpleNamespace(poll=lambda:0));thread=SimpleNamespace(is_alive=lambda:False)
  def __init__(self,*args):pass
  def start(self):return self
  def raise_if_unsafe(self):pass
  def sample_rows(self):return [{'used_bytes':1},{'used_bytes':1}]
  def close(self):pass
  def summary(self):return {'over_budget_observed':False}
 monkeypatch.setattr(c,'DeviceMemoryObserver',Observer);commands=[]
 def stop(command,**kwargs):commands.append((command,kwargs));raise RuntimeError('synthetic stop before any real child')
 monkeypatch.setattr(c.subprocess,'Popen',stop)
 with pytest.raises(SystemExit):c.supervise(phase,gpu_approved=True)
 assert len(commands)==1 and commands[0][0]==[str(c.TRAINING_PYTHON),str(root/'scripts/train_redaction_rescue_v8.py'),trainer_action,'--gpu-authorized']
 assert commands[0][1]['env']['CUDA_VISIBLE_DEVICES']=='0'
 assert json.loads(paths['receipt'].read_text())['GPU_child_started'] is False


def test_retry_control_retention_is_exact_and_never_preserves_private_extras(tmp_path,monkeypatch):
 from scripts import research_retention as retention
 monkeypatch.setattr(retention,'ROOT',tmp_path)
 folder=tmp_path/'data/training/redaction-rescue-v8';folder.mkdir(parents=True)
 path=folder/'external-preflight-supervisor-retry1.json'
 value={'format':'rescue-v8-external-supervision-retry1-v1','phase':'preflight','status':'failed_supervision',
  'failure_reason':'ValueError:own_child_nonzero_exit','sources_unchanged':True}
 path.write_text(json.dumps(value))
 with pytest.raises(retention.PreservedResearchFile,match='exact_retry_control'):retention.eligible(path,'identifiable_manifest')
 for extra in ({'image_sha256':'a'*64},{'debug_record':'private source image example.jpg'},{'coordinates':[1,2]},
               {'unexpected_private_uuid':'00000000-0000-0000-0000-000000000000'},
               {'device_memory':{'private_case_token':'00000000-0000-0000-0000-000000000000'}},
               {'current_environment_verification':{'unrecognized_source_record':'private'}}):
  path.write_text(json.dumps(value|extra));retention.eligible(path,'identifiable_manifest')
 other=folder/'unrecognized-retry1.json';other.write_text(json.dumps(value))
 retention.eligible(other,'identifiable_manifest')
