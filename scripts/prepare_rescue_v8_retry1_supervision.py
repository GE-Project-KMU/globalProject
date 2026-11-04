"""Metadata-only retry1 freeze; original failure and learning inputs are immutable."""
import copy,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import sha256
from scripts import supervise_redaction_rescue_v8_retry1 as control


def pointer(path):return {'path':path.relative_to(control.ROOT).as_posix(),'sha256':sha256(path)}


def build():
 parent=control.read_json(control.LEGACY_SPEC)
 value=copy.deepcopy(parent)
 value.update(format='rescue-v8-external-single-GPU-supervision-retry1-plan-v1',
  parent_supervision_plan=pointer(control.LEGACY_SPEC),supervisor_source_sha256=sha256(Path(control.__file__)),
  preparation_source_sha256=sha256(Path(__file__)),
  revision_reason='Original separate preflight failed with ModuleNotFoundError; add only pinned missing project-environment dependencies and retry in new supervision namespaces. Frozen trainer, learning plan and training math unchanged.',
  failed_preflight={key:pointer(path) for key,path in control.failed_preflight_paths().items()},
  environment_evidence={name:pointer(control.ENV_ROOT/name) for name in control.ENV_NAMES},
  training_python_sha256=sha256(control.TRAINING_PYTHON),requires_explicit_GPU_phase_approval_flag=True,
  current_CPU_environment_probe_required_before_GPU=True,current_environment_package_map_must_equal_snapshot=True,
  CPU_environment_probe_timeout_seconds=60,exclusive_attempt_claim_precedes_CPU_probe=True,
  no_original_receipt_or_log_overwrites=True,original_failure_was_not_a_completed_GPU_preflight=True)
 extra=[control.LEGACY_SPEC,control.LEGACY_SOURCE,*control.failed_preflight_paths().values(),
  *(control.ENV_ROOT/name for name in control.ENV_NAMES),control.TRAINING_PYTHON,
  control.ROOT/'requirements-training.txt',control.ROOT/'requirements-training-lock-windows311.txt']
 value['input_sha256'].update({path.relative_to(control.ROOT).as_posix():sha256(path) for path in extra})
 control.validate_pins(value)
 return value


def prepare():
 if control.PUBLIC_SPEC.exists():raise ValueError('retry1_plan_already_exists_no_overwrite')
 value=build();raw=json.dumps(value,indent=2,allow_nan=False);control.PUBLIC_SPEC.parent.mkdir(parents=True,exist_ok=True)
 with control.PUBLIC_SPEC.open('x',encoding='utf-8') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 return {'supervision_plan_sha256':sha256(control.PUBLIC_SPEC),'supervisor_source_sha256':value['supervisor_source_sha256'],'GPU_operations':0,'training_updates':0}


if __name__=='__main__':
 try:print(json.dumps(prepare()))
 except (ValueError,OSError,KeyError,TypeError) as error:print(json.dumps({'status':'retry_plan_refused','error_type':type(error).__name__}));raise SystemExit(2)
