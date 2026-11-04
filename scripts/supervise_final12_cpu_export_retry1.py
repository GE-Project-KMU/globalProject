"""Root-only fixed-command CPU export supervisor; no GPU or automatic retries."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.paths import atomic_json, workspace_path, sha256
from scripts.cycle2_device_memory import OwnedChild
from scripts.setup_training import environment

OUT = ROOT/'data/training/redaction-rescue-v8-final12-retry1'
PARENT = ROOT/'data/training/redaction-rescue-v8'
RECEIPT = OUT/'root-export-execution.json'
LOG = ROOT/'.cache/cycle2-final12-cpu-export-retry1.log'
EXPORTER = ROOT/'scripts/export_redaction_final12_retry1.py'
POLICY = ROOT/'docs/reports/PRIVACY-RESCUE-V8-FINAL12-RETRY1-EXPORT-PLAN.json'
EXPORT_SHA = '76513ae98a97dc9e6ae49d8054635f01c9683b3503435c6e54653e475acd612f'
POLICY_SHA = '56d58768fd4814ce6e46f0b00e069230e4ad3d564a0ef93d0d602c498415c051'
TRAIN_SHA = '7bc192db04ff79ff0cad9c3b440aa8357c99f387eecf4a7093ce3f59d9475f65'
TRAIN_PLAN_SHA = '6bbd59191a2c352a6a2bf7dc0fbd15f25ced0f500be6010fa3011a4e0f94164d'
SUPERVISOR_SHA = '00a60af75bab8f145189e0b0502dc560ca25d4988fa8ebf8a97b03650ba48a4a'
SUPERVISION_SHA = 'cb5092c1a2726d4038d0fc51af1d4e7c3bb1d30ac81416fb7923c3b420ba3cd1'
PREDECLARATION_SHA = '2717f0a737aa73d72afca8cc567519d0630c3ae2ebe77b980d608a9018228117'
PINS = {
    'scripts/export_redaction_final12_retry1.py': EXPORT_SHA,
    'docs/reports/PRIVACY-RESCUE-V8-FINAL12-RETRY1-EXPORT-PLAN.json': POLICY_SHA,
    'data/training/redaction-rescue-v8-final12-retry1/plan.json': POLICY_SHA,
    'docs/reports/PRIVACY-FAMILY-V9-PREDECLARATION.json': PREDECLARATION_SHA,
    'scripts/train_redaction_rescue_v8.py': TRAIN_SHA,
    'data/training/redaction-rescue-v8/plan.json': TRAIN_PLAN_SHA,
    'docs/reports/PRIVACY-RESCUE-V8-PLAN.json': TRAIN_PLAN_SHA,
    'scripts/supervise_redaction_rescue_v8_retry1.py': SUPERVISOR_SHA,
    'docs/reports/PRIVACY-RESCUE-V8-RETRY1-SUPERVISION-PLAN.json': SUPERVISION_SHA,
}
TIMEOUT_SECONDS = 900


class ExecutionNotReady(ValueError):
    pass


def read(path):
    return json.loads(workspace_path(path, exists=True).read_text(encoding='utf-8'))


def child_absent(value):
    pid, created, parent = (value.get(k) for k in ['own_child_pid','own_child_creation_time','parent_pid'])
    if (type(pid) is not int or pid <= 0 or type(parent) is not int or parent <= 0
            or type(created) not in (int,float) or not math.isfinite(created) or created <= 0):
        raise ValueError('exact_child_identity_fields_required')
    try:
        process = psutil.Process(pid)
        if process.create_time() == created and process.is_running():
            if process.ppid() != parent:
                raise ValueError('matching_child_creation_time_but_parent_changed')
            raise ExecutionNotReady('exact_child_identity_still_exists')
    except psutil.NoSuchProcess:
        pass


def deadline(value):
    result = datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None or result <= datetime.now(timezone.utc):
        raise ValueError('original_source_expired_or_naive')
    return result


def validate_source_pins(check_deadline=True):
    for relative,digest in PINS.items():
        if sha256(workspace_path(ROOT/relative, exists=True)) != digest:
            raise ValueError('fixed_export_training_or_predeclaration_pin_changed')
    policy = read(POLICY)
    for relative,digest in policy['source_sha256'].items():
        if sha256(workspace_path(ROOT/relative, exists=True)) != digest:
            raise ValueError('frozen_export_dependency_changed')
    if check_deadline:
        deadline(policy['source_expires_at'])
    return policy


def upstream(check_preTEST=True):
    policy = validate_source_pins()
    from scripts import export_redaction_final12_retry1 as exporter
    if check_preTEST:
        exporter.ensure_preTEST()
    receipt_path = PARENT/'external-train-supervisor-retry1.json'
    status_path, recovery_path = PARENT/'training-status.json', PARENT/'recovery-receipt.json'
    receipt, status, recovery = read(receipt_path), read(status_path), read(recovery_path)
    learning = read(PARENT/'plan.json')
    exporter.validate_completed(receipt, status, recovery, learning)
    if (receipt.get('format') != 'rescue-v8-external-supervision-retry1-v1'
            or receipt.get('supervisor_source_sha256') != SUPERVISOR_SHA
            or receipt.get('supervision_plan_sha256') != SUPERVISION_SHA
            or receipt.get('trainer_report_sha256') != sha256(status_path)
            or receipt.get('observer_child_stopped') is not True or receipt.get('observer_thread_stopped') is not True
            or policy['source_expires_at'] != learning['source_expires_at']):
        raise ValueError('exact_original_completed_training_binding_required')
    child_absent(receipt)
    if recovery['checkpoint_sha256'] != exporter.SOURCE_RECOVERY_SHA:
        raise ValueError('same_original_fixed12_recovery_required')
    failed_path=exporter.OLD_OUT/'root-export-execution.json'
    return {'source_expires_at': policy['source_expires_at'],
            'parent_export_plan_sha256':exporter.OLD_POLICY_SHA,
            'parent_failed_execution_sha256':sha256(failed_path),
            'training_completion_sha256': sha256(receipt_path), 'training_status_sha256': sha256(status_path),
            'recovery_metadata_sha256': sha256(recovery_path), 'source_recovery_sha256': recovery['checkpoint_sha256']}


def output_binding(context):
    exported_path, freeze_path = OUT/'export-receipt.json', OUT/'candidate-freeze.json'
    exported, freeze = read(exported_path), read(freeze_path)
    if (exported.get('status') != 'completed' or exported.get('epoch') != 12 or exported.get('EMA') is not True
            or exported.get('training_complete') is not True or type(exported.get('TEST_images')) is not int or exported['TEST_images'] != 0
            or exported.get('source_plan_sha256') != TRAIN_PLAN_SHA or exported.get('export_plan_sha256') != POLICY_SHA
            or exported.get('export_source_sha256') != EXPORT_SHA or exported.get('GPU_used') is not False
            or exported.get('CUDA_initialized') is not False or exported.get('parent_inputs_unchanged') is not True
            or exported.get('legacy_raw_prefix_matches_frozen_v4') is not True
            or exported.get('parent_training_supervisor_sha256') != context['training_completion_sha256']
            or exported.get('source_recovery_sha256') != context['source_recovery_sha256']
            or exported.get('source_expires_at') != context['source_expires_at']
            or exported.get('candidate_freeze_sha256') != sha256(freeze_path)):
        raise ValueError('completed_CPU_export_receipt_binding_required')
    if (exported.get('original_trainer_subset') != {'train_face':3000,'validation_face':400,'validation_plate':108,'total':3508}
            or exported.get('same_original_final12_EMA') is not True
            or exported.get('parent_export_policy_sha256') != context['parent_export_plan_sha256']
            or exported.get('membership_audit_sha256') != sha256(OUT/'metadata-membership-audit.json')):
        raise ValueError('exact_original3508_provenance_retry_binding_required')
    actual = {'manifest_sha256': sha256(OUT/'model.json'), 'checkpoint_sha256': sha256(OUT/'best.pt'), 'onnx_sha256': sha256(OUT/'model.onnx')}
    if any(exported.get(key) != value or freeze.get(key) != value for key,value in actual.items()):
        raise ValueError('actual_export_artifact_hashes_required')
    if (freeze.get('epoch') != 12 or freeze.get('fixed_final12_EMA') is not True or freeze.get('training_complete') is not True
            or freeze.get('TEST_images') != 0 or freeze.get('export_plan_sha256') != POLICY_SHA
            or freeze.get('export_source_sha256') != EXPORT_SHA):
        raise ValueError('same_fixed12_candidate_freeze_required')
    manifest = read(OUT/'model.json')
    if (manifest.get('sha256') != actual['onnx_sha256'] or manifest.get('trained_checkpoint_sha256') != actual['checkpoint_sha256']
            or manifest.get('weights') != (OUT/'model.onnx').relative_to(ROOT).as_posix()
            or manifest.get('research_only') is not True or manifest.get('production_approved') is not False):
        raise ValueError('same_research_manifest_and_checkpoint_required')
    return {'export_receipt_sha256': sha256(exported_path), 'candidate_freeze_sha256': sha256(freeze_path), **actual}


def completed_execution():
    """Read-only queue admission: missing/running waits; failed/mismatched stops."""
    if not RECEIPT.exists():
        raise ExecutionNotReady('root_CPU_export_receipt_not_created')
    value = read(RECEIPT)
    if value.get('status') in ('starting','running'):
        raise ExecutionNotReady('root_CPU_export_still_running')
    if (value.get('status') != 'completed' or type(value.get('child_exit_code')) is not int or value['child_exit_code'] != 0
            or value.get('own_child_running_after_cleanup') is not False or value.get('sources_unchanged') is not True
            or value.get('own_child_identity_acquired') is not True
            or value.get('parent_artifacts_preserved_verified') is not True
            or value.get('CPU_export_only') is not True or value.get('GPU_used') is not False
            or type(value.get('TEST_images')) is not int or value['TEST_images'] != 0
            or value.get('wrapper_source_sha256') != sha256(Path(__file__))
            or value.get('export_source_sha256') != EXPORT_SHA or value.get('export_plan_sha256') != POLICY_SHA
            or value.get('training_source_sha256') != TRAIN_SHA or value.get('source_plan_sha256') != TRAIN_PLAN_SHA
            or value.get('upstream_supervisor_source_sha256') != SUPERVISOR_SHA
            or value.get('upstream_supervision_plan_sha256') != SUPERVISION_SHA
            or value.get('parent_predeclaration_sha256') != PREDECLARATION_SHA):
        raise ValueError('root_CPU_export_failed_or_changed')
    child_absent(value)
    context = upstream(check_preTEST=False)
    if any(value.get(key) != expected for key,expected in context.items()):
        raise ValueError('upstream_training_changed_after_root_export')
    actual = output_binding(context)
    if any(value.get(key) != expected for key,expected in actual.items()):
        raise ValueError('root_execution_output_binding_changed')
    return value


def run(approved=False):
    if approved is not True:
        raise ValueError('explicit_root_CPU_export_approval_required')
    path, log_path = workspace_path(RECEIPT), workspace_path(LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {'format': 'fixed12-root-CPU-export-supervision-retry1-v1', 'status': 'starting',
              'wrapper_source_sha256': sha256(Path(__file__)), 'export_source_sha256': EXPORT_SHA, 'export_plan_sha256': POLICY_SHA,
              'training_source_sha256': TRAIN_SHA, 'source_plan_sha256': TRAIN_PLAN_SHA,
              'upstream_supervisor_source_sha256': SUPERVISOR_SHA, 'upstream_supervision_plan_sha256': SUPERVISION_SHA,
              'parent_predeclaration_sha256': PREDECLARATION_SHA,
              'CPU_export_only': True, 'GPU_used': False, 'TEST_images': 0, 'timeout_seconds': TIMEOUT_SECONDS,
              'owns_export_child_only': True, 'automatic_retry_allowed': False, 'parent_artifact_modification_permitted': False,
              'parent_artifacts_preserved_verified': False,
              'own_child_identity_acquired': False}
    # Exclusive claim: a second invocation never modifies the first receipt.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream); stream.flush(); os.fsync(stream.fileno())
    process = owned = log = None
    start = time.monotonic()
    try:
        if log_path.exists():
            raise ValueError('fresh_export_log_required_no_overwrite')
        context = upstream(); report.update(context)
        expires = deadline(context['source_expires_at'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open('x', encoding='utf-8')
        env = environment(); env['CUDA_VISIBLE_DEVICES'] = ''
        command = [str(ROOT/'.venv-training/python.exe'), str(EXPORTER), 'export', '--cpu-export-approved', '--expected-plan-sha256', POLICY_SHA]
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log)
        owned = OwnedChild(process)
        report.update(status='running', own_child_identity_acquired=True,
                      own_child_pid=owned.pid, own_child_creation_time=owned.creation_time, parent_pid=owned.parent_pid)
        atomic_json(path, report)
        while process.poll() is None:
            if not owned.identity_matches():
                raise ValueError('owned_export_child_identity_changed')
            if time.monotonic()-start >= TIMEOUT_SECONDS:
                raise TimeoutError('fixed_CPU_export_900second_timeout')
            if datetime.now(timezone.utc) >= expires:
                raise ValueError('original_source_deadline_during_CPU_export')
            time.sleep(.1)
        if process.returncode != 0:
            raise ValueError('CPU_export_child_nonzero_exit')
        if time.monotonic()-start > TIMEOUT_SECONDS:
            raise TimeoutError('fixed_CPU_export_900second_timeout_at_exit')
        deadline(context['source_expires_at']); child_absent(report)
        if upstream() != context:
            raise ValueError('original_upstream_artifacts_changed_during_export')
        report.update(output_binding(context)); report['status'] = 'completed'; report['parent_artifacts_preserved_verified'] = True
    except BaseException as error:
        report.update(status='failed', failure_reason=type(error).__name__+':'+str(error))
        if owned is not None:
            try:
                report['own_child_stopped_after_identity_check'] = owned.stop()
            except BaseException as stop_error:
                report['cleanup_error'] = type(stop_error).__name__
        elif process is not None and process.poll() is None:
            # Exact handle returned by our Popen only, never a PID lookup or
            # search. Failed identity acquisition always leaves status failed.
            try:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
                report['created_Popen_handle_stopped_before_identity_acquired'] = True
            except BaseException as stop_error:
                report['cleanup_error'] = type(stop_error).__name__
    finally:
        if log is not None:
            log.close()
        report['wall_seconds'] = time.monotonic()-start
        report['child_exit_code'] = process.poll() if process is not None else None
        report['own_child_running_after_cleanup'] = process is not None and process.poll() is None
        try:
            validate_source_pins(check_deadline=False)
            report['sources_unchanged'] = sha256(Path(__file__)) == report['wrapper_source_sha256']
        except BaseException:
            report['sources_unchanged'] = False
        if not report['sources_unchanged'] or report['own_child_running_after_cleanup']:
            report['status'] = 'failed'
        atomic_json(path, report)
    print(json.dumps(report), flush=True)
    return report['status'] == 'completed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-export-approved', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.cpu_export_approved) else 1)
