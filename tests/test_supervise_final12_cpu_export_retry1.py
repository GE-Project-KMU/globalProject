import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import pytest
from scripts import supervise_final12_cpu_export_retry1 as m
REAL_CHILD_ABSENT = m.child_absent
REAL_OUTPUT_BINDING = m.output_binding


class FakeProcess:
    def __init__(self, code=0):
        self.pid = 12345
        self.returncode = code
        self.terminated = self.killed = False
    def poll(self):return self.returncode
    def terminate(self):self.terminated = True; self.returncode = -15
    def kill(self):self.killed = True; self.returncode = -9
    def wait(self,timeout=None):return self.returncode


class FakeOwned:
    def __init__(self, process):
        self.process=process; self.pid=process.pid; self.creation_time=123.5; self.parent_pid=6789
    def identity_matches(self):return self.process.poll() is None
    def stop(self):
        if self.process.poll() is not None:return False
        self.process.terminate(); return True


@pytest.fixture
def sandbox(monkeypatch):
    with tempfile.TemporaryDirectory(dir=m.ROOT/'.cache', prefix='final12-cpu-contract-') as folder:
        root=Path(folder); calls=[]; process=FakeProcess()
        context={'source_expires_at':'2099-01-01T00:00:00+00:00','training_completion_sha256':'a'*64,
                 'training_status_sha256':'b'*64,'recovery_metadata_sha256':'c'*64,'source_recovery_sha256':'d'*64}
        outputs={key:chr(101+i)*64 for i,key in enumerate(['export_receipt_sha256','candidate_freeze_sha256','manifest_sha256','checkpoint_sha256','onnx_sha256'])}
        monkeypatch.setattr(m,'RECEIPT',root/'receipt.json'); monkeypatch.setattr(m,'LOG',root/'export.log')
        monkeypatch.setattr(m,'upstream',lambda **kwargs: dict(context))
        monkeypatch.setattr(m,'validate_source_pins',lambda **kwargs: {})
        monkeypatch.setattr(m,'output_binding',lambda value: dict(outputs))
        monkeypatch.setattr(m,'child_absent',lambda value: None)
        monkeypatch.setattr(m,'environment',lambda: {'CUDA_VISIBLE_DEVICES':'0','synthetic':'true'})
        def popen(command,**kwargs):
            calls.append((command,kwargs)); return process
        monkeypatch.setattr(m.subprocess,'Popen',popen); monkeypatch.setattr(m,'OwnedChild',FakeOwned)
        monkeypatch.setattr(m.time,'sleep',lambda delay: None)
        yield root,calls,process,context,outputs


def saved():return json.loads(m.RECEIPT.read_text())


def test_success_fixed_CPU_command_and_bindings(sandbox):
    _,calls,_,context,outputs=sandbox
    assert m.run(True) is True
    result=saved()
    assert result['status']=='completed' and result['child_exit_code']==0
    assert result['own_child_running_after_cleanup'] is False
    assert result['own_child_identity_acquired'] is True
    assert all(result[k]==v for k,v in context.items())
    assert all(result[k]==v for k,v in outputs.items())
    command,kwargs=calls[0]
    assert command == [str(m.ROOT/'.venv-training/python.exe'),str(m.EXPORTER),'export','--cpu-export-approved','--expected-plan-sha256',m.POLICY_SHA]
    assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and kwargs['cwd']==m.ROOT
    assert m.completed_execution()['status']=='completed'


def test_repeated_execution_does_not_change_receipt_or_launch_again(sandbox):
    _,calls,_,_,_=sandbox
    m.run(True); before=m.RECEIPT.read_bytes()
    with pytest.raises(FileExistsError):m.run(True)
    assert m.RECEIPT.read_bytes()==before and len(calls)==1


def test_no_approval_no_claim_no_Popen(sandbox):
    _,calls,_,_,_=sandbox
    with pytest.raises(ValueError,match='explicit_root_CPU_export_approval'):
        m.run(False)
    assert not m.RECEIPT.exists() and not calls


def test_upstream_failure_records_failed_without_Popen(sandbox,monkeypatch):
    _,calls,_,_,_=sandbox
    def bad(**kwargs):raise ValueError('training_not_completed12')
    monkeypatch.setattr(m,'upstream',bad)
    assert m.run(True) is False
    assert saved()['status']=='failed' and 'training_not_completed12' in saved()['failure_reason']
    assert not calls


def test_existing_log_preserved_and_failure_recorded(sandbox):
    _,calls,_,_,_=sandbox
    m.LOG.write_text('original')
    assert m.run(True) is False
    assert saved()['status']=='failed' and m.LOG.read_text()=='original' and not calls


def test_nonzero_child_always_records_failure_and_exit(sandbox):
    _,_,process,_,_=sandbox;process.returncode=7
    assert m.run(True) is False
    assert saved()['status']=='failed' and saved()['child_exit_code']==7
    assert not process.terminated


def test_Popen_exception_is_recorded(sandbox,monkeypatch):
    def fail(*args,**kwargs):raise OSError('synthetic_spawn_failure')
    monkeypatch.setattr(m.subprocess,'Popen',fail)
    assert m.run(True) is False
    assert saved()['status']=='failed' and 'synthetic_spawn_failure' in saved()['failure_reason']
    assert saved()['own_child_running_after_cleanup'] is False


def test_timeout_stops_owned_identity_only(sandbox,monkeypatch):
    _,_,process,_,_=sandbox;process.returncode=None
    monkeypatch.setattr(m,'TIMEOUT_SECONDS',0)
    assert m.run(True) is False
    assert process.terminated and saved()['own_child_stopped_after_identity_check'] is True
    assert saved()['child_exit_code']==-15 and saved()['own_child_running_after_cleanup'] is False


def test_expired_source_refused_before_Popen(sandbox):
    _,calls,_,context,_=sandbox;context['source_expires_at']='2000-01-01T00:00:00+00:00'
    assert m.run(True) is False
    assert saved()['status']=='failed' and not calls


def test_identity_acquisition_failure_only_stops_exact_created_handle(sandbox,monkeypatch):
    _,_,process,_,_=sandbox;process.returncode=None
    def refused(value):raise ValueError('identity_not_acquired')
    monkeypatch.setattr(m,'OwnedChild',refused)
    assert m.run(True) is False
    assert process.terminated and saved()['own_child_identity_acquired'] is False
    assert saved()['created_Popen_handle_stopped_before_identity_acquired'] is True


def test_source_change_cannot_be_completed_even_if_export_exit0(sandbox,monkeypatch):
    def changed(**kwargs):raise ValueError('changed')
    monkeypatch.setattr(m,'validate_source_pins',changed)
    assert m.run(True) is False
    assert saved()['status']=='failed' and saved()['sources_unchanged'] is False


def test_output_binding_failure_preserves_failed_status(sandbox,monkeypatch):
    def mismatch(value):raise ValueError('export_receipt_mismatch')
    monkeypatch.setattr(m,'output_binding',mismatch)
    assert m.run(True) is False
    assert saved()['status']=='failed' and 'export_receipt_mismatch' in saved()['failure_reason']


@pytest.mark.parametrize('status',['starting','running'])
def test_queue_waits_for_absent_or_running(sandbox,status):
    with pytest.raises(m.ExecutionNotReady):m.completed_execution()
    m.RECEIPT.write_text(json.dumps({'status':status}))
    with pytest.raises(m.ExecutionNotReady):m.completed_execution()


def test_queue_stops_on_failed_receipt(sandbox):
    m.RECEIPT.write_text(json.dumps({'status':'failed'}))
    with pytest.raises(ValueError,match='root_CPU_export_failed_or_changed'):m.completed_execution()


def test_completed_queue_revalidation_does_not_reapply_preTEST_start_gate(sandbox,monkeypatch):
    m.run(True);calls=[]
    context=sandbox[3]
    def check(**kwargs):calls.append(kwargs);return dict(context)
    monkeypatch.setattr(m,'upstream',check)
    assert m.completed_execution()['status']=='completed'
    assert calls==[{'check_preTEST':False}]


def test_no_torch_or_GPU_calls_in_wrapper_source():
    tree=ast.parse(Path(m.__file__).read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Import) and any(a.name in ['torch','onnxruntime','cv2'] for a in n.names) for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Attribute) and n.attr=='cuda' for n in ast.walk(tree))


@pytest.mark.parametrize('creation,parent,expected', [(12.,44,None),(11.,44,'wait'),(11.,99,'parent')])
def test_exact_finished_child_identity_not_reused_PID(monkeypatch,creation,parent,expected):
    class Info:
        def create_time(self):return creation
        def ppid(self):return parent
        def is_running(self):return True
    monkeypatch.setattr(m.psutil,'Process',lambda pid: Info())
    value={'own_child_pid':1234,'own_child_creation_time':11.,'parent_pid':44}
    if expected=='wait':
        with pytest.raises(m.ExecutionNotReady):REAL_CHILD_ABSENT(value)
    elif expected=='parent':
        with pytest.raises(ValueError,match='parent_changed'):REAL_CHILD_ABSENT(value)
    else:REAL_CHILD_ABSENT(value)


@pytest.mark.parametrize('tamper',[None,'freeze','legacy','recovery'])
def test_actual_output_binding_contract_with_synthetic_metadata_only(monkeypatch,tamper):
    hashes={'export-receipt.json':'a'*64,'candidate-freeze.json':'b'*64,'model.json':'c'*64,'best.pt':'d'*64,'model.onnx':'e'*64,'metadata-membership-audit.json':'1'*64}
    context={'training_completion_sha256':'f'*64,'source_recovery_sha256':'0'*64,'source_expires_at':'2099-01-01T00:00:00+00:00','parent_export_plan_sha256':'2'*64}
    actual={'manifest_sha256':hashes['model.json'],'checkpoint_sha256':hashes['best.pt'],'onnx_sha256':hashes['model.onnx']}
    exported={'status':'completed','epoch':12,'EMA':True,'training_complete':True,'TEST_images':0,
              'source_plan_sha256':m.TRAIN_PLAN_SHA,'export_plan_sha256':m.POLICY_SHA,'export_source_sha256':m.EXPORT_SHA,
              'GPU_used':False,'CUDA_initialized':False,'parent_inputs_unchanged':True,'legacy_raw_prefix_matches_frozen_v4':True,
              'parent_training_supervisor_sha256':context['training_completion_sha256'],
              'source_recovery_sha256':context['source_recovery_sha256'],'source_expires_at':context['source_expires_at'],
              'original_trainer_subset':{'train_face':3000,'validation_face':400,'validation_plate':108,'total':3508},
              'same_original_final12_EMA':True,'parent_export_policy_sha256':context['parent_export_plan_sha256'],
              'membership_audit_sha256':hashes['metadata-membership-audit.json'],
              'candidate_freeze_sha256':hashes['candidate-freeze.json'],**actual}
    freeze={'epoch':12,'fixed_final12_EMA':True,'training_complete':True,'TEST_images':0,
            'export_plan_sha256':m.POLICY_SHA,'export_source_sha256':m.EXPORT_SHA,**actual}
    manifest={'sha256':actual['onnx_sha256'],'trained_checkpoint_sha256':actual['checkpoint_sha256'],
              'weights':(m.OUT/'model.onnx').relative_to(m.ROOT).as_posix(),'research_only':True,'production_approved':False}
    if tamper=='freeze':freeze['checkpoint_sha256']='9'*64
    if tamper=='legacy':exported['legacy_raw_prefix_matches_frozen_v4']=False
    if tamper=='recovery':exported['source_recovery_sha256']='9'*64
    data={'export-receipt.json':exported,'candidate-freeze.json':freeze,'model.json':manifest}
    monkeypatch.setattr(m,'read',lambda path: data[Path(path).name])
    monkeypatch.setattr(m,'sha256',lambda path: hashes[Path(path).name])
    if tamper:
        with pytest.raises(ValueError):REAL_OUTPUT_BINDING(context)
    else:
        value=REAL_OUTPUT_BINDING(context)
        assert value['export_receipt_sha256']==hashes['export-receipt.json']
        assert all(value[k]==v for k,v in actual.items())
