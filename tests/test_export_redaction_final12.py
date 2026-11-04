import ast
import copy
from pathlib import Path
import numpy as np
import pytest
from scripts import export_redaction_final12 as m


def completion():
    learning = {'source_expires_at': '2099-01-01T00:00:00+00:00'}
    receipt = {'phase': 'train', 'status': 'completed', 'child_exit_code': 0,
               'own_child_running_after_cleanup': False, 'sources_unchanged': True,
               'training_source_sha256': m.TRAINER_SHA, 'learning_plan_sha256': m.TRAIN_PLAN_SHA, 'TEST_images': 0}
    status = {'status': 'completed', 'epochs': 12, 'TEST_images': 0, 'family_eligible': True, **learning}
    recovery = {'completed_epoch': 12, 'plan_sha256': m.TRAIN_PLAN_SHA, 'automatic_resume_allowed': False}
    return receipt, status, recovery, learning


def payload():
    ema = {'face.w': np.array([1., -0.], np.float32), 'plate.w': np.array([2.], np.float32),
           'rescue_head.w': np.array([3.], np.float32)}
    return {'format': 'rescue-epoch-recovery-v1', 'plan_sha256': m.TRAIN_PLAN_SHA,
            'completed_epoch': 12, 'updates': 60, 'automatic_resume_allowed': False,
            'EMA': ema, 'model': {k: value+10 for k,value in ema.items()},
            'selection_history': [{'epoch': i, 'updates': 5*i} for i in range(13)]}


def test_complete_original_metadata_required():
    m.validate_completed(*completion())


@pytest.mark.parametrize('part,key,value', [(0,'status','running'), (0,'own_child_running_after_cleanup',True),
    (0,'sources_unchanged',False), (0,'child_exit_code',1), (0,'TEST_images',True),
    (1,'epochs',11), (1,'TEST_images',1), (2,'completed_epoch',11), (2,'automatic_resume_allowed',True)])
def test_incomplete_or_changed_original_rejected(part,key,value):
    values = list(completion()); values[part][key] = value
    with pytest.raises(ValueError):
        m.validate_completed(*values)


def test_uses_complete_final_EMA_not_live_model_or_AP_best():
    value = payload()
    result = m.validate_recovery(value, value['EMA'])
    assert result is value['EMA']
    assert not np.array_equal(result['rescue_head.w'], value['model']['rescue_head.w'])


@pytest.mark.parametrize('change', ['partial','extra','epoch','updates','history'])
def test_partial_or_nonfinal_recovery_rejected(change):
    value = payload(); expected = set(value['EMA'])
    if change == 'partial':value['EMA'].pop('plate.w')
    if change == 'extra':value['EMA']['unknown.w'] = np.array([4.])
    if change == 'epoch':value['completed_epoch'] = 11
    if change == 'updates':value['updates'] = 0
    if change == 'history':value['selection_history'] = value['selection_history'][:-1]
    with pytest.raises(ValueError):m.validate_recovery(value, expected)


def test_legacy_digest_includes_signed_zero_BN_and_preserves_new_head_independence():
    value = payload()['EMA']; original = m.state_digest(value, legacy_only=True)
    new_head = copy.deepcopy(value); new_head['rescue_head.w'] *= 2
    assert m.state_digest(new_head, legacy_only=True) == original
    changed = copy.deepcopy(value); changed['face.w'][1] = 0.
    assert m.state_digest(changed, legacy_only=True) != original
    changed = copy.deepcopy(value); changed['plate.running_mean'] = np.array([0.], np.float32)
    assert m.state_digest(changed, legacy_only=True) != original


def test_provenance_copy_keeps_TTL_parent_immutable_and_clears_parent_deletion_refs():
    parent = {'format':'training-provenance-v1','release_sha256':m.RELEASE_SHA,'test_split_exported':False,
              'data_revoked':False,'needs_revalidation':False,'models':['parent/model.json'],
              'assets':[{'sample_id':'synthetic','expires_at':'2099-01-01T00:00:00+00:00',
                         'files':['parent/image'],'source_files':['source/image'],'label_refs':['parent/labels'],'delete_refs':['parent/extra']}]}
    before = copy.deepcopy(parent)
    child = m.derived_provenance(parent, {'path':'parent/provenance.json','sha256':'a'*64}, 'child/model.json')
    assert parent == before
    assert child['models'] == ['child/model.json']
    assert child['assets'][0]['expires_at'] == parent['assets'][0]['expires_at']
    assert all(child['assets'][0][key] == [] for key in ['files','source_files','label_refs','delete_refs'])


def test_unapproved_export_refuses_before_any_file_hash_or_recovery_read(monkeypatch):
    monkeypatch.setattr(m,'sha256',lambda path: pytest.fail('unapproved export touched files'))
    with pytest.raises(ValueError,match='separate_explicit_CPU_export_approval'):
        m.export()


def test_source_CPU_only_lazy_torch_and_completion_precedes_actual_load():
    tree = ast.parse(Path(m.__file__).read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Import) and any(a.name == 'torch' for a in n.names) for n in tree.body)
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='export')
    calls = [n for n in ast.walk(function) if isinstance(n,ast.Call)]
    guard = min(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='validate_completed')
    loads = [n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='load' and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch']
    assert loads and all(line > guard for line in loads)
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr=='cuda' for n in calls)
