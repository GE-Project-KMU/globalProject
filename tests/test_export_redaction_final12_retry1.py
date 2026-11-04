import ast
import copy
from pathlib import Path
import numpy as np
import pytest
from scripts import export_redaction_final12_retry1 as m


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


def synthetic_development():
    from agent.governance import identities
    rows=[]
    for split,label,count in [('train','face',3000),('train','license_plate',1993),
                              ('validation','face',400),('validation','license_plate',108)]:
        for i in range(count):
            key=f'{split}-{label}-{i:04d}'
            rows.append({'split':split,'evaluation_class':label,'sample_id':key,'source_id':'src-'+key,
                         'path':'data/synthetic/'+key+'.jpg','sha256':key,'expires_at':'2099-01-01T00:00:00+00:00'})
    selected=[r for r in rows if (r['split']=='train' and r['evaluation_class']=='face') or r['split']=='validation']
    assets=[{'sample_id':r['sample_id'],'source_id':r['source_id'],'observation_id':None,
             'source_path':r['path'],'source_sha256':r['sha256'],'expires_at':r['expires_at'],'identity_hashes':identities(r)} for r in selected]
    parent={'release_sha256':m.RELEASE_SHA,'test_split_exported':False,'assets':assets}
    return rows,parent


def test_exact_original_trainer_AST_predicate_and3508_identity_membership():
    rows,parent=synthetic_development()
    source=m.ROOT/'scripts/redaction_rescue_data.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_development')
    statements=[n for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['train','validation'] for t in n.targets)]
    scope={'rows':rows}
    exec(compile(ast.Module(body=statements,type_ignores=[]),str(source),'exec'),scope)
    expected=sorted(scope['train'],key=lambda r:r['sample_id'])+sorted(scope['validation'],key=lambda r:r['sample_id'])
    actual,digest=m.original_training_membership(rows,parent,m.TRAIN_PLAN_SHA,m.TRAINER_SHA)
    assert len(rows)==5501 and len(actual)==3508 and actual==expected and len(digest)==64
    assert all(not(r['split']=='train' and r['evaluation_class']=='license_plate') for r in actual)
    trainer=ast.parse(m.TRAINER.read_text(encoding='utf-8'))
    calls=[n for n in ast.walk(trainer) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='begin_training']
    assert len(calls)==1 and ast.unparse(calls[0].args[2])=='trainrows + valrows'


@pytest.mark.parametrize('field',['source_id','observation_id','source_path','source_sha256','expires_at','identity_hashes'])
def test_same_count_wrong_source_identity_or_TTL_is_rejected(field):
    rows,parent=synthetic_development()
    parent['assets'][0][field]=['wrong'] if field=='identity_hashes' else 'wrong'
    with pytest.raises(ValueError,match='source_identity_or_TTL_mismatch'):
        m.original_training_membership(rows,parent,m.TRAIN_PLAN_SHA,m.TRAINER_SHA)


def test_same_count_unused_plate_train_substitution_rejected():
    rows,parent=synthetic_development()
    extra=next(r for r in rows if r['split']=='train' and r['evaluation_class']=='license_plate')
    parent['assets'][0]['sample_id']=extra['sample_id']
    with pytest.raises(ValueError,match='selected_sample_membership'):
        m.original_training_membership(rows,parent,m.TRAIN_PLAN_SHA,m.TRAINER_SHA)


def test_membership_source_plan_and_trainer_must_be_original():
    rows,parent=synthetic_development()
    with pytest.raises(ValueError,match='original_trainer_source_plan_binding'):
        m.original_training_membership(rows,parent,'wrong',m.TRAINER_SHA)
