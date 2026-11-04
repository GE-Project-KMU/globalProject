"""Count-only correction contracts; no learned model or TEST photo access."""
import ast,json,os,copy
from pathlib import Path
import pytest
from scripts import compare_privacy_regression_count_retry1 as t


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8');return t.sha256(path)


def test_only_expected_count_changes_and_original_dict_is_untouched():
    old={'expected_samples':1400,'benchmark_sha256':t.BENCHMARK_SHA,'models':{'candidate':'same.json'},'raw_rtol':.001,'raw_atol':.0001,'test_used_for_fitting_or_selection':False}
    snapshot=copy.deepcopy(old);new=t.corrected_plan(old)
    assert old==snapshot and new==dict(old,expected_samples=600)
    assert {k for k in old if old[k]!=new[k]}=={'expected_samples'}

@pytest.mark.parametrize('count,sha',[(600,t.BENCHMARK_SHA),(1401,t.BENCHMARK_SHA),(1400,'0'*64)])
def test_other_parent_or_cohort_is_rejected(count,sha):
    with pytest.raises(ValueError):t.corrected_plan({'expected_samples':count,'benchmark_sha256':sha})


def test_length_mismatch_short_circuits_image_hash_validator():
    calls=[]
    def validator(rows):calls.append(1);pytest.fail('must not touch TEST photos')
    rows=[None]*600;expected=1400
    assert len(rows)!=expected or not validator(rows)['valid']
    assert calls==[]


def test_shared_scoring_scheduling_and_model_math_are_direct_original_functions():
    for name in ('ordered_frame','truth_rows','model_bindings','launch','validate_worker_output','claim_attempt'):
        assert getattr(t,name) is getattr(t.original,name)


def test_worker_prediction_loop_AST_is_unchanged():
    def loop(path):
        tree=ast.parse(Path(path).read_text(encoding='utf-8'));worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        return next(n for n in ast.walk(worker) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='row')
    assert ast.dump(loop(t.__file__),include_attributes=False)==ast.dump(loop(t.original.__file__),include_attributes=False)


def test_no_authorization_never_admits_or_claims(monkeypatch):
    monkeypatch.setattr(t,'admit',lambda *a:pytest.fail('must not admit'))
    with pytest.raises(ValueError,match='authorization'):t.run(False)


def test_direct311_required_before_actual_admission(monkeypatch):
    monkeypatch.setattr(t.sys,'version_info',(3,14,0));monkeypatch.setattr(t,'admit',lambda *a:pytest.fail('must not admit'))
    with pytest.raises(ValueError,match='Python311'):t.run(True)


def test_independent_attempt_is_exclusive_and_fsynced(tmp_path,monkeypatch):
    path=tmp_path/'new-attempt.json';calls=[];real=os.fsync
    monkeypatch.setattr(t.original.os,'fsync',lambda fd:(calls.append(fd),real(fd))[1])
    t.claim_attempt(path,{'status':'started'})
    assert calls
    with pytest.raises(FileExistsError):t.claim_attempt(path,{'status':'started'})

@pytest.mark.parametrize('change',['parent','old_attempt','failed','sha'])
def test_direct_or_stale_worker_cannot_bypass_new_attempt(tmp_path,monkeypatch,change):
    path=tmp_path/'new-attempt.json';monkeypatch.setattr(t,'ATTEMPT',path);monkeypatch.setattr(t.os,'getppid',lambda:10)
    value={'status':'started','writer':'two_process_CPU','plan_sha256':'a','parallel_protocol_sha256':'b','selection_allowed':False,'automatic_retry_allowed':False}
    if change=='failed':value['status']='failed'
    digest=write(path,value)
    execution={'parent_pid':10 if change!='parent' else 99,'comparison_role':'regression','attempt_path':str(path),'attempt_sha256':digest,'parent_plan_sha256':'a','parallel_protocol_sha256':'b'}
    if change=='old_attempt':execution['attempt_path']=str(tmp_path/'old-attempt.json');write(Path(execution['attempt_path']),value)
    if change=='sha':execution['attempt_sha256']='0'*64
    with pytest.raises(ValueError):t.assert_worker_attempt(execution)


def test_changed_corrected_plan_fails_before_attempt(tmp_path,monkeypatch):
    p=tmp_path/'plan.json';monkeypatch.setattr(t,'PLAN',p)
    original={'expected_samples':1400,'benchmark_sha256':t.BENCHMARK_SHA,'models':{'candidate':'same.json'}}
    write(p,dict(t.corrected_plan(original),models={'candidate':'other.json'}))
    monkeypatch.setattr(t,'inputs',lambda:({}, {}, {}, None,original));monkeypatch.setattr(t,'check_protocol',lambda:{})
    with pytest.raises(ValueError,match='only_original_expected_count'):t.admit()


def test_fresh_role_cannot_be_replayed():
    with pytest.raises(ValueError,match='fresh_preserved'):t.admit('fresh')


@pytest.fixture
def admission_case(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'DEST',tmp_path)
    rows=[{'sample_id':str(i),'sha256':str(i),'split':'test','evaluation_class':'face' if i<300 else 'license_plate'} for i in range(600)]
    benchmark=tmp_path/'benchmark.json';benchmark_sha=write(benchmark,rows);monkeypatch.setattr(t,'BENCHMARK_SHA',benchmark_sha)
    plan={'expected_samples':1400,'benchmark_sha256':benchmark_sha,'benchmark_manifest':str(benchmark)}
    parent=tmp_path/'regression-plan.json';monkeypatch.setattr(t,'ORIGINAL_PLAN_SHA',write(parent,plan))
    monkeypatch.setattr(t,'FAILED_SHA',write(tmp_path/'regression-attempt.json',{'status':'failed','error_type':'ValueError','automatic_retry_allowed':False}))
    boundary=tmp_path/'boundary.json';monkeypatch.setattr(t,'BOUNDARY',boundary)
    monkeypatch.setattr(t,'BOUNDARY_SHA',write(boundary,{'short_circuit_dataset_validator_calls':0,'TEST_image_hash_or_decode_calls_in_failed_phase':0,'model_worker_launches_in_failed_phase':0,'worker_directory_exists':False}))
    freshplan=tmp_path/'fresh-plan.json';fresh_sha=write(freshplan,{})
    write(tmp_path/'fresh-attempt.json',{'status':'completed','samples':200,'plan_sha256':fresh_sha});write(tmp_path/'fresh-comparison.json',{'samples':200,'plan_sha256':fresh_sha})
    monkeypatch.setattr(t.original,'admit',lambda role:({'worker_runtime':{},'worker_timeout_seconds':7200},{'source_expires_at':'2099-01-01T00:00:00Z'},{},parent,plan))
    monkeypatch.setattr(t,'model_bindings',lambda *a:({}, {}, []));monkeypatch.setattr(t,'truth_rows',lambda *a:[])
    return tmp_path,rows,benchmark


def test_exact_metadata_admission_does_not_create_output_or_attempt(admission_case):
    root,_,_=admission_case;t.inputs()
    assert not (root/'parallel-regression-count-retry1').exists()
    assert not (root/'regression-count-retry1-attempt.json').exists()

@pytest.mark.parametrize('change',['fresh_failed','old_worker_launched','changed_boundary','duplicate_cohort','missing_cohort'])
def test_retry_requires_successful_fresh_and_exact_unlaunched_count_failure(admission_case,change):
    root,rows,benchmark=admission_case
    if change=='fresh_failed':write(root/'fresh-attempt.json',{'status':'failed'})
    elif change=='old_worker_launched':(root/'parallel-regression-v10').mkdir()
    elif change=='changed_boundary':write(t.BOUNDARY,{'short_circuit_dataset_validator_calls':1})
    elif change=='duplicate_cohort':rows[1]=rows[0];write(benchmark,rows)
    else:write(benchmark,rows[:-1])
    with pytest.raises(ValueError):t.inputs()
