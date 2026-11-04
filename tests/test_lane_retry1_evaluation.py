"""Synthetic policy/full179 coupons and attempts; no TEST image inference."""
import hashlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
from scripts import evaluate_lane_retry1 as e
from scripts import gpu_model_set as g
from test_final_gpu_model_set_preparation import case as gpu_case


@pytest.fixture
def case(gpu_case,tmp_path,monkeypatch):
    folder,_,_,request,rows,save=gpu_case;entry=request['models']['lane'];candidate=folder/'lane-model.json'
    monkeypatch.setattr(e,'CANDIDATE',candidate);monkeypatch.setattr(e,'DEST',tmp_path/'retry-eval')
    monkeypatch.setattr(e,'POLICY',folder/'new-policy.json');monkeypatch.setattr(e,'ORIGINAL_DEST',tmp_path/'old-eval')
    monkeypatch.setattr(e,'NUMERIC',folder/'numeric-policy.json');monkeypatch.setattr(e,'CASES',folder/'positive-cases.json')
    monkeypatch.setattr(e,'LEARNING',folder/'retry-learning.json');monkeypatch.setattr(e,'SUPERVISION',folder/'supervision.json')
    monkeypatch.setattr(e,'PARENT_LEARNING',folder/'original-learning.json')
    monkeypatch.setattr(e,'SUPERVISOR_RECEIPT',folder/'supervisor-receipt.json')
    monkeypatch.setattr(e,'PARENTS',{role:folder/(role+'-parent.json') for role in e.COUNTS})
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return g.pointer(path)
    for row in rows['lane']:row['leakage_group_id']='VAL-recording-'+row['recording']
    release=put(g.workspace_path(entry['full_release']['path']),rows['lane'])
    entry['full_release']=release;entry['development_manifest']=release
    provenance=g.pinned(entry['training_provenance']);provenance['release_sha256']=release['sha256']
    entry['training_provenance']=put(g.workspace_path(entry['training_provenance']['path']),provenance)
    checkpoint=candidate.parent/'lane_best.pt';checkpoint.write_bytes(g.workspace_path(entry['checkpoint']['path']).read_bytes())
    calibration=put(folder/'best-validation.json',{'epoch':1,'selected':{'paint_threshold':.5}})
    model=g.pinned(entry['manifest']);model.update(paint_threshold=.5,best_epoch=1,
        dataset_release='manifest-sha256:'+release['sha256'],
        training_development_release_path=entry['development_manifest']['path'],
        training_development_release_sha256=entry['development_manifest']['sha256'],
        threshold_calibration={'selected_on':'validation','report_sha256':calibration['sha256'],'validation_samples':179,'test_images_used':0})
    manifest=put(candidate,model)
    positives=[]
    for index,ref in enumerate(entry['cpu_verifications']):
        proof=g.pinned(ref);proof['manifest_sha256']=manifest['sha256']
        name='verification-final.json' if index==0 else f'verification-val-{index}.json'
        put(folder/name,proof);row=next(row for row in rows['lane'] if row['sha256']==proof['image_sha256'])
        positives.append({key:row[key] for key in ('path','sha256','expires_at','source_id')})
    put(e.CASES,positives)
    fullplan=g.pinned(entry['full_validation_plan']);fullplan.update(manifest_sha256=manifest['sha256'],development_sha256=release['sha256'])
    planref=put(folder/'full-validation-131-plan.json',fullplan)
    full=g.pinned(entry['full_validation']);full.update(manifest_sha256=manifest['sha256'],plan_sha256=planref['sha256'])
    put(folder/'full-validation-131.json',full)
    put(folder/'history.json',{'history':[{'epoch':i} for i in range(1,7)],'best_epoch':1,'test_used':False})
    baseline=folder/'base/lane.json';baseline_model=dict(model);baseline_model['threshold_calibration']={'selected_on':'validation'}
    base_ref=put(baseline,baseline_model)
    proof=g.read(folder/'verification-final.json');proof['manifest_sha256']=base_ref['sha256'];put(baseline.parent/'verification-final.json',proof)
    put(e.PARENT_LEARNING,{'validation_count':179,'pre_training_original_plan':True})
    put(e.LEARNING,{'validation_count':179,'optimizer_state_resumed':False,'data_unchanged_from_parent':True,'parent_plan_sha256':g.sha256(e.PARENT_LEARNING),
        'release_sha256':entry['full_release']['sha256'],'development_sha256':entry['development_manifest']['sha256'],
        'training_code_sha256':g.sha256(e.ROOT/'scripts/train_lane_cycle2_retry1.py')})
    put(e.SUPERVISION,{'supervisor_source_sha256':g.sha256(e.ROOT/'scripts/supervise_lane_cycle2_retry1.py'),
        'learning_plan_sha256':g.sha256(e.LEARNING),'training_source_sha256':g.read(e.LEARNING)['training_code_sha256']})
    put(e.SUPERVISOR_RECEIPT,{'format':'lane-exclusive-retry1-training-supervision-v1','status':'completed','child_exit_code':0,
        'sources_unchanged':True,'own_child_running_after_cleanup':False,'learning_plan_sha256':g.sha256(e.LEARNING),
        'supervision_plan_sha256':g.sha256(e.SUPERVISION),'learning_source_sha256':g.read(e.LEARNING)['training_code_sha256'],
        'supervisor_source_sha256':g.read(e.SUPERVISION)['supervisor_source_sha256']})
    numeric={'task':'lane','positive_VAL_cases':3,'all_lane_VAL_count':179,'raw_rtol':.001,'raw_atol':.0001,
        'exact_postprocess_required':True,'CPU_ORT':'1.31.1','TEST_used':False,'positive_cases_sha256':g.sha256(e.CASES)}
    put(e.NUMERIC,numeric)
    for role,count in e.COUNTS.items():
        benchmark=[{'split':'test','sample_id':f'{role}-{i}','source_id':f'{role}-{i}',
            'sha256':hashlib.sha256(f'SYNTHETIC:{role}:{i}'.encode()).hexdigest(),
            'leakage_group_id':f'{role}-recording','expires_at':rows['lane'][0]['expires_at']} for i in range(count)]
        bench=put(folder/(role+'-manifest.json'),benchmark)
        put(e.PARENTS[role],{'created_before_final_predictions':True,
            'models':{'baseline':base_ref['path'],'candidate':'data/training/lane-cycle2-v5/lane.json'},
            'baseline_manifest_sha256':base_ref['sha256'],'candidate_release_sha256':entry['full_release']['sha256'],
            'benchmark_manifest':bench['path'],'benchmark_sha256':bench['sha256'],'numeric_plan_sha256':g.sha256(e.NUMERIC),
            'raw_rtol':.001,'raw_atol':.0001,'test_used_for_fitting_or_selection':False,'role':role})
    return folder,put


def test_policy_creation_needs_no_unfinished_candidate_and_preserves_parent_hashes(case,monkeypatch):
    _,_=case;real=e.read
    def read(path):
        assert g.workspace_path(path)!=e.CANDIDATE,'unfinished candidate read during policy creation'
        return real(path)
    monkeypatch.setattr(e,'read',read)
    result=e.prepare_policy();policy=g.read(e.POLICY)
    assert result['candidate_artifact_read'] is False and policy['cohort_samples']=={'fresh':119,'regression':180}
    assert policy['candidate_hash_bound_in_separate_receipt_after_completion'] is True
    assert 'candidate_manifest_sha256' not in policy
    assert policy['parent_comparison_policies']=={role:g.pointer(path) for role,path in e.PARENTS.items()}


def test_all_strict_metadata_freeze_and_plan_preparation_read_no_TEST_images(case,monkeypatch):
    e.prepare_policy()
    import cv2
    monkeypatch.setattr(cv2,'imread',lambda *args,**kwargs:pytest.fail('image read during metadata preparation'))
    e.freeze();result=e.prepare_plans();assert result['TEST_images_read']==0
    freeze=g.read(e.DEST/'candidate-freeze.json');assert len(freeze['canonical_proofs'])==3
    for role in e.COUNTS:
        plan=g.read(e.DEST/(role+'-plan.json'));parent=g.read(e.PARENTS[role])
        assert plan['benchmark_sha256']==parent['benchmark_sha256'] and plan['models']['candidate']==g.pointer(e.CANDIDATE)['path']
        assert plan['parent_original_policy']==g.pointer(e.PARENTS[role])
    with pytest.raises(ValueError,match='already_frozen'):e.freeze()


@pytest.mark.parametrize('problem',['training_running','nonzero_exit','source_changed','canonical_raw','canonical_checkpoint','full_raw','full_mask','full_index','full_image','full_source','full_plan','calibration','provenance'])
def test_strict_failures_refuse_freeze_before_TEST(case,problem):
    folder,put=case;e.prepare_policy()
    if problem in ('training_running','nonzero_exit','source_changed'):
        value=g.read(e.SUPERVISOR_RECEIPT)
        value[{'training_running':'status','nonzero_exit':'child_exit_code','source_changed':'learning_source_sha256'}[problem]]= 'running' if problem=='training_running' else 1 if problem=='nonzero_exit' else '0'*64
        put(e.SUPERVISOR_RECEIPT,value)
    elif problem.startswith('canonical'):
        path=folder/'verification-final.json';value=g.read(path)
        if problem=='canonical_raw':value['app_vs_torch']['passed']=False
        else:value['checkpoint_sha256']='0'*64
        put(path,value)
    elif problem.startswith('full_'):
        path=folder/'full-validation-131.json';value=g.read(path)
        if problem=='full_raw':value['cases'][17]['raw']['app_vs_original_FP32_torch']['passed']=False
        elif problem=='full_mask':value['cases'][17]['post']['torch']['mismatching_pixels']=1
        elif problem=='full_index':value['cases'][17]['index']=16
        elif problem=='full_image':value['cases'][17]['image_sha256']='0'*64
        elif problem=='full_source':value['worker_source_sha256']='0'*64
        else:value['plan_sha256']='0'*64
        put(path,value)
    elif problem=='calibration':
        value=g.read(folder/'best-validation.json');value['selected']['paint_threshold']=.6;put(folder/'best-validation.json',value)
    else:
        model=g.read(e.CANDIDATE);path=g.workspace_path(model['training_provenance']);value=g.read(path);value['data_revoked']=True;put(path,value)
    with pytest.raises(ValueError):e.freeze()
    assert not (e.DEST/'candidate-freeze.json').exists()


def test_original_attempt_and_changed_parent_prevent_new_policy(case):
    _,put=case;put(e.ORIGINAL_DEST/'lane-fresh-attempt.json',{'status':'started'})
    with pytest.raises(ValueError,match='original_cycle2_TEST_started'):e.prepare_policy()


def test_changed_original_parent_policy_cannot_rebind_existing_retry_policy(case):
    _,put=case;e.prepare_policy();parent=g.read(e.PARENTS['fresh']);parent['role']='changed-after-policy'
    put(e.PARENTS['fresh'],parent)
    with pytest.raises(ValueError,match='policy_or_parent_changed'):e.freeze()


def test_metadata_only_TRAIN_VAL_overlap_is_rejected_before_comparison(case):
    _,put=case;parent=g.read(e.PARENTS['fresh']);path=g.workspace_path(parent['benchmark_manifest']);rows=g.read(path)
    model=g.read(e.CANDIDATE);dev=g.read(model['source_release_path'])
    rows[0]['sha256']=dev[0]['sha256'];rows[0]['leakage_group_id']=dev[0]['leakage_group_id']
    ref=put(path,rows);parent['benchmark_sha256']=ref['sha256'];put(e.PARENTS['fresh'],parent)
    e.prepare_policy()
    with pytest.raises(ValueError,match='TEST_training_or_VAL_metadata_leakage'):e.freeze()
    assert not (e.DEST/'fresh-attempt.json').exists()


def test_postfreeze_proof_bytes_cannot_be_silently_replaced(case):
    folder,put=case;e.prepare_policy();e.freeze()
    path=folder/'verification-final.json';value=g.read(path);value['extra_diagnostic']='changed-bytes';put(path,value)
    with pytest.raises(ValueError,match='artifact_or_proof_changed'):e.prepare_plans()


def test_explicit_approval_and_permanent_attempt_survive_failed_worker(case,monkeypatch):
    e.prepare_policy();e.freeze();e.prepare_plans()
    with pytest.raises(ValueError,match='explicit_root_TEST_approval'):e.run('fresh')
    assert not (e.DEST/'fresh-attempt.json').exists()
    def fail(*args):raise RuntimeError('synthetic worker failure, no inference')
    monkeypatch.setattr(e,'compare',fail)
    with pytest.raises(RuntimeError):e.run('fresh',approved=True)
    assert g.read(e.DEST/'fresh-attempt.json')['status']=='started'
    with pytest.raises(ValueError,match='already_started'):e.run('fresh',approved=True)


def test_exclusive_attempt_prevents_parallel_duplicate_comparison(case,monkeypatch):
    e.prepare_policy();e.freeze();e.prepare_plans();value=e.validated_freeze();calls=[]
    monkeypatch.setattr(e,'validated_freeze',lambda:value)
    def fake_compare(plan,output):
        calls.append(1);g.write(output,{'samples':119});return {'samples':119}
    monkeypatch.setattr(e,'compare',fake_compare)
    def call():
        try:return e.run('fresh',approved=True)
        except (ValueError,OSError):return 'refused'
    with ThreadPoolExecutor(max_workers=2) as executor:results=list(executor.map(lambda _:call(),range(2)))
    assert len(calls)==1 and len([r for r in results if r=='refused'])==1
    assert g.read(e.DEST/'fresh-completion.json')['samples']==119
