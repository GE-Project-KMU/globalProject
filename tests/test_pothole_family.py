"""Synthetic metadata/proof contracts; never reads real images or runs models."""
import hashlib,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest
from scripts import evaluate_pothole_family as f


def metrics(old=.7,new=.5,epoch=0):return {'epoch':epoch,'old_validation_ap50':old,'new_validation_ap50':new}


def members():
    return {'baseline':{'status':'available','validation':metrics(.5,.5)},'tiny':{'status':'rejected_no_artifact'},
        'nano':{'status':'available','validation':metrics(.5,.5,1)},'small':{'status':'available','validation':metrics(.5,.5,1)}}


def test_finite_validation_old_domain_gate_and_exact_tie_order():
    values=members();assert f.choose_family(values)['selected_member']=='baseline'
    values['nano']['validation']=metrics(.49,.61,1);values['small']['validation']=metrics(.5,.6,1)
    assert f.choose_family(values)['selected_member']=='nano'
    values['small']['validation']=metrics(.46,1.,1)
    result=f.choose_family(values);assert result['selected_member']=='nano' and result['members']['small']['eligible'] is False
    values['small']['status']='running'
    with pytest.raises(ValueError,match='must_complete'):f.choose_family(values)


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),-.1,1.1,'0.8'])
def test_invalid_validation_metric_is_never_selection_evidence(value):
    values=members();values['nano']['validation']['old_validation_ap50']=value
    with pytest.raises(ValueError,match='finite_unit'):f.choose_family(values)


@pytest.fixture
def case(tmp_path,monkeypatch):
    real_root=f.ROOT;assert tmp_path.is_relative_to(real_root)
    monkeypatch.setattr(f,'ROOT',tmp_path)
    def path(value,exists=False):
        result=Path(value);result=result if result.is_absolute() else tmp_path/result;result=result.resolve()
        if not result.is_relative_to(tmp_path):raise ValueError('synthetic_scope_only')
        if exists and not result.exists():raise FileNotFoundError('synthetic_missing')
        return result
    monkeypatch.setattr(f,'workspace_path',path)
    monkeypatch.setattr(f,'sha256',lambda value:hashlib.sha256(path(value,True).read_bytes()).hexdigest())
    def put(value,body):
        p=path(value);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(body));return f.sha256(value)
    def raw(value,body):
        p=path(value);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body);return f.sha256(value)
    for name in {*f.EVAL_SOURCES,'scripts/package_development_export.py','scripts/train_pothole_small.py'}:raw(name,b'SYNTHETIC-SOURCE')
    monkeypatch.setattr(f,'__file__',str(path('scripts/evaluate_pothole_family.py')))
    cap=(datetime.now(timezone.utc)+timedelta(days=2)).isoformat()
    rows=[{'sample_id':'train','sha256':'a'*64,'split':'train','expires_at':cap,'objects':[{'class':'pothole'}]}]+[
        {'sample_id':'val'+str(i),'sha256':str(i+1)*64,'split':'validation','expires_at':cap,'objects':[{'class':'pothole'}]} for i in range(3)]
    def artifact(folder,selected,packaged=False):
        release=put(folder+'/release.json',rows)
        plan={'minimum_epochs':1,'maximum_epochs':2,'development_sha256':release}
        plan_sha=put(folder+'/plan.json',plan)
        weight=raw(folder+'/model.onnx',('SYNTHETIC-ONNX:'+folder).encode());checkpoint=raw(folder+'/best.pt',('SYNTHETIC-PT:'+folder).encode())
        model={'format':'onnx','task':'pothole','classes':['pothole'],'research_only':True,'production_approved':False,
            'weights':folder+'/model.onnx','sha256':weight,'source_release_path':folder+'/release.json','dataset_release':'manifest-sha256:'+release,
            'confidence':.1,'nms_iou':.5,'best_epoch':selected['epoch'],'trained_checkpoint_sha256':checkpoint,'refinement_plan_sha256':plan_sha,
            'development_only_release':True,'exported_before_final_test':True,'test_split_used':False}
        manifest=put(folder+'/model.json',model)
        if folder==f.BASE:
            put(folder+'/selection-freeze.json',{'test_inference_performed':False,'selected_validation':selected,'manifest_sha256':manifest,'checkpoint_sha256':checkpoint,'onnx_sha256':weight})
        else:
            history=[metrics(),metrics(.68,.6,1),selected]
            put(folder+'/history.json',{'history':history,'selected':selected})
            snap={'TEST_inference_performed':False,'epochs_completed':2,'selected':selected,'manifest_sha256':manifest,'checkpoint_sha256':checkpoint,'weights_sha256':weight,
                ('candidate_improved_on_VAL' if packaged else 'candidate_selected'):True}
            put(folder+'/selection-freeze.json',snap)
            if packaged:
                put(folder+'/model-development.json',model|{'development_only_release':True,'executed_export_manifest_sha256':manifest,
                    'selection_freeze_sha256':f.sha256(folder+'/selection-freeze.json'),'packaging_source_sha256':f.sha256('scripts/package_development_export.py')})
    artifact(f.BASE,metrics(.7,.5,7));artifact(f.NANO,metrics(.69,.61,2),True);artifact(f.SMALL,metrics(.68,.66,2))
    put(f.TINY+'/selection-freeze.json',{'candidate_selected':False,'test_used':False})
    put(f.TINY+'/history.json',{'history':[metrics(),metrics(.6,.5,1)],'selected':metrics()})
    copied={}
    for name in ('preflight.json','GPU-memory-preflight.json'):put(f.SMALL+'/'+name,{'synthetic':True})
    for name in ('plan.json','release.json','preflight.json','GPU-memory-preflight.json'):
        copied[name]=f.sha256(f.SMALL+'/'+name);raw(f.ORIGINAL_SMALL+'/'+name,path(f.SMALL+'/'+name).read_bytes())
    parent_paths=[f.ORIGINAL_SMALL+'/shared-training-supervisor.json','data/training/public-nextcycle/P2-shared-training-supervisor.json','data/training/public-nextcycle/lane-shared-training-supervisor.json']
    stopped={name:put(name,{'format':'small-v9-shared-supervision-v1','status':'failed','sources_unchanged':True,'own_child_running_after_cleanup':False,
        'own_child_stopped':True,'device_memory':{'over_budget_observed':True}}) for name in parent_paths}
    retry={'format':'pothole-small-v9-sequential-full-restart-v1','copied_input_sha256':copied,'stopped_parent_receipts':stopped,
        'restart_from_official_initialization':True,'optimizer_state_resumed':False,'prior_partial_weights_used':False,'learning_recipe_changed':False,'TEST_used':False,
        'learning_source_sha256':f.sha256('scripts/train_pothole_small.py'),'supervisor_sha256':'c'*64}
    retry_sha=put(f.RETRY_PUBLIC,retry);monkeypatch.setattr(f,'RETRY_PUBLIC_SHA',retry_sha);put(f.SMALL+'/retry-plan.json',retry)
    put(f.SMALL+'/shared-training-supervisor.json',{'format':'small-v9-sequential-restart-supervision-v1','status':'completed','child_exit_code':0,
        'sources_unchanged':True,'own_child_running_after_cleanup':False,'training_math_changed':False,'device_memory':{'over_budget_observed':False},
        'plan_sha256':copied['plan.json'],'supervisor_sha256':'c'*64})
    protocol_sha=put(f.US100_PROTOCOL,{'synthetic':'protocol'})
    put(f.US100_ORIGINAL_PLAN,{'protocol_sha256':'prior','plan':{'model_inferences_before_freeze':0,'train_or_selection_use':False}})
    put(f.PARENTS['fresh'],{'protocol_sha256':protocol_sha,'plan':{'model_inferences_before_freeze':0,'train_or_selection_use':False,'previous_pre_build_protocol_sha256':'prior'}})
    for name,total,positive in ((f.US100,100,50),(f.REGRESSION,145,97)):
        data=[{'sample_id':str(i),'sha256':hashlib.sha256((name+str(i)).encode()).hexdigest(),'split':'test','objects':[{'class':'pothole'}] if i<positive else [],'expires_at':cap} for i in range(total)]
        pin=put(name,data);monkeypatch.setattr(f,'US100_SHA' if total==100 else 'REGRESSION_SHA',pin)
    put(f.US100_ACQUISITION,{'status':'frozen','model_inferences':0,'release_sha256':f.US100_SHA,'protocol_sha256':protocol_sha})
    put(f.PARENTS['regression'],{'benchmark_sha256':f.REGRESSION_SHA})
    put(f.BASE+'/canonical-plan.json',{'cases':[{'sha256':row['sha256'],'image':'NEVER_READ'} for row in rows[1:]]})
    def proofs(folder,manifest_name):
        artifact=path(folder+'/'+manifest_name)
        for row,name in zip(rows[1:],f.PROOFS):
            put(folder+'/'+name,{'manifest_sha256':f.sha256(artifact),'weight_sha256':f.read(artifact)['sha256'],'image_sha256':row['sha256'],'checkpoint_sha256':f.sha256(folder+'/best.pt'),
                'numerical_parity_passed':True,'postprocess_parity_passed':True,'artifact_contract_verified':True,'torch_checkpoint_compared':True,'synthetic_input':False,
                'adapter_result_count':1,'app_backend':'onnxruntime-cpu','app_backend_version':'1.31.1','metadata_valid':True,'weight_hash_verified':True,'preprocess_contract_verified':True,'finite_output':True,
                **{key:{'passed':True,'rtol':.001,'atol':.0001} for key in ('app_repeatability','app_vs_ort','app_vs_torch','ort_vs_torch')},
                'detection_postprocess_parity':{'ort':{'passed':True},'torch':{'passed':True}}})
    for folder,name in ((f.BASE,'model.json'),(f.NANO,'model-development.json'),(f.SMALL,'model.json')):proofs(folder,name)
    return type('Case',(),{'root':tmp_path,'path':staticmethod(path),'put':staticmethod(put)})


def test_family_freeze_and_strict_plan_preparation_read_metadata_only(case,monkeypatch):
    import cv2
    monkeypatch.setattr(cv2,'imread',lambda *a:pytest.fail('TEST image read during metadata preparation'))
    f.prepare_policy();result=f.freeze_family();assert result['selected_member']=='small' and result['TEST_inferences']==0
    assert f.prepare_comparison_plans()['TEST_inferences']==0
    for role in ('fresh','regression'):
        value=f.read(f.DEST+'/'+role+'-plan.json');assert value['family_digest']==result['family_digest'] and len(value['strict_proofs']['candidate'])==3
    with pytest.raises(ValueError,match='already_frozen'):f.freeze_family()


@pytest.mark.parametrize('bad',['retry_running','missing_retry_selection','new_TEST_used','copied_input','tiny_artifact'])
def test_incomplete_or_unbound_family_cannot_freeze(case,bad):
    f.prepare_policy()
    if bad=='retry_running':
        path=f.SMALL+'/shared-training-supervisor.json';value=f.read(path);value['status']='running';case.put(path,value)
    elif bad=='missing_retry_selection':case.path(f.SMALL+'/selection-freeze.json').unlink()
    elif bad=='new_TEST_used':
        path=f.SMALL+'/selection-freeze.json';value=f.read(path);value['TEST_inference_performed']=True;case.put(path,value)
    elif bad=='copied_input':case.put(f.ORIGINAL_SMALL+'/preflight.json',{'changed':True})
    else:case.put(f.TINY+'/model.json',{'forbidden':'artifact'})
    with pytest.raises((ValueError,FileNotFoundError)):f.freeze_family()
    assert not case.path(f.DEST+'/family-freeze.json').exists()


def test_strict_failure_cannot_prepare_TEST_plan(case):
    f.prepare_policy();f.freeze_family()
    path=f.SMALL+'/verification-final.json';value=f.read(path);value['app_vs_torch']['passed']=False;case.put(path,value)
    with pytest.raises(ValueError,match='default_tolerance'):f.prepare_comparison_plans()
    assert not case.path(f.DEST+'/fresh-plan.json').exists()


def test_durable_attempt_precedes_comparator_and_failed_attempt_never_retries(case,monkeypatch):
    f.prepare_policy();f.freeze_family();f.prepare_comparison_plans();calls=[]
    def failure(plan,output):
        assert case.path(f.DEST+'/fresh-attempt.json').exists();calls.append(plan);raise RuntimeError('synthetic_failure')
    monkeypatch.setattr(f,'compare',failure)
    with pytest.raises(RuntimeError):f.run('fresh')
    assert len(calls)==1 and f.read(f.DEST+'/fresh-attempt.json')['status']=='started'
    with pytest.raises(ValueError,match='already_started'):f.run('fresh')
    assert len(calls)==1


def test_family_digest_and_parent_plan_changes_refused_before_comparator(case,monkeypatch):
    f.prepare_policy();f.freeze_family();f.prepare_comparison_plans()
    path=f.DEST+'/family-freeze.json';value=f.read(path);value['selection']['selected_member']='nano';case.put(path,value)
    monkeypatch.setattr(f,'compare',lambda *a:pytest.fail('comparison after family mutation'))
    with pytest.raises(ValueError,match='family_digest'):f.run('regression')


def test_concurrent_attempts_can_start_comparator_only_once(case,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    f.prepare_policy();f.freeze_family();f.prepare_comparison_plans()
    barrier=threading.Barrier(2);original=f.write_new;calls=[]
    def write(path,value):
        if str(path).endswith('fresh-attempt.json'):barrier.wait(timeout=5)
        return original(path,value)
    def compare(plan,output):
        calls.append(plan);case.put(output,{'samples':100});return {'samples':100}
    monkeypatch.setattr(f,'write_new',write);monkeypatch.setattr(f,'compare',compare)
    def run():
        try:f.run('fresh');return 'completed'
        except (ValueError,FileExistsError):return 'blocked'
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:run(),range(2)))
    assert sorted(results)==['blocked','completed'] and len(calls)==1


def test_preexisting_US100_model_inferences_block_policy_before_any_freeze(case):
    path=f.US100_ACQUISITION;value=f.read(path);value['model_inferences']=1;case.put(path,value)
    with pytest.raises(ValueError,match='zero_inference'):f.prepare_policy()
    assert not case.path(f.POLICY).exists()


def test_parent_benchmark_plan_tampering_is_not_used_as_new_protocol(case):
    f.prepare_policy();f.freeze_family();f.prepare_comparison_plans()
    path=f.PARENTS['regression'];value=f.read(path);value['changed']=True;case.put(path,value)
    with pytest.raises(ValueError,match='parent_plan_changed'):f.run('regression')
