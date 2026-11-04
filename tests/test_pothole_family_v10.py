"""Synthetic same-CPU family metadata; no photographs, ONNX execution or TEST."""
import hashlib,json
from pathlib import Path
import pytest
from test_pothole_family import case
from scripts import evaluate_pothole_family as v9
from scripts import evaluate_pothole_family_v10 as f


def score(old,new):return {'status':'available','domains':{'original':{'images':402,'ap50':old},'mwpd':{'images':250,'ap50':new}}}


def table():return {'baseline':score(.7,.5),'nano':score(.69,.51),'small':score(.68,.52),'nano960':score(.69,.55),'nano1280':score(.69,.60)}


def test_same_CPU_only_old_gate_and_tie_order_never_use_native_scores():
    values=table();assert f.choose_cpu(values)['selected_member']=='nano1280'
    values['small']['native_validation_mean']=1.
    assert f.choose_cpu(values)['selected_member']=='nano1280'
    values={name:score(.5,.5) for name in f.ORDER};assert f.choose_cpu(values)['selected_member']=='baseline'
    values['nano']=score(.49,.61);values['small']=score(.5,.6);assert f.choose_cpu(values)['selected_member']=='nano'
    values['nano1280']=score(.46,1.)
    assert f.choose_cpu(values)['members']['nano1280']['eligible'] is False


@pytest.mark.parametrize('bad',[True,float('nan'),float('inf'),'0.8'])
def test_invalid_CPU_AP_is_not_replaced_with_native_AP(bad):
    values=table();values['nano']['domains']['original']['ap50']=bad;values['nano']['native_validation_mean']=.9
    with pytest.raises(ValueError,match='finite_same_CPU'):f.choose_cpu(values)


@pytest.fixture
def v10_case(case,monkeypatch):
    # Upgrade only this synthetic project to actual schema sizes. No image exists.
    put=case.put;raw=lambda name,data:(case.path(name).parent.mkdir(parents=True,exist_ok=True),case.path(name).write_bytes(data))[-1]
    old=v9.read(v9.NANO+'/release.json');vals=[row|{'development_domain':'original'} for row in old if row['split']=='validation']
    cap=vals[0]['expires_at']
    for index in range(3,652):
        vals.append({'sample_id':f'v{index:04d}','sha256':hashlib.sha256(str(index).encode()).hexdigest(),'split':'validation','expires_at':cap,
            'development_domain':'original' if index<402 else 'mwpd','objects':[{'class':'pothole'}]})
    rows=[row for row in old if row['split']=='train']+vals
    for folder in (v9.BASE,v9.NANO,v9.SMALL):
        release=put(folder+'/release.json',rows);plan=v9.read(folder+'/plan.json');plan['development_sha256']=release;plan_sha=put(folder+'/plan.json',plan)
        model=v9.read(folder+'/model.json');model.update(dataset_release='manifest-sha256:'+release,refinement_plan_sha256=plan_sha,input_size=[640,640]);manifest=put(folder+'/model.json',model)
        snap=v9.read(folder+'/selection-freeze.json');snap['manifest_sha256']=manifest;put(folder+'/selection-freeze.json',snap)
        if folder==v9.NANO:
            put(folder+'/model-development.json',model|{'development_only_release':True,'executed_export_manifest_sha256':manifest,
                'selection_freeze_sha256':v9.sha256(folder+'/selection-freeze.json'),'packaging_source_sha256':v9.sha256('scripts/package_development_export.py')})
        for filename in v9.PROOFS:
            proof=v9.read(folder+'/'+filename);proof['manifest_sha256']=v9.sha256(folder+('/model-development.json' if folder==v9.NANO else '/model.json'));put(folder+'/'+filename,proof)
    retry=v9.read(v9.RETRY_PUBLIC)
    for name in retry['copied_input_sha256']:
        raw(v9.ORIGINAL_SMALL+'/'+name,case.path(v9.SMALL+'/'+name).read_bytes());retry['copied_input_sha256'][name]=v9.sha256(v9.SMALL+'/'+name)
    retry_sha=put(v9.RETRY_PUBLIC,retry);put(v9.SMALL+'/retry-plan.json',retry);monkeypatch.setattr(v9,'RETRY_PUBLIC_SHA',retry_sha)
    supervisor=v9.read(v9.SMALL+'/shared-training-supervisor.json');supervisor['plan_sha256']=v9.sha256(v9.SMALL+'/plan.json');put(v9.SMALL+'/shared-training-supervisor.json',supervisor)
    v9.prepare_policy();monkeypatch.setattr(f,'V9_POLICY_SHA',v9.sha256(v9.POLICY))
    for name in f.EVAL_SOURCES:
        if not case.path(name).exists():raw(name,b'SYNTHETIC-SOURCE-V10')
    monkeypatch.setattr(f,'__file__',str(case.path('scripts/evaluate_pothole_family_v10.py')))
    trial={'format':'pothole-resolution-v10-predeclared-plan-v1','source_sha256':v9.sha256('scripts/evaluate_pothole_resolution.py'),
        'dependencies':{'agent/models.py':v9.sha256('agent/models.py')},
        'parent_manifest':v9.pointer(v9.NANO+'/model-development.json'),'checkpoint':v9.pointer(v9.NANO+'/best.pt'),
        'validation_release':v9.pointer(v9.NANO+'/release.json'),'parent_family_policy':v9.pointer(v9.POLICY),'baseline_manifest':v9.pointer(v9.BASE+'/model.json'),
        'small_learning_plan':v9.pointer(v9.SMALL+'/plan.json'),'small_retry_protocol':v9.pointer(v9.SMALL+'/retry-plan.json'),
        'input_sizes':[640,960,1280],'new_exports':[960,1280],'CPU_family_members':list(f.ORDER),'new_training_updates':0,'parameter_or_checkpoint_changes':False,
        'TEST_images':0,'TEST_used_for_choice':False,'user_recordings_used':False,'GPU_used':False,'source_expiry_extended':False,
        'confidence_floor':.01,'operating_confidence':.1,'nms_iou':.5,'VAL_membership_sha256':f.membership(vals)}
    trial_sha=put(f.RES+'/trial-plan.json',trial);put('docs/reports/POTHOLE-RESOLUTION-V10-PLAN.json',trial);monkeypatch.setattr(f,'TRIAL_SHA',trial_sha)
    parent=v9.read(trial['parent_manifest']['path']);variants={}
    for size in (960,1280):
        name='nano'+str(size);folder=f.RES+'/'+name;raw(folder+'/model.onnx',('SYNTHETIC-VARIANT:'+name).encode());raw(folder+'/best.pt',case.path(v9.NANO+'/best.pt').read_bytes())
        model={key:value for key,value in parent.items() if key not in ('executed_export_manifest_sha256','selection_freeze_sha256','packaging_source_sha256')}
        model.update(input_size=[size,size],weights=folder+'/model.onnx',sha256=v9.sha256(folder+'/model.onnx'),inference_parent_manifest_sha256=trial['parent_manifest']['sha256'],inference_option_plan_sha256=trial_sha,inference_option_new_training_updates=0)
        put(folder+'/model.json',model)
        variants[name]={'manifest':v9.pointer(folder+'/model.json'),'checkpoint':v9.pointer(folder+'/best.pt'),'weights_sha256':model['sha256'],'input_size':[size,size],'output_shape':[1,sum((size//stride)**2 for stride in (8,16,32)),6]}
        for filename in v9.PROOFS:
            proof=v9.read(v9.NANO+'/'+filename);proof.update(manifest_sha256=v9.sha256(folder+'/model.json'),weight_sha256=model['sha256']);put(folder+'/'+filename,proof)
    put(f.RES+'/exports.json',{'models':variants,'plan_sha256':trial_sha,'checkpoint_unchanged':True,'GPU_used':False,'TEST_images':0})
    def write_cpu():
        models=table()
        identities={name:{'manifest':v9.pointer(folder+'/'+manifest),'checkpoint':v9.pointer(folder+'/best.pt'),'weights_sha256':v9.read(folder+'/'+manifest)['sha256'],'input_size':[640,640]}
            for name,folder,manifest in [('baseline',v9.BASE,'model.json'),('nano',v9.NANO,'model-development.json'),('small',v9.SMALL,'model.json')]}
        identities.update({name:{key:value for key,value in entry.items() if key!='output_shape'} for name,entry in variants.items()})
        for name,entry in identities.items():models[name].update(entry)
        put(f.RES+'/cpu-validation-attempt.json',{'plan_sha256':trial_sha,'status':'started','TEST_images':0})
        put(f.RES+'/cpu-validation.json',{'format':'pothole-resolution-v10-CPU-VAL652','complete':True,'plan_sha256':trial_sha,
            'validation_release_sha256':trial['validation_release']['sha256'],'VAL_membership_sha256':f.membership(vals),'runtime':f.RUNTIME,'TEST_images':0,'confidence_floor':.01,'nms_iou':.5,'models':models})
    return type('V10Case',(),{'base':case,'write_cpu':staticmethod(write_cpu)})


def test_full_metadata_freeze_uses_only_same_CPU_values_and_strict_plan(case,v10_case,monkeypatch):
    import cv2
    monkeypatch.setattr(cv2,'imread',lambda *a:pytest.fail('image read in metadata selection'))
    f.prepare_policy();v10_case.write_cpu();result=f.freeze_family()
    assert result['selected_member']=='nano1280'  # Native fixture would select Small.
    family=f.validated_family();assert family['native_or_GPU_AP_used_for_family_ranking'] is False
    assert all('validation' not in entry for entry in family['members'].values())
    assert f.prepare_comparison_plans()['TEST_inferences']==0
    assert not case.path(v9.DEST+'/fresh-attempt.json').exists()


@pytest.mark.parametrize('bad',['runtime','membership','counts','checkpoint','TEST','incomplete','retry_running'])
def test_bad_or_incomplete_actual_CPU_report_never_falls_back_to_native(v10_case,bad):
    case=v10_case.base;f.prepare_policy();v10_case.write_cpu();path=f.RES+'/cpu-validation.json';value=v9.read(path)
    if bad=='runtime':value['runtime']['GPU_used']=True
    elif bad=='membership':value['VAL_membership_sha256']='wrong'
    elif bad=='counts':value['models']['nano960']['domains']['original']['images']=401
    elif bad=='checkpoint':value['models']['nano1280']['checkpoint']['sha256']='wrong'
    elif bad=='TEST':value['TEST_images']=1
    elif bad=='incomplete':value['complete']=False
    else:
        other=v9.read(v9.SMALL+'/shared-training-supervisor.json');other['status']='running';case.put(v9.SMALL+'/shared-training-supervisor.json',other)
    case.put(path,value)
    with pytest.raises(ValueError):f.freeze_family()
    assert not case.path(f.DEST+'/family-freeze.json').exists()


def test_v9_attempt_or_changed_geometry_blocks_v10(v10_case):
    case=v10_case.base;case.put(v9.DEST+'/fresh-attempt.json',{'status':'started'})
    with pytest.raises(ValueError,match='v9_TEST_started'):f.prepare_policy()
    case.path(v9.DEST+'/fresh-attempt.json').unlink();f.prepare_policy();v10_case.write_cpu()
    path=f.RES+'/nano960/model.json';value=v9.read(path);value['confidence']=.2;case.put(path,value)
    with pytest.raises(ValueError):f.freeze_family()


def test_winner_proof_failure_and_permanent_v10_attempt(v10_case,monkeypatch):
    case=v10_case.base;f.prepare_policy();v10_case.write_cpu();f.freeze_family()
    path=f.RES+'/nano1280/verification-final.json';proof=v9.read(path);proof['app_vs_torch']['passed']=False;case.put(path,proof)
    with pytest.raises(ValueError,match='default_tolerance'):f.prepare_comparison_plans()
    proof['app_vs_torch']['passed']=True;case.put(path,proof);f.prepare_comparison_plans();calls=[]
    def fail(*args):calls.append(args);raise RuntimeError('synthetic inference failure')
    monkeypatch.setattr(v9,'compare',fail)
    with pytest.raises(RuntimeError):f.run('fresh')
    with pytest.raises(ValueError,match='already_started'):f.run('fresh')
    assert len(calls)==1 and case.path(f.DEST+'/fresh-attempt.json').exists()
