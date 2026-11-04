"""Synthetic metadata admission tests; no real GPU or model accuracy claims."""
from datetime import datetime,timedelta,timezone
import json,uuid
from pathlib import Path
import pytest
from scripts import gpu_model_set as modelset


@pytest.fixture
def fixture_set(monkeypatch):
    folder=modelset.ROOT/'.cache/gpu-model-set-tests'/uuid.uuid4().hex;folder.mkdir(parents=True)
    monkeypatch.setattr(modelset,'PUBLIC_ROOTS',(folder,))
    def save(name,value):
        path=folder/name;path.write_text(json.dumps(value));return modelset.pointer(path)
    specification={'models':{}};release_rows={}
    expiry=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
    for task,count in modelset.COUNTS.items():
        rows=[]
        for index in range(count+2):
            image=folder/f'{task}-{index}.bin';image.write_bytes(f'SYNTHETIC:{task}:{index}'.encode())
            label='license_plate' if task=='redaction' and index>=20 else 'face' if task=='redaction' else task
            rows.append({'sample_id':str(index),'source_id':str(index).zfill(4),'path':image.relative_to(modelset.ROOT).as_posix(),
                'sha256':modelset.sha256(image),'expires_at':expiry,'positive_pixels':1,
                'split':'validation' if index<count else 'train' if index==count else 'test',
                'objects':[{'class':label,'bbox_xywh_px':[1,1,2,2]}],**({'evaluation_class':label} if task=='redaction' else {})})
        weights=folder/(task+'.onnx');weights.write_bytes(('SYNTHETIC-WEIGHT:'+task).encode())
        release_ref=save(task+'-release.json',rows)
        manifest={'task':task,'format':'onnx','input_size':[16,16],'sha256':modelset.sha256(weights),
            'weights':weights.relative_to(modelset.ROOT).as_posix(),'research_only':True,'production_approved':False,
            'source_release_path':release_ref['path'],'dataset_release':'manifest-sha256:'+release_ref['sha256']}
        manifest_ref=save(task+'-model.json',manifest)
        proof={'task':task,'manifest_sha256':manifest_ref['sha256'],'weight_sha256':manifest['sha256'],
            'synthetic_input':False,'image_sha256':rows[0]['sha256'],'torch_checkpoint_compared':True,
            'artifact_contract_verified':True,'numerical_parity_passed':True,'postprocess_parity_passed':True,
            'app_backend':'onnxruntime-cpu','app_backend_version':'1.31.1','metadata_valid':True,'weight_hash_verified':True,
            'finite_output':True,'preprocess_contract_verified':True,'output_shape':[1,2,16,16] if task=='lane' else [1,3,7 if task=='redaction' else 6],
            'adapter_result_count':1,
            **{key:{'passed':True,'rtol':.001,'atol':.0001} for key in ('app_repeatability','app_vs_ort','app_vs_torch','ort_vs_torch')},
            'lane_adapter_postprocess_matches_reference':True,'mask_agreement':{mode:{'pixel_agreement':1.} for mode in ('app','ort','torch')},
            'detection_postprocess_parity':{mode:{'passed':True} for mode in ('ort','torch')}}
        specification['models'][task]={'manifest':manifest_ref,'cpu_verifications':[save(task+'-canonical.json',proof)],
            'validation_manifest':release_ref}
        release_rows[task]=rows
    source=folder/'specification.json';source.write_text(json.dumps(specification))
    return folder,source,specification,release_rows


def test_freeze_has_exact_validation_counts_and_excludes_train_test(fixture_set):
    folder,source,_,rows=fixture_set;output=folder/'plan.json'
    result=modelset.freeze(source,output);plan=modelset.validate_plan(modelset.read(output))
    assert result['gpu_sessions_started']==0
    assert {task:len(entry['cases']) for task,entry in plan['models'].items()}==modelset.COUNTS
    for task,entry in plan['models'].items():
        assert not ({row['sha256'] for row in rows[task] if row['split']!='validation'}&{row['sha256'] for row in entry['cases']})
    assert plan['new_gpu_support_approved'] is False


def test_failed_or_wrong_artifact_cpu_canonical_blocks_freeze(fixture_set):
    folder,source,spec,_=fixture_set;proof_path=modelset.workspace_path(spec['models']['lane']['cpu_verifications'][0]['path'])
    proof=modelset.read(proof_path);proof['app_vs_torch']['passed']=False
    proof_path.write_text(json.dumps(proof));spec['models']['lane']['cpu_verifications'][0]=modelset.pointer(proof_path)
    source.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='full_guarded_cpu131'):modelset.freeze(source,folder/'must-not-freeze.json')
    assert not (folder/'must-not-freeze.json').exists()


def test_changed_weights_or_cases_cannot_reuse_frozen_plan(fixture_set):
    folder,source,spec,_=fixture_set;output=folder/'plan.json';modelset.freeze(source,output)
    plan=modelset.read(output);plan['models']['pothole']['cases'].reverse()
    with pytest.raises(ValueError,match='case_set_changed'):modelset.validate_plan(plan)
    plan=modelset.read(output);manifest=modelset.pinned(spec['models']['pothole']['manifest'])
    modelset.workspace_path(manifest['weights']).write_bytes(b'CHANGED-SYNTHETIC-WEIGHT')
    with pytest.raises(ValueError,match='weight_changed'):modelset.validate_plan(plan)


def test_failed_gpu_trial_cannot_generate_child_or_registration(fixture_set):
    folder,source,_,_=fixture_set;plan_path=folder/'plan.json';modelset.freeze(source,plan_path)
    failed=folder/'failed-gpu.json';failed.write_text(json.dumps({'stage':'diagnostic','passed':False,'platform':modelset.GPU_PROFILE}))
    with pytest.raises(ValueError,match='strict_model_set_gpu_proof_required'):
        modelset.stage(plan_path,failed,folder/'must-not-stage')
    assert not (folder/'must-not-stage').exists()


def test_user_recording_namespace_is_not_public_validation():
    with pytest.raises(ValueError,match='public_research_case_namespace_required'):modelset.public_case_path(modelset.ROOT/'데이터/private.svo2')


def trial_proof(plan_path,stage='diagnostic',core=None):
    plan=modelset.read(plan_path)
    value={'format':'frozen-model-set-cuda-validation-v1','stage':stage,'platform':modelset.GPU_PROFILE,
        'raw_tolerance':{'rtol':.001,'atol':.0001},
        'test_or_user_images_read':0,'threshold_or_model_selection_performed':False,
        'per_model_gpu_node_events':{task:1 for task in modelset.TASKS},
        'plan_sha256':modelset.sha256(plan_path),'passed':True,'gpu_node_events_observed':True,
        'source_sha256':{name:modelset.sha256(modelset.ROOT/name) for name in modelset.RUNTIME_SOURCES},
        'validated_models':{task:{'manifest_sha256':entry['manifest']['sha256'],'weights_sha256':entry['weights_sha256'],'case_set_sha256':entry['case_set_sha256']} for task,entry in plan['models'].items()},
        'results':{task:{'cases':count,'raw_failures':0,'post_failures':0} for task,count in modelset.COUNTS.items()}}
    if core:value['descriptor_core_sha256']=core
    return value


def test_staging_is_not_registered_and_native_exact_proof_is_required(fixture_set,monkeypatch):
    folder,source,_,_=fixture_set;plan_path=folder/'plan.json';modelset.freeze(source,plan_path)
    diagnostic=folder/'diagnostic.json';diagnostic.write_text(json.dumps(trial_proof(plan_path)))
    modelset.stage(plan_path,diagnostic,folder/'stage');staging=folder/'stage/staging.json';descriptor=modelset.read(staging)
    modelset.validate_descriptor(descriptor)
    with pytest.raises(ValueError,match='registered_gpu_model_set_required'):modelset.validate_descriptor(descriptor,True)
    native=folder/'native.json';native.write_text(json.dumps(trial_proof(plan_path,'native','wrong-core')))
    monkeypatch.setattr(modelset,'REGISTRY_ROOT',folder/'registry')
    with pytest.raises(ValueError,match='native_descriptor_proof_required'):modelset.register(staging,native,folder/'registry/profile.json')
    native.write_text(json.dumps(trial_proof(plan_path,'native',descriptor['core_sha256'])))
    modelset.register(staging,native,folder/'registry/profile.json')
    assert modelset.validate_descriptor(modelset.read(folder/'registry/profile.json'),True)


def test_different_case_proof_cannot_stage_same_model(fixture_set):
    folder,source,_,_=fixture_set;plan_path=folder/'plan.json';modelset.freeze(source,plan_path)
    proof=trial_proof(plan_path);proof['validated_models']['lane']['case_set_sha256']='0'*64
    diagnostic=folder/'mixed-cases.json';diagnostic.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='model_or_case_proof_mismatch'):modelset.stage(plan_path,diagnostic,folder/'invalid-stage')


def test_relaxed_canonical_tolerance_cannot_freeze_gpu_trial(fixture_set):
    folder,source,spec,_=fixture_set;entry=spec['models']['pothole']['cpu_verifications'][0]
    path=modelset.workspace_path(entry['path']);proof=modelset.read(path);proof['app_vs_torch']['atol']=1.
    path.write_text(json.dumps(proof));spec['models']['pothole']['cpu_verifications'][0]=modelset.pointer(path)
    source.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='default_tolerance_required'):modelset.freeze(source,folder/'relaxed-plan.json')


def test_per_model_cpu_fallback_only_proof_cannot_stage(fixture_set):
    folder,source,_,_=fixture_set;plan=folder/'plan.json';modelset.freeze(source,plan)
    proof=trial_proof(plan);proof['per_model_gpu_node_events']['lane']=0
    path=folder/'fallback-only.json';path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='actual_gpu_nodes_required_per_model'):modelset.stage(plan,path,folder/'no-stage')


def test_explicit_validation_cases_preserve_order_and_canonical(fixture_set):
    folder,source,spec,rows=fixture_set
    for task in modelset.TASKS:
        spec['models'][task]['case_sha256s']=[row['sha256'] for row in reversed(rows[task]) if row['split']=='validation']
    source.write_text(json.dumps(spec));output=folder/'explicit-plan.json';modelset.freeze(source,output)
    plan=modelset.validate_plan(modelset.read(output))
    for task in modelset.TASKS:
        assert [row['sha256'] for row in plan['models'][task]['cases']]==spec['models'][task]['case_sha256s']


@pytest.mark.parametrize('mutation,expected',[
    ('duplicate','explicit_case_count_or_duplicates'),
    ('test','explicit_case_not_validation'),
    ('missing_canonical','explicit_cases_omit_canonical'),
])
def test_explicit_cases_reject_invalid_selection(fixture_set,mutation,expected):
    _,_,_,rows=fixture_set;pool=rows['pothole'];canonical={pool[0]['sha256']}
    explicit=[row['sha256'] for row in pool if row['split']=='validation']
    if mutation=='duplicate':explicit[-1]=explicit[0]
    elif mutation=='test':explicit[-1]=pool[-1]['sha256']
    else:canonical={'not-in-validation'}
    with pytest.raises(ValueError,match=expected):modelset.select_cases('pothole',pool,canonical,explicit)
