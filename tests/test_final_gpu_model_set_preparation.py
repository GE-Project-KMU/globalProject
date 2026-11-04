"""Synthetic metadata coupons, no model/image/GPU inference or accuracy claim."""
import hashlib,json
from datetime import datetime,timedelta,timezone
from collections import Counter
from pathlib import Path
import pytest
from scripts import gpu_model_set as g
from scripts import prepare_final_gpu_model_set as p


@pytest.fixture
def case(tmp_path,monkeypatch):
    folder=tmp_path/'metadata';folder.mkdir();out=tmp_path/'output'
    monkeypatch.setattr(p,'OUTPUT_ROOT',out);monkeypatch.setattr(g,'PUBLIC_ROOTS',(folder,))
    monkeypatch.setattr(p,'PROTOCOL',folder/'preparation-protocol.json')
    expiry=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
    def save(name,value):
        path=folder/name;path.write_text(json.dumps(value));return g.pointer(path)
    def blob(name):
        path=folder/name;path.write_bytes(('SYNTHETIC-'+name).encode());return g.pointer(path)
    request={'format':p.FORMAT,'source_sha256':p.source_hashes(),'TEST_images':0,'GPU_used':False,'models':{}}
    request['preparation_protocol']=save('preparation-protocol.json',p.protocol())
    allrows={}
    for task,count in p.COUNTS.items():
        rows=[]
        for i in range(count):
            label='face' if task=='redaction' and i<400 else 'license_plate' if task=='redaction' else 'pothole'
            row={'sample_id':f'{task}-{i:04}','source_id':f'{count-i:04}',
                'path':(folder/(f'{task}-{i}.jpg')).relative_to(g.ROOT).as_posix(),
                'sha256':hashlib.sha256(f'SYNTHETIC:{task}:{i}'.encode()).hexdigest(),
                'expires_at':expiry,'split':'validation','objects':[{'class':label,'bbox_xywh_px':[0,0,5,5]}]}
            if task=='lane':row.update(recording='old' if i<100 else 'new',positive_pixels=10)
            elif task=='pothole':row['development_domain']='original' if i<402 else 'mwpd'
            else:row['evaluation_class']=label
            rows.append(row)
        allrows[task]=rows
        release=save(task+'-release.json',rows)
        weights=blob(task+'.onnx');checkpoint=blob(task+'.pt')
        provenance=save(task+'-provenance.json',{'task':task,'release_sha256':release['sha256'],'assets':[],'data_revoked':False,'needs_revalidation':False})
        model={'task':task,'input_size':[16,16],'weights':weights['path'],'sha256':weights['sha256'],
            'research_only':True,'production_approved':False,'source_release_path':release['path'],
            'training_provenance':provenance['path'],
            'dataset_release':'manifest-sha256:'+release['sha256'],'trained_checkpoint_sha256':checkpoint['sha256']}
        entry={'manifest':save(task+'-model.json',model),'checkpoint':checkpoint,'full_release':release,
            'training_provenance':provenance,
            'development_manifest':release,'cpu_verifications':[]}
        indices=[0,1,402] if task=='pothole' else [0,1,100] if task=='lane' else [0,400]
        for n,i in enumerate(indices):
            proof={'task':task,'manifest_sha256':entry['manifest']['sha256'],'weight_sha256':weights['sha256'],
                'checkpoint_sha256':checkpoint['sha256'],'synthetic_input':False,'image_sha256':rows[i]['sha256'],
                'torch_checkpoint_compared':True,'artifact_contract_verified':True,'numerical_parity_passed':True,
                'postprocess_parity_passed':True,'app_backend':'onnxruntime-cpu','app_backend_version':'1.31.1',
                'metadata_valid':True,'weight_hash_verified':True,'finite_output':True,'preprocess_contract_verified':True,
                'output_shape':[1,2,16,16] if task=='lane' else [1,3,7 if task=='redaction' else 6],
                'adapter_result_count':1,'lane_adapter_postprocess_matches_reference':True,
                'mask_agreement':{key:{'pixel_agreement':1.} for key in ('app','ort','torch')},
                'detection_postprocess_parity':{key:{'passed':True} for key in ('ort','torch')},
                **{key:{'passed':True,'rtol':.001,'atol':.0001} for key in ('app_repeatability','app_vs_ort','app_vs_torch','ort_vs_torch')}}
            entry['cpu_verifications'].append(save(f'{task}-canonical-{n}.json',proof))
        if task=='lane':
            plan={'manifest_sha256':entry['manifest']['sha256'],'checkpoint_sha256':checkpoint['sha256'],
                'development_sha256':release['sha256'],'rtol':.001,'atol':.0001,'test_accessed':False,
                'cases':[{k:row[k] for k in ('path','sha256','expires_at')} for row in sorted(rows,key=lambda row:row['source_id'])]}
            planref=save('lane-full-plan.json',plan)
            full={'manifest_sha256':entry['manifest']['sha256'],'checkpoint_sha256':checkpoint['sha256'],
                'weights_sha256':weights['sha256'],'plan_sha256':planref['sha256'],
                'worker_source_sha256':g.sha256(p.ROOT/'scripts/check_lane_cycle2_runtime.py'),
                'complete':True,'all_raw_passed':True,'all_post_passed':True,'test_accessed':False,'rtol':.001,'atol':.0001,
                'reference_runtime':{'ort':'1.31.1','providers':['CPUExecutionProvider'],'GPU_initialized':False},
                'app_runtime':{'engine':'onnxruntime-cpu','version':'1.31.1'},'cases':[]}
            for i,row in enumerate(plan['cases']):
                full['cases'].append({'index':i,'image_sha256':row['sha256'],'raw_passed':True,'post_passed':True,
                    'raw':{key:{'passed':True,'rtol':.001,'atol':.0001} for key in ('app_vs_original_FP32_torch','app_vs_reference_ORT131','reference_ORT131_vs_original_FP32_torch')},
                    'post':{key:{'exact_match':True,'mismatching_pixels':0} for key in ('app','torch','ort')}})
            entry.update(full_validation=save('lane-full.json',full),full_validation_plan=planref,
                full_validation_source_plan=save('lane-source-plan.json',{'validation_count':179,'development_sha256':release['sha256'],'release_sha256':release['sha256']}))
        request['models'][task]=entry
    path=folder/'request.json';path.write_text(json.dumps(request))
    return folder,out,path,request,allrows,save


def test_metadata_only_subsets_preserve_original_rows_and_match_existing_gpu_selector(case,monkeypatch):
    folder,out,path,request,rows,_=case
    original=g.sha256
    def no_images(value):
        assert str(value).lower().endswith(('.json','.py','.pt','.onnx')),'image bytes were read'
        return original(value)
    monkeypatch.setattr(g,'sha256',no_images)
    result=p.prepare(path,out/'final')
    assert result['counts']==g.COUNTS and result['plan_frozen'] is False and result['image_bytes_read']==0
    spec=g.read(out/'final/input-specification.json')
    for task,entry in spec['models'].items():
        selected=g.pinned(entry['validation_manifest']);canon={g.pinned(ref)['image_sha256'] for ref in entry['cpu_verifications']}
        assert len(selected)==g.COUNTS[task] and all(row in rows[task] for row in selected)
        assert g.select_cases(task,selected,canon,entry['case_sha256s'])==selected
    assert Counter(g.pinned(spec['models']['lane']['validation_manifest'])[i]['recording'] for i in range(100))=={'old':50,'new':50}
    with pytest.raises(ValueError,match='fresh_final_modelset_child'):p.prepare(path,out/'final')


@pytest.mark.parametrize('problem',['source','missing_coupon','canonical_fail','checkpoint','full179_fail','full179_index','full179_source','full179_membership','wrong_count','wrong_GT','expiry','privacy_group','parent_row','TEST','revoked','expired_asset'])
def test_incomplete_or_changed_evidence_refuses_before_output(case,problem):
    folder,out,path,request,rows,save=case
    task='lane' if problem.startswith('full179') else 'redaction' if problem=='privacy_group' else 'pothole'
    entry=request['models'][task]
    if problem=='source':request['source_sha256']['agent/models.py']='0'*64
    elif problem=='missing_coupon':entry['cpu_verifications'].pop()
    elif problem in ('canonical_fail','checkpoint'):
        proof=g.pinned(entry['cpu_verifications'][0]);proof['artifact_contract_verified']=False if problem=='canonical_fail' else True
        if problem=='checkpoint':proof['checkpoint_sha256']='0'*64
        entry['cpu_verifications'][0]=save('bad-proof.json',proof)
    elif problem.startswith('full179'):
        full=g.pinned(entry['full_validation'])
        if problem=='full179_fail':full['cases'][17]['raw']['app_vs_original_FP32_torch']['passed']=False
        elif problem=='full179_index':full['cases'][17]['index']=16
        elif problem=='full179_source':full['worker_source_sha256']='0'*64
        else:full['cases'][17]['image_sha256']='0'*64
        entry['full_validation']=save('bad-full.json',full)
    elif problem=='TEST':request['TEST_images']=1
    elif problem in ('revoked','expired_asset'):
        provenance=g.pinned(entry['training_provenance'])
        if problem=='revoked':provenance['data_revoked']=True
        else:provenance['assets']=[{'expires_at':'2020-01-01T00:00:00Z'}]
        entry['training_provenance']=save(Path(entry['training_provenance']['path']).name,provenance)
    else:
        changed=json.loads(json.dumps(rows[task]))
        if problem=='wrong_count':changed.pop()
        elif problem=='wrong_GT':changed[0]['objects']=[]
        elif problem=='expiry':changed[0]['expires_at']='2020-01-01T00:00:00Z'
        elif problem=='privacy_group':changed[400]['evaluation_class']='face'
        else:changed[0]['path']='data/public/altered.jpg'
        entry['development_manifest']=save('changed-development.json',changed)
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError):p.prepare(path,out/'must-not-exist')
    assert not (out/'must-not-exist').exists()


def test_lane50_each_and_pothole10_each_keep_canonical_first_without_MUTATING_parent(case):
    _,_,_,request,rows,_=case
    original=json.dumps(rows,sort_keys=True)
    for task in ('pothole','lane'):
        canon={g.pinned(ref)['image_sha256'] for ref in request['models'][task]['cpu_verifications']}
        selected=p.select(task,rows[task],canon)
        group='recording' if task=='lane' else 'development_domain'
        assert sorted(Counter(row[group] for row in selected).values())==([50,50] if task=='lane' else [10,10])
        for label in {row[group] for row in selected}:
            grouped=[row for row in selected if row[group]==label]
            prefix=[row for row in grouped if row['sha256'] in canon]
            assert grouped[:len(prefix)]==prefix
    assert json.dumps(rows,sort_keys=True)==original


def test_nonpositive_required_coupon_and_old_lane100_are_rejected(case):
    _,_,_,request,rows,_=case
    canonical={g.pinned(ref)['image_sha256'] for ref in request['models']['pothole']['cpu_verifications']}
    rows['pothole'][0]['objects']=[]
    with pytest.raises(ValueError,match='canonical_not_positive'):p.select('pothole',rows['pothole'],canonical)
    with pytest.raises(ValueError,match='FULLVAL179'):p.select('lane',rows['lane'][:100],set())


def test_legacy_lane_manifest_without_checkpoint_field_uses_all_bound_proofs(case):
    _,_,_,request,_,save=case;entry=request['models']['lane']
    model=g.pinned(entry['manifest']);model.pop('trained_checkpoint_sha256')
    ref=save('lane-no-checkpoint-model.json',model);entry['manifest']=ref
    for index,proof_ref in enumerate(entry['cpu_verifications']):
        proof=g.pinned(proof_ref);proof['manifest_sha256']=ref['sha256'];entry['cpu_verifications'][index]=save(f'new-lane-proof{index}.json',proof)
    plan=g.pinned(entry['full_validation_plan']);plan['manifest_sha256']=ref['sha256'];entry['full_validation_plan']=save('new-full-plan.json',plan)
    full=g.pinned(entry['full_validation']);full['manifest_sha256']=ref['sha256'];full['plan_sha256']=entry['full_validation_plan']['sha256'];entry['full_validation']=save('new-full.json',full)
    selected,_=p.validate_entry('lane',entry);assert len(selected)==100
