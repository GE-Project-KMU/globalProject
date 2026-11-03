"""Sealed-test-safe CPU export and validation-only inference option selection.

Torch is imported only by the explicit CPU export helper. Final training
model.json, summaries, logs and TEST images/predictions are never inputs.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,re,sys,time,uuid
from pathlib import Path
from datetime import datetime,timezone
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CLASSES=('face','license_plate')
GRID=(.01,.03,.05,.1,.2,.35,.5)
EXPECTED_RELEASE='4eeae0afdf67fcaf727e3e8b3a9b35486ef6f1126ae486da64637a7aec5bbd16'
STANDARD_FACTORY='9d26b063b22b54b1606cb33be1291b6e71f57945849aab21119764e1c847ca5d'
TILED_FACTORY='f8a1fe9121ad36cbab1e14a1f2477dd821ec6f04ee7bd6fd862996967dbd376f'
TILED_ARCHITECTURE='two YOLOX-Nano specialists global plus four tiles'
EXPECTED_COUNTS={'face':400,'license_plate':108}
SAFE_TRAINING_FILES=('training-config.json','candidate-freeze.json','confidence-calibration.json','provenance.json','best.pt','model.onnx')

def scoped(value,*,exists=False):
    path=Path(value);path=path if path.is_absolute() else ROOT/path
    cursor=path
    while cursor!=ROOT and cursor!=cursor.parent:
        if cursor.is_symlink() or getattr(cursor,'is_junction',lambda:False)():raise ValueError('links_not_allowed')
        cursor=cursor.parent
    path=path.resolve()
    if not path.is_relative_to(ROOT):raise ValueError('path_outside_workspace')
    if exists and not path.is_file():raise ValueError('required_file_missing')
    return path

def rel(path):return scoped(path).relative_to(ROOT).as_posix()
def sha(path):
    digest=hashlib.sha256()
    with scoped(path,exists=True).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def pinned(value):
    if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value):raise ValueError('pinned_sha256_required')
    return value
def json_snapshot(path,expected=None):
    path=scoped(path,exists=True)
    if path.stat().st_size>40_000_000:raise ValueError('json_metadata_too_large')
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if expected is not None and digest!=pinned(expected):raise ValueError('pinned_file_changed')
    return json.loads(raw),digest
def atomic_json(path,value):
    path=scoped(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)
def now():return datetime.now(timezone.utc).isoformat()
def date(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('timezone_required')
    return result

@contextmanager
def lock(path):
    path=scoped(path);path.parent.mkdir(parents=True,exist_ok=True);token=uuid.uuid4().hex
    try:fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:raise ValueError('option_operation_already_locked')
    def verify():
        if not path.is_file() or path.read_text(encoding='ascii')!=token:raise ValueError('option_lock_changed')
    try:
        with os.fdopen(fd,'w',encoding='ascii') as stream:stream.write(token)
        yield verify
        verify()
    finally:
        if path.exists() and path.read_text(encoding='ascii')==token:path.unlink()

def load_validation(path,expected_sha,expected_counts=None):
    expected_counts=EXPECTED_COUNTS if expected_counts is None else expected_counts
    rows,_=json_snapshot(path,expected_sha)
    if not isinstance(rows,list) or len(rows)!=sum(expected_counts.values()):raise ValueError('validation_count_mismatch')
    counts={label:0 for label in CLASSES};ids=set();deadlines=[]
    for row in rows:
        if not isinstance(row,dict) or row.get('split')!='validation':raise ValueError('validation_only_required')
        label=row.get('evaluation_class')
        if label not in CLASSES:raise ValueError('known_class_scope_required')
        if row.get('sample_id') in ids or not isinstance(row.get('sample_id'),str):raise ValueError('unique_validation_sample_id_required')
        ids.add(row['sample_id']);counts[label]+=1
        if row.get('privacy_status')!='restricted_research' or row.get('human_privacy_reviewed') is not False or row.get('not_for_deployment') is not True:raise ValueError('restricted_research_scope_required')
        if row.get('qc_status')!='publisher_annotations_machine_checked' or row.get('release_scope')!='offline_research':raise ValueError('publisher_research_qc_required')
        expiry=date(row['expires_at'])
        if expiry<=datetime.now(timezone.utc):raise ValueError('validation_source_expired')
        deadlines.append(expiry)
        scoped(row['path']);pinned(row['sha256'])
        width,height=row['image']['width'],row['image']['height']
        if type(width) is not int or type(height) is not int or min(width,height)<1:raise ValueError('invalid_source_dimensions')
        if not row.get('objects'):raise ValueError('known_class_ground_truth_required')
        for item in row['objects']:
            box=item.get('bbox_xywh_px')
            if item.get('class')!=label or not isinstance(box,list) or len(box)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box):raise ValueError('invalid_known_class_ground_truth')
            x,y,w,h=box
            if min(x,y)<0 or min(w,h)<=0 or x+w>width or y+h>height:raise ValueError('ground_truth_outside_source')
    if counts!=expected_counts:raise ValueError('validation_class_counts_mismatch')
    return rows,counts,min(deadlines).isoformat()

def provenance_core(value):return {key:item for key,item in value.items() if key!='models'}

def check_provenance(value,release_sha):
    if value.get('task')!='redaction' or value.get('release_sha256')!=release_sha or value.get('test_split_exported') is not False or value.get('production_approved') is not False:raise ValueError('new_training_provenance_required')
    if value.get('data_revoked') or value.get('needs_revalidation') or not value.get('assets'):raise ValueError('provenance_blocked_or_empty')
    for asset in value['assets']:
        if date(asset['expires_at'])<=datetime.now(timezone.utc) or asset.get('revoked_at'):raise ValueError('training_source_expired_or_revoked')
        if not asset.get('identity_hashes') or any(pinned(v)!=v for v in asset['identity_hashes']):raise ValueError('source_identity_hash_required')

def inspect_safe_inputs(training_dir,release,release_sha,validation_manifest,validation_sha,expected_counts=None):
    training_dir=scoped(training_dir);release=scoped(release,exists=True);validation_manifest=scoped(validation_manifest,exists=True)
    if not training_dir.is_relative_to(ROOT/'data/training'):raise ValueError('training_directory_scope_required')
    if release.name!='release.json' or validation_manifest.name!='validation-only.json':raise ValueError('safe_release_metadata_name_required')
    if sha(release)!=pinned(release_sha):raise ValueError('release_hash_mismatch')
    files={name:training_dir/name for name in SAFE_TRAINING_FILES}
    hashes={name:sha(path) for name,path in files.items()}
    config,_=json_snapshot(files['training-config.json'],hashes['training-config.json'])
    receipt,_=json_snapshot(files['candidate-freeze.json'],hashes['candidate-freeze.json'])
    calibration,_=json_snapshot(files['confidence-calibration.json'],hashes['confidence-calibration.json'])
    provenance,_=json_snapshot(files['provenance.json'],hashes['provenance.json'])
    if config.get('release_sha256')!=release_sha or config.get('architecture')!='two YOLOX-Nano specialists' or config.get('research_only') is not True or config.get('production_approved') is not False:raise ValueError('frozen_training_configuration_mismatch')
    if any(type(config.get(key)) is not int or config[key]!=value for key,value in {'shared_input_size':960,'face_input_size':960,'plate_input_size':640}.items()):raise ValueError('frozen_training_geometry_mismatch')
    if config.get('confidence_grid')!=list(GRID) or config.get('nms_iou')!=.5 or config.get('minimum_recall_goal')!=.95:raise ValueError('frozen_validation_policy_mismatch')
    if config.get('factory_sha256')!=STANDARD_FACTORY or sha(ROOT/'agent/redaction_network.py')!=STANDARD_FACTORY or sha(ROOT/'agent/redaction_tiled_network.py')!=TILED_FACTORY:raise ValueError('frozen_factory_changed')
    if config.get('training_script_sha256')!=sha(ROOT/'scripts/improve_redaction_train.py'):raise ValueError('frozen_training_source_changed')
    if receipt.get('checkpoint_sha256')!=hashes['best.pt'] or receipt.get('onnx_sha256')!=hashes['model.onnx'] or receipt.get('calibration_sha256')!=hashes['confidence-calibration.json']:raise ValueError('candidate_receipt_hash_mismatch')
    pinned(receipt.get('model_manifest_sha256')) # hash only: original model.json stays sealed
    if receipt.get('test_evaluated') is not False:raise ValueError('pretest_candidate_receipt_required')
    if calibration.get('grid')!=list(GRID) or calibration.get('minimum_recall_goal')!=.95 or 'validation' not in str(calibration.get('selection_data','')).lower():raise ValueError('validation_only_calibration_required')
    if calibration.get('selected_confidence') not in GRID:raise ValueError('validation_calibration_point_required')
    check_provenance(provenance,release_sha)
    if provenance.get('release_path')!=rel(release):raise ValueError('new_training_provenance_required')
    rows,counts,expiry=load_validation(validation_manifest,validation_sha,expected_counts)
    proof_path=validation_manifest.with_name('validation-only.source.json');proof,proof_sha=json_snapshot(proof_path)
    if proof.get('parent_manifest_sha256')!=release_sha or proof.get('validation_manifest_sha256')!=validation_sha or proof.get('only_split')!='validation' or proof.get('source_expiries_not_extended') is not True:raise ValueError('validation_parent_binding_mismatch')
    if proof.get('counts')!=counts or date(proof['source_expiry_min'])!=date(expiry) or proof.get('training_manifest_unchanged') is not True or proof.get('test_images_or_predictions_opened') is not False:raise ValueError('validation_source_proof_mismatch')
    return {'training_dir':training_dir,'release':release,'release_sha':release_sha,'validation_manifest':validation_manifest,'validation_sha':validation_sha,'validation_counts':counts,'validation_expiry':expiry,'proof_path':proof_path,'proof_sha':proof_sha,'files':files,'hashes':hashes,'config':config,'candidate_receipt':receipt,'provenance':provenance,'provenance_core_sha':canonical_sha(provenance_core(provenance)),'rows':rows}

def export_tiled_checkpoint(checkpoint,output):
    # This explicit helper runs in the isolated training Python, CPU only.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from agent.redaction_tiled_network import build_redaction_tiled_network
    torch.set_num_threads(2)
    value=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or not isinstance(value.get('model'),dict) or value.get('architecture')!='two YOLOX-Nano specialists':raise ValueError('shared_specialist_checkpoint_required')
    model=build_redaction_tiled_network().eval().cpu();model.load_state_dict(value['model'],strict=True)
    for parameter in model.parameters():parameter.requires_grad_(False)
    if any(parameter.device.type!='cpu' for parameter in model.parameters()):raise ValueError('cpu_export_required')
    with torch.inference_mode():torch.onnx.export(model,torch.zeros(1,3,1920,1920),str(output),input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False)

def link_option_model(provenance,manifest):
    from agent.governance import link_model
    link_model(provenance,manifest)

def option_manifest(inputs,option,weights):
    tiled=option=='tiled'
    value={'format':'onnx','task':'redaction','classes':list(CLASSES),'architecture':TILED_ARCHITECTURE if tiled else 'two YOLOX-Nano specialists','input_size':[1920,1920] if tiled else [960,960],'face_input_size':960,'plate_input_size':640,'weights':rel(weights),'sha256':sha(weights),'output_layout':'yolox_decoded_xywh','confidence':.01,'nms_iou':.5,'target_trained':True,'dataset_release':'manifest-sha256:'+inputs['release_sha'],'source_release_path':rel(inputs['release']),'training_provenance':rel(inputs['files']['provenance.json']),'research_only':True,'production_approved':False,'project_acceptance_eligible':False,'human_privacy_reviewed':False,'independent_route_verified':False,'source_licenses':['CC-BY-NC-ND-4.0','MIT'],'artifact_redistribution_allowed':False,'test_split_used':False,'test_split_used_scope':'training_and_validation_selection','data_revoked':False,'needs_revalidation':False,'trained_checkpoint_sha256':inputs['hashes']['best.pt'],'no_new_gradient_steps':True,'validation_manifest_sha256':inputs['validation_sha'],'inference_option':option,'option_selection_pending':True,'architecture_factory_sha256':TILED_FACTORY if tiled else STANDARD_FACTORY}
    if tiled:value.update(shared_input_size=1920,face_tile_grid=2)
    return value

def export_options(training_dir,release,release_sha,validation_manifest,validation_sha,output,*,expected_counts=None):
    output=scoped(output)
    if not output.is_relative_to(ROOT/'data/training') or output==ROOT/'data/training':raise ValueError('new_training_options_directory_required')
    with lock(output.with_name('.'+output.name+'.export.lock')) as verify_lock:
        if output.exists():raise ValueError('option_output_already_exists')
        inputs=inspect_safe_inputs(training_dir,release,release_sha,validation_manifest,validation_sha,expected_counts)
        output.mkdir(parents=True)
        standard=output/'option-standard/model.json';tiled=output/'option-tiled/model.json';tiled.parent.mkdir(parents=True)
        tiled_weights=tiled.parent/'model.onnx';export_tiled_checkpoint(inputs['files']['best.pt'],tiled_weights)
        # Refuse changes during export; neither source weights nor TTL is edited.
        if any(sha(path)!=inputs['hashes'][name] for name,path in inputs['files'].items()):raise ValueError('safe_training_input_changed_during_export')
        verify_lock()
        manifests={'standard':(standard,option_manifest(inputs,'standard',inputs['files']['model.onnx'])),'tiled':(tiled,option_manifest(inputs,'tiled',tiled_weights))}
        for path,value in manifests.values():atomic_json(path,value)
        for path,_ in manifests.values():link_option_model(inputs['files']['provenance.json'],path)
        current,_=json_snapshot(inputs['files']['provenance.json'])
        if canonical_sha(provenance_core(current))!=inputs['provenance_core_sha']:raise ValueError('provenance_assets_or_ttl_changed')
        inputs['hashes']['provenance.json']=sha(inputs['files']['provenance.json'])
        verify_lock()
        receipt={'format':'redaction-inference-option-export-v1','created_at':now(),'safe_training_inputs':{name:{'path':rel(path),'sha256':inputs['hashes'][name]} for name,path in inputs['files'].items()},'original_candidate_manifest_sha256_hash_only':inputs['candidate_receipt']['model_manifest_sha256'],'release_path':rel(inputs['release']),'release_sha256':release_sha,'validation_manifest':rel(inputs['validation_manifest']),'validation_manifest_sha256':validation_sha,'validation_proof_path':rel(inputs['proof_path']),'validation_proof_sha256':inputs['proof_sha'],'validation_counts':inputs['validation_counts'],'validation_source_expiry_min':inputs['validation_expiry'],'provenance_core_sha256':inputs['provenance_core_sha'],'options':{name:{'manifest':rel(path),'exported_manifest_sha256':sha(path),'weights':value['weights'],'onnx_sha256':value['sha256']} for name,(path,value) in manifests.items()},'confidence_grid':list(GRID),'nms_iou':.5,'minimum_class_recall_goal':.95,'selection_policy':'If either option reaches min-class recall>=.95, maximize macro precision then min recall; otherwise maximize min recall then macro precision. Exact ties prefer standard.','no_new_gradient_steps':True,'shared_checkpoint_sha256':inputs['hashes']['best.pt'],'test_inputs_read':False,'original_final_model_json_read':False,'source_expiries_changed':False}
        path=output/'export-receipt.json';atomic_json(path,receipt)
        return {'options_root':rel(output),'export_receipt_sha256':sha(path),'validation_counts':inputs['validation_counts'],'test_inputs_read':False}

def iou(a,b):
    x,y,w,h=a;xx,yy,ww,hh=b
    overlap=max(0.,min(x+w,xx+ww)-max(x,xx))*max(0.,min(y+h,yy+hh)-max(y,yy))
    total=w*h+ww*hh-overlap
    return overlap/total if total else 0.

def operating_metrics(truth,predictions,confidence):
    byclass={}
    for label in CLASSES:
        gt=[row for row in truth if row['class']==label];matched=set()
        for prediction in sorted((p for p in predictions if p['class']==label and p['confidence']>=confidence),key=lambda p:p['confidence'],reverse=True):
            score,index=max(((iou(prediction['bbox_xywh_px'],item['bbox_xywh_px']),i) for i,item in enumerate(gt) if i not in matched and item['sample_id']==prediction['sample_id']),default=(0.,None))
            if score>=.5:matched.add(index)
        count=sum(p['class']==label and p['confidence']>=confidence for p in predictions);tp=len(matched)
        if not gt:raise ValueError('both_validation_classes_required')
        byclass[label]={'gt_count':len(gt),'prediction_count':count,'tp':tp,'fp':count-tp,'fn':len(gt)-tp,'recall':tp/len(gt),'precision':tp/count if count else 0.}
    return {'confidence':confidence,'classes':byclass,'min_class_recall':min(v['recall'] for v in byclass.values()),'macro_precision':sum(v['precision'] for v in byclass.values())/2}

def calibrate(truth,predictions):
    scores=[operating_metrics(truth,predictions,t) for t in GRID];eligible=[r for r in scores if r['min_class_recall']>=.95]
    selected=max(eligible,key=lambda r:(r['macro_precision'],r['min_class_recall'],r['confidence'])) if eligible else max(scores,key=lambda r:(r['min_class_recall'],r['macro_precision'],r['confidence']))
    return {'selected_on':'validation_only','grid':list(GRID),'minimum_recall_goal':.95,'goal_achieved':bool(eligible),'scores':scores,'selected':selected}

def select_option(calibrations):
    if set(calibrations)!={'standard','tiled'}:raise ValueError('exactly_two_predeclared_options_required')
    points={name:value['selected'] for name,value in calibrations.items()}
    eligible=[name for name,row in points.items() if row['min_class_recall']>=.95]
    names=eligible or list(points)
    def key(name):
        r=points[name];metrics=(r['macro_precision'],r['min_class_recall']) if eligible else (r['min_class_recall'],r['macro_precision'])
        return (*metrics,name=='standard')
    return max(names,key=key)

def verified_export(output,receipt_sha):
    receipt,_=json_snapshot(output/'export-receipt.json',receipt_sha)
    if receipt.get('format')!='redaction-inference-option-export-v1' or receipt.get('confidence_grid')!=list(GRID) or receipt.get('nms_iou')!=.5 or receipt.get('minimum_class_recall_goal')!=.95:raise ValueError('invalid_export_receipt')
    if set(receipt.get('options',{}))!={'standard','tiled'} or set(receipt.get('safe_training_inputs',{}))!=set(SAFE_TRAINING_FILES):raise ValueError('invalid_safe_input_allowlist')
    for name,item in receipt['safe_training_inputs'].items():
        path=scoped(item['path'],exists=True)
        if path.name!=name or sha(path)!=pinned(item['sha256']):raise ValueError('safe_input_changed_after_export')
    if sha(receipt['release_path'])!=receipt['release_sha256'] or sha(receipt['validation_proof_path'])!=receipt['validation_proof_sha256']:raise ValueError('release_or_validation_binding_changed')
    provenance,_=json_snapshot(receipt['safe_training_inputs']['provenance.json']['path'])
    check_provenance(provenance,receipt['release_sha256'])
    if canonical_sha(provenance_core(provenance))!=receipt['provenance_core_sha256']:raise ValueError('provenance_assets_or_ttl_changed')
    for name,item in receipt['options'].items():
        path=scoped(item['manifest'],exists=True)
        if path!=output/('option-'+name)/'model.json' or sha(path)!=item['exported_manifest_sha256'] or sha(item['weights'])!=item['onnx_sha256']:raise ValueError('exported_option_changed')
        manifest,_=json_snapshot(path)
        if manifest.get('confidence')!=.01 or manifest.get('option_selection_pending') is not True or manifest.get('sha256')!=item['onnx_sha256'] or manifest.get('weights')!=item['weights']:raise ValueError('validation_option_manifest_changed')
        from scripts.verify_trained_models import validate_architecture
        validate_architecture(manifest)
    return receipt

def app_model(manifest):
    from agent.models import OnnxModel
    return OnnxModel(manifest)

def decode_validation_image(row):
    import numpy as np,cv2
    raw=scoped(row['path'],exists=True).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('validation_image_hash_mismatch')
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.shape[:2]!=(row['image']['height'],row['image']['width']):raise ValueError('validation_source_decode_or_shape_failed')
    return image

def percentiles(values):
    import numpy as np
    return {'first':values[0],'median':float(np.median(values)),'p95':float(np.percentile(values,95))}

def validate_options(output,receipt_sha,*,public_report=None):
    output=scoped(output);public_report=scoped(public_report) if public_report is not None else None
    if not output.is_dir():raise ValueError('exported_options_directory_required')
    with lock(output/'.selection.lock') as verify_lock:
        if any((output/name).exists() for name in ('option-selection-freeze.json','validation-details.json','validation-aggregate.json')) or public_report is not None and public_report.exists():raise ValueError('selection_output_already_exists')
        receipt=verified_export(output,receipt_sha)
        rows,counts,expiry=load_validation(receipt['validation_manifest'],receipt['validation_manifest_sha256'],receipt['validation_counts'])
        models={name:app_model(scoped(item['manifest'])) for name,item in receipt['options'].items()}
        truth=[];predictions={name:[] for name in models};timings={name:[] for name in models};ignored={name:0 for name in models}
        for row in rows:
            image=decode_validation_image(row);known=row['evaluation_class']
            truth.extend({'sample_id':row['sample_id'],'class':known,'bbox_xywh_px':item['bbox_xywh_px']} for item in row['objects'])
            for name,model in models.items():
                start=time.perf_counter();result=model.predict(image);timings[name].append(time.perf_counter()-start)
                for item in result:
                    if item['class']!=known:ignored[name]+=1;continue
                    if not math.isfinite(item['confidence']) or not 0<=item['confidence']<=1 or item['confidence']<.01:raise ValueError('invalid_validation_prediction')
                    box=item.get('bbox_xywh_px')
                    if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box) or min(box[:2])<0 or min(box[2:])<=0 or box[0]+box[2]>image.shape[1] or box[1]+box[3]>image.shape[0]:raise ValueError('invalid_validation_prediction_box')
                    predictions[name].append({'sample_id':row['sample_id'],'class':known,'bbox_xywh_px':item['bbox_xywh_px'],'confidence':item['confidence']})
        calibrations={name:calibrate(truth,value) for name,value in predictions.items()};selected=select_option(calibrations)
        # Recheck every sealed input before changing only the new option metadata.
        verified_export(output,receipt_sha)
        verify_lock()
        if sha(receipt['validation_manifest'])!=receipt['validation_manifest_sha256']:raise ValueError('validation_manifest_changed_during_prediction')
        final={}
        for name,item in receipt['options'].items():
            path=scoped(item['manifest']);manifest,_=json_snapshot(path)
            manifest.update(confidence=calibrations[name]['selected']['confidence'],option_selection_pending=False,confidence_calibration={'selected_on':'validation','validation_manifest_sha256':receipt['validation_manifest_sha256'],'grid':list(GRID),'minimum_class_recall_goal':.95},selected_inference_option=name==selected)
            atomic_json(path,manifest)
            final[name]={'manifest':rel(path),'model_manifest_sha256':sha(path),'export_stage_manifest_sha256':item['exported_manifest_sha256'],'onnx_sha256':item['onnx_sha256'],'confidence':manifest['confidence'],'selected':name==selected}
        aggregate={'format':'redaction-inference-option-validation-v1','evaluated_at':now(),'source_release_sha256':receipt['release_sha256'],'validation_manifest_sha256':receipt['validation_manifest_sha256'],'validation_counts':counts,'validation_source_expiry_min':expiry,'known_class_conditional_evaluation':True,'unknown_other_class_predictions_not_scored':ignored,'calibrations':calibrations,'selected_option':selected,'exact_ties_prefer_standard':True,'timing_seconds':{name:percentiles(value) for name,value in timings.items()},'backends':{name:{'engine':getattr(model,'backend','custom-test-double'),'version':getattr(model,'backend_version',None)} for name,model in models.items()},'no_new_gradient_steps':True,'shared_checkpoint_sha256':receipt['shared_checkpoint_sha256'],'source_expiries_changed':False,'test_images_or_metrics_read':False,'project_acceptance_passed':False,'production_approved':False,'limits':['CPU application predict latency under shared host load, not full pipeline throughput.','Known-class source annotations only; unknown other-class predictions are not false positives and exhaustive privacy coverage is not established.','Both source expiry and redistribution restrictions remain in effect.']}
        details={'scope':'private_validation_only','source_expires_at':expiry,'truth':truth,'predictions':predictions}
        atomic_json(output/'validation-details.json',details);atomic_json(output/'validation-aggregate.json',aggregate)
        verify_lock()
        freeze={'format':'redaction-inference-option-selection-freeze-v1','frozen_at':now(),'export_receipt_sha256':pinned(receipt_sha),'validation_manifest_sha256':receipt['validation_manifest_sha256'],'source_release_sha256':receipt['release_sha256'],'selected_option':selected,'options':final,'confidence_grid':list(GRID),'nms_iou':.5,'selection_policy':receipt['selection_policy'],'validation_aggregate_sha256':sha(output/'validation-aggregate.json'),'validation_details_sha256':sha(output/'validation-details.json'),'test_images_or_metrics_read':False,'no_new_gradient_steps':True,'shared_checkpoint_sha256':receipt['shared_checkpoint_sha256'],'production_approved':False}
        atomic_json(output/'option-selection-freeze.json',freeze)
        if public_report is not None:atomic_json(public_report,aggregate)
        return freeze

def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    export=commands.add_parser('export');export.add_argument('--training-dir',required=True);export.add_argument('--release',required=True);export.add_argument('--release-sha256',default=EXPECTED_RELEASE);export.add_argument('--validation-manifest',required=True);export.add_argument('--validation-sha256',required=True);export.add_argument('--output',required=True)
    validation=commands.add_parser('validate');validation.add_argument('--options-root',required=True);validation.add_argument('--export-receipt-sha256',required=True);validation.add_argument('--public-report')
    args=parser.parse_args()
    try:
        result=export_options(args.training_dir,args.release,args.release_sha256,args.validation_manifest,args.validation_sha256,args.output) if args.command=='export' else validate_options(args.options_root,args.export_receipt_sha256,public_report=args.public_report)
        print(json.dumps(result,indent=2));return 0
    except Exception as error:
        reason=str(error);reason=reason if re.fullmatch('[a-z0-9_]{1,100}',reason) else 'option_operation_failed'
        print(json.dumps({'status':'option_operation_refused','error_type':type(error).__name__,'reason':reason}));return 2

if __name__=='__main__':raise SystemExit(main())
