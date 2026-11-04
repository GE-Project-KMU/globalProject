"""Predeclared CPU-only inference geometry trial; no new training or TEST."""
import argparse,hashlib,json,os,shutil,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
OUT=ROOT/'data/training/pothole-resolution-v10'
NANO=ROOT/'data/training/pothole-localization-v8'
SIZES=(960,1280)
DEPS=('scripts/verify_trained_models.py','agent/models.py','agent/evaluate.py','agent/dataset.py')

def read(p):return json.loads(workspace_path(p,exists=True).read_text(encoding='utf-8'))
def pointer(p):
    p=workspace_path(p,exists=True);return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha256(p)}
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validation_rows():
    rows=[r for r in read(NANO/'release.json') if r['split']=='validation']
    if Counter(r['development_domain'] for r in rows)!={'original':402,'mwpd':250}:raise ValueError('exact_VAL652_required')
    return sorted(rows,key=lambda r:r['sample_id'])
def membership(rows):return canonical([{k:r[k] for k in ('sample_id','sha256','development_domain')} for r in rows])
def check_plan():
    plan=read(OUT/'trial-plan.json')
    if sha256(Path(__file__))!=plan['source_sha256'] or any(sha256(ROOT/p)!=v for p,v in plan['dependencies'].items()):raise ValueError('frozen_resolution_sources_changed')
    for key in ('parent_manifest','checkpoint','validation_release','parent_family_policy','baseline_manifest','small_learning_plan','small_retry_protocol'):
        p=plan[key]
        if sha256(p['path'])!=p['sha256']:raise ValueError('resolution_parent_input_changed')
    if membership(validation_rows())!=plan['VAL_membership_sha256']:raise ValueError('fixed_VAL_membership_changed')
    return plan

def prepare():
    if OUT.exists():raise ValueError('fresh_inference_trial_required')
    rows=validation_rows();expiry=min(r['expires_at'] for r in rows)
    plan={'format':'pothole-resolution-v10-predeclared-plan-v1','source_sha256':sha256(Path(__file__)),'dependencies':{p:sha256(ROOT/p) for p in DEPS},
          'parent_manifest':pointer(NANO/'model-development.json'),'checkpoint':pointer(NANO/'best.pt'),'validation_release':pointer(NANO/'release.json'),
          'parent_family_policy':pointer('docs/reports/POTHOLE-FAMILY-V9-ONE-SHOT-PLAN.json'),'baseline_manifest':pointer('data/training/pothole-mwpd-v6/model.json'),
          'small_learning_plan':pointer('data/training/pothole-small-v9-retry1/plan.json'),'small_retry_protocol':pointer('data/training/pothole-small-v9-retry1/retry-plan.json'),
          'input_sizes':[640,*SIZES],'new_exports':list(SIZES),'parameter_or_checkpoint_changes':False,'new_training_updates':0,
          'rationale':'VAL localization-overlap errors motivate a bounded inference-resolution trial; larger input need not improve and is never selected from TEST',
          'VAL_images':652,'domains':{'original':402,'mwpd':250},'VAL_membership_sha256':membership(rows),'source_expires_at':expiry,'source_expiry_extended':False,
          'CPU_family_members':['baseline','nano','small','nano960','nano1280'],'selection':'Same actual-app ORTCPU1.31.1 VAL for every available member; oldAP>=CPUbaseline-.03; maximum mean two-domain AP50; tie baseline,nano,small,nano960,nano1280',
          'confidence_floor':.01,'operating_confidence':.1,'nms_iou':.5,'TEST_images':0,'TEST_used_for_choice':False,'user_recordings_used':False,'GPU_used':False,'production_approved':False}
    OUT.mkdir();atomic_json(OUT/'trial-plan.json',plan);atomic_json(ROOT/'docs/reports/POTHOLE-RESOLUTION-V10-PLAN.json',plan)
    print(json.dumps({'plan_sha256':sha256(OUT/'trial-plan.json'),'inference_performed':False}))

def export():
    os.environ['CUDA_VISIBLE_DEVICES']='';plan=check_plan()
    import torch
    from scripts.verify_trained_models import build_torch_reference,validate_metadata
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise ValueError('CPU_export_only')
    parent=read(plan['parent_manifest']['path']);model=build_torch_reference(parent,ROOT/'.cache/vendor/YOLOX').cpu().eval()
    model.load_state_dict(torch.load(workspace_path(plan['checkpoint']['path']),map_location='cpu',weights_only=True)['model'],strict=True)
    result={}
    for size in SIZES:
        folder=OUT/f'nano{size}'
        if folder.exists():raise ValueError('preserve_existing_resolution_export')
        folder.mkdir();shutil.copyfile(workspace_path(plan['checkpoint']['path']),folder/'best.pt')
        sample=torch.zeros(1,3,size,size)
        with torch.inference_mode():raw=model(sample)
        expected=sum((size//stride)**2 for stride in (8,16,32))
        if list(raw.shape)!=[1,expected,6] or not torch.isfinite(raw).all():raise ValueError('finite_fixed_resolution_output_required')
        weight=folder/'model.onnx';torch.onnx.export(model,sample,str(weight),input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False)
        metadata={k:v for k,v in parent.items() if k not in ('executed_export_manifest_sha256','selection_freeze_sha256','packaging_source_sha256')}
        metadata.update(input_size=[size,size],weights=weight.relative_to(ROOT).as_posix(),sha256=sha256(weight),inference_parent_manifest_sha256=plan['parent_manifest']['sha256'],inference_option_plan_sha256=sha256(OUT/'trial-plan.json'),inference_option_new_training_updates=0)
        atomic_json(folder/'model.json',metadata);validate_metadata(folder/'model.json')
        result[f'nano{size}']={'manifest':pointer(folder/'model.json'),'checkpoint':pointer(folder/'best.pt'),'weights_sha256':metadata['sha256'],'input_size':[size,size],'output_shape':list(raw.shape)}
    check_plan();atomic_json(OUT/'exports.json',{'models':result,'plan_sha256':sha256(OUT/'trial-plan.json'),'checkpoint_unchanged':all(r['checkpoint']['sha256']==plan['checkpoint']['sha256'] for r in result.values()),'GPU_used':False,'TEST_images':0});print(json.dumps({'exported':list(result),'GPU_used':False,'TEST_images':0}))

def evaluate():
    import cv2,numpy as np,onnxruntime as ort
    from agent.models import OnnxModel
    from agent.dataset import validate_research_release
    from agent.evaluate import detection_metrics
    from scripts.evaluate_pothole_family import retry_contract,trained_member
    plan=check_plan();retry_contract();small=trained_member('data/training/pothole-small-v9-retry1');rows=validation_rows()
    if (OUT/'cpu-validation.json').exists() or (OUT/'cpu-validation-attempt.json').exists():raise ValueError('preserve_completed_or_interrupted_CPU_trial')
    if ort.__version__!='1.31.1' or not validate_research_release(rows)['valid']:raise ValueError('current_licensed_hashed_VAL_and_ORT1311_required')
    manifests={'baseline':'data/training/pothole-mwpd-v6/model.json','nano':str(NANO/'model-development.json')}
    if small['status']=='available':manifests['small']=small['manifest']['path']
    manifests.update({f'nano{s}':str(OUT/f'nano{s}'/'model.json') for s in SIZES})
    atomic_json(OUT/'cpu-validation-attempt.json',{'plan_sha256':sha256(OUT/'trial-plan.json'),'TEST_images':0,'status':'started'});cv2.setNumThreads(1);results={};truth={d:[dict(o,sample_id=r['sample_id']) for r in rows if r['development_domain']==d for o in r['objects']] for d in ('original','mwpd')}
    for name,path in manifests.items():
        model=OnnxModel(path,cpu_threads=2,allow_spinning=False);predictions={d:[] for d in truth};artifact=dict(model.manifest);p=workspace_path(path);start=time.perf_counter()
        proof={'status':'available','manifest':pointer(p),'checkpoint':pointer(p.parent/'best.pt'),'weights_sha256':artifact['sha256'],'input_size':artifact['input_size'],'domains':{}}
        try:
            if model.net.session.get_providers()!=['CPUExecutionProvider'] or model.backend_version!='1.31.1':raise ValueError('CPU_only_exact_runtime_required')
            model.manifest=dict(artifact,confidence=.01)
            for row in rows:
                if datetime.fromisoformat(row['expires_at'].replace('Z','+00:00'))<=datetime.now(timezone.utc):raise ValueError('source_expired_during_VAL')
                raw=workspace_path(row['path'],exists=True).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('VAL_source_changed_before_decode')
                image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);found=model.predict(image)
                predictions[row['development_domain']].extend(dict(item,sample_id=row['sample_id']) for item in found)
            for domain in truth:proof['domains'][domain]={'images':sum(r['development_domain']==domain for r in rows),'ap50':detection_metrics(truth[domain],predictions[domain],['pothole'])['map50']}
            if sha256(p)!=proof['manifest']['sha256'] or sha256(artifact['weights'])!=proof['weights_sha256']:raise ValueError('evaluated_artifact_changed')
        finally:model.close()
        proof['elapsed_seconds']=time.perf_counter()-start;results[name]=proof;atomic_json(OUT/f'{name}-VAL-predictions.json',{'truth':truth,'predictions':predictions,'TEST_images':0});print(json.dumps({'member':name,'domains':proof['domains']}),flush=True)
    if small['status']!='available':results['small']={'status':'rejected_no_artifact','selection_freeze':small['selection_freeze']}
    check_plan();result={'format':'pothole-resolution-v10-CPU-VAL652','complete':True,'plan_sha256':sha256(OUT/'trial-plan.json'),'validation_release_sha256':sha256(NANO/'release.json'),'VAL_membership_sha256':membership(rows),'runtime':{'engine':'onnxruntime-cpu','version':'1.31.1','providers':['CPUExecutionProvider'],'GPU_used':False},'TEST_images':0,'confidence_floor':.01,'nms_iou':.5,'models':results}
    atomic_json(OUT/'cpu-validation.json',result);atomic_json(ROOT/'docs/reports/POTHOLE-RESOLUTION-V10-CPU-VAL.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','export','evaluate']);a=p.parse_args();{'prepare':prepare,'export':export,'evaluate':evaluate}[a.action]()
