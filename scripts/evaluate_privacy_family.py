"""Independent complete-VAL family freeze and strict-gated one-shot benchmarks.

Preparation/selection are metadata-only. Pixel inference requires an explicit
operator flag after the Rescue run completes. No current TRAIN/TEST execution.
"""
from __future__ import annotations
import argparse,copy,json,sys,math
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from scripts.redaction_rescue_data import load_development,read_image
from scripts.train_redaction_rescue_v8 import calibrate,large_recall,family_winner,grouped_match

DEST=ROOT/'data/training/privacy-family-v8-evaluation'
BASE=ROOT/'data/training/redaction-inference-options-v4/option-tiled/model.json'
BASE_CHECKPOINT=ROOT/'data/training/redaction-specialists-v4-efficient/best.pt'
RESCUE=ROOT/'data/training/redaction-rescue-v8'
FRESH=ROOT/'data/training/research-redaction-cycle2-v7/fresh-test.json'
REGRESSION=ROOT/'data/training/cycle2-evaluation/privacy-regression-manifest.json'
PUBLIC=ROOT/'docs/reports/PRIVACY-FAMILY-V8-PROTOCOL.json'
P2_PARENT=ROOT/'docs/reports/PRIVACY-P2-CYCLE2-PLAN-V2.json'
PIN={'baseline_manifest':'971dd630295aaffce09760b73ba07c080e92a65a76daad5d4a91bf0a14642edc',
     'baseline_checkpoint':'6c985f7e3afb0fa061cbb82d7ec7b304eb5f9fbfcd0cf9f775dbab28873c82ad',
     'development':'37cdd77565438c1254768347a7d7abe37c686ba20239c3ade2fda23cd98a8d0c',
     'fresh':'8230874168c8b2d4d8de09c5c11cb3197f641c0ef6a28999ef77e3ea2465c9a7',
     'regression':'03dbb7a8e60b0316e4c06508bcb76175097f594a29d0a25d40a2129998ae4c8a'}

def read(path):return json.loads(workspace_path(path,exists=True).read_text(encoding='utf-8'))
def relative(path):return path.relative_to(ROOT).as_posix()
def runtime_profile(backend):
    if backend=='native-gpu':return {'engine':'torch-native-fp32','torch':'2.7.1+cu126','cudnn_benchmark':False,'TF32':False}
    if backend=='onnxruntime-cpu':return {'engine':'onnxruntime-cpu','version':'1.31.1','providers':['CPUExecutionProvider'],'cpu_threads':2,'cv2_threads':1}
    raise ValueError('known_family_runtime_required')
def fresh_json(path,value):
    path=workspace_path(path)
    if path.exists():raise ValueError('immutable_family_artifact_already_exists')
    atomic_json(path,value)

def validation():
    _,rows,digest=load_development()
    if digest!=PIN['development']:raise ValueError('frozen_VAL_source_changed')
    return rows

def assert_validation_binding(result,rows):
    by={r['sample_id']:r for r in rows}
    expected=Counter((r['sample_id'],o['class'],*o['bbox_xywh_px']) for r in rows for o in r['objects'] if o['class']==r['evaluation_class'])
    actual=Counter((r['sample_id'],r['class'],*r['bbox_xywh_px']) for r in result['truth'])
    if actual!=expected:raise ValueError('complete_exact_VAL_ground_truth_required')
    for p in result['predictions']:
        if p['sample_id'] not in by or p['class']!=by[p['sample_id']]['evaluation_class']:raise ValueError('non_VAL_or_unknown_class_prediction')
        if type(p['confidence']) not in (int,float) or not math.isfinite(p['confidence']) or not .01<=p['confidence']<=1 or len(p['bbox_xywh_px'])!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in p['bbox_xywh_px']) or min(p['bbox_xywh_px'][2:])<=0:raise ValueError('finite_positive_VAL_prediction_required')
    if result.get('images')!=len(rows):raise ValueError('complete_VAL_image_count_required')

def prepare(backend='native-gpu'):
    if backend not in ('native-gpu','onnxruntime-cpu'):raise ValueError('fixed_supported_comparison_backend_required')
    rows=validation()
    for path,key in [(BASE,'baseline_manifest'),(BASE_CHECKPOINT,'baseline_checkpoint'),(FRESH,'fresh'),(REGRESSION,'regression')]:
        if sha256(path)!=PIN[key]:raise ValueError('frozen_artifact_or_benchmark_changed')
    # Benchmarks are opaque SHA only; their TEST labels/photos are not decoded.
    value={'format':'privacy-family-v8-predeclared-protocol-v1','source_sha256':sha256(Path(__file__)),
        'trainer_source_sha256':sha256(ROOT/'scripts/train_redaction_rescue_v8.py'),'trainer_plan_sha256':sha256(RESCUE/'plan.json'),
        'P2_parent_protocol_sha256':sha256(P2_PARENT),'P2_original_parent_plan_sha256':read(P2_PARENT).get('parent_plan_sha256'),
        'development_sha256':PIN['development'],'comparison_backend':backend,'GPU_evaluation_requires_single_root_slot':True,
        'baseline_manifest_sha256':PIN['baseline_manifest'],'baseline_checkpoint_sha256':PIN['baseline_checkpoint'],
        'VAL_images':508,'VAL_face':400,'VAL_plate':108,'families':['legacy_v4','rescue_v8'],
        'P2_incomplete':'diagnostic only, never eligible; original run/state/source preserved',
        'baseline_completeness':'Immutable model/checkpoint/DEV/runtime-bound face400 plus plate108, or explicitly evaluate full508. Never relabel mutable best400 as baseline.',
        'backend_fairness':'Native GPU baseline508 vs completed frozen native trainer selected508; CPU mode replays BOTH models508 with identical CPU ORT runtime.',
        'runtime_profile':runtime_profile(backend),
        'confidence_grid':[.01,.03,.05,.1,.2,.35,.5],'largeface_gate':'original sqrtGTarea>=64, fixed confidence.1, recall>=legacy baseline-.01',
        'family_winner':'complete runs only; bothRecall95-feasible→macroPrecision then minRecall; otherwise minRecall then macroPrecision; exact ties baseline first',
        'canonical_positive_cases':'Per class, original GT area>=4096 correctly matched by BOTH baseline and winner at frozen operating thresholds; source_id ascending first; VAL-only numerical branch coverage',
        'strict_before_TEST':{'raw_rtol':.001,'raw_atol':.0001,'CPU_ORT':'1.31.1','face_and_plate_positive':True,'both_baseline_and_winner':True,'no_tolerance_relaxation':True},
        'benchmarks':{'fresh':{'sha256':PIN['fresh'],'samples':200,'role':'new unseen source only; physical identity independence unverified'},
                      'regression':{'sha256':PIN['regression'],'samples':1400,'role':'previously examined regression only; no selection'}},
        'one_shot_attempt_receipt':True,'TEST_predictions_before_freeze':0,'user_recordings_used':False,
        'source_expires_at':min(r['expires_at'] for r in rows),'research_only':True,'production_approved':False}
    workspace_path(DEST).mkdir(exist_ok=True);fresh_json(DEST/'protocol.json',value);fresh_json(PUBLIC,value)
    print(json.dumps({'status':'family_protocol_prepared_no_inference','protocol_sha256':sha256(DEST/'protocol.json'),'backend':backend,'VAL_images':508,'TEST_predictions':0}))

def protocol():
    value=read(DEST/'protocol.json')
    if read(PUBLIC)!=value or value['source_sha256']!=sha256(Path(__file__)):raise ValueError('frozen_family_protocol_or_source_changed')
    if value['trainer_source_sha256']!=sha256(ROOT/'scripts/train_redaction_rescue_v8.py') or value['trainer_plan_sha256']!=sha256(RESCUE/'plan.json'):raise ValueError('frozen_trainer_changed')
    for path,key in [(BASE,'baseline_manifest'),(BASE_CHECKPOINT,'baseline_checkpoint'),(FRESH,'fresh'),(REGRESSION,'regression')]:
        if sha256(path)!=PIN[key]:raise ValueError('original_frozen_artifact_changed')
    if datetime.fromisoformat(value['source_expires_at'])<=datetime.now(timezone.utc):raise ValueError('VAL_source_expired')
    return value

def completed_rescue():
    status=read(RESCUE/'training-status.json');manifest=read(RESCUE/'model.json');freeze=read(RESCUE/'candidate-freeze.json')
    if status.get('status')!='completed' or status.get('epochs')!=12 or manifest.get('training_complete') is not True:raise ValueError('complete_rescue_training_required')
    if freeze['manifest_sha256']!=sha256(RESCUE/'model.json') or freeze['checkpoint_sha256']!=sha256(RESCUE/'best.pt'):raise ValueError('completed_rescue_artifact_changed')
    if sha256(RESCUE/'training-config.json')!=sha256(RESCUE/'plan.json'):raise ValueError('approved_training_configuration_required')
    return manifest

def payload(role,result,rows,backend):
    assert_validation_binding(result,rows)
    manifest=BASE if role=='baseline' else RESCUE/'model.json';checkpoint=BASE_CHECKPOINT if role=='baseline' else RESCUE/'best.pt'
    return {'format':'privacy-family-complete-VAL-v1','role':role,'backend':backend,'development_sha256':PIN['development'],
            'runtime_profile':runtime_profile(backend),
            'manifest_sha256':sha256(manifest),'checkpoint_sha256':sha256(checkpoint),'source_expires_at':min(r['expires_at'] for r in rows),
            'case_sample_ids':sorted(r['sample_id'] for r in rows),'result':result,'TEST_images':0}

def evaluate_VAL(role,authorized=False,component='all'):
    if authorized is not True:raise ValueError('explicit_posttraining_VAL_inference_authorization_required')
    value=protocol();completed_rescue();rows=validation()
    if role not in ('baseline','rescue_v8'):raise ValueError('known_family_required')
    if component not in ('all','face','license_plate') or component!='all' and role!='baseline':raise ValueError('baseline_components_only')
    if component!='all':rows=[r for r in rows if r['evaluation_class']==component]
    output=DEST/(role+'-VAL508.json' if component=='all' else 'baseline-'+component+'-VAL.json')
    if output.exists():raise ValueError('VAL_artifact_already_exists')
    source=BASE if role=='baseline' else RESCUE/'model.json'
    if value['comparison_backend']=='native-gpu' and role=='rescue_v8':
        import torch
        if torch.__version__!='2.7.1+cu126':raise ValueError('same_approved_native_runtime_required')
        result=read(RESCUE/'selected-validation-predictions.json')
        fresh_json(output,payload(role,result,rows,'native-gpu'))
        print(json.dumps({'status':'completed_native_VAL508_imported','role':role,'new_image_inferences':0,'artifact_sha256':sha256(output),'TEST_images':0}))
        return
    if value['comparison_backend']=='native-gpu':
        import torch
        from agent.redaction_tiled_network import build_redaction_tiled_network
        from scripts.train_redaction_rescue_v8 import evaluate,configure_GPU,budget
        if torch.__version__!='2.7.1+cu126':raise ValueError('same_approved_native_runtime_required')
        torch.set_num_threads(2);configure_GPU(torch);budget(torch)
        if sha256(BASE_CHECKPOINT)!=PIN['baseline_checkpoint']:raise ValueError('baseline_checkpoint_changed')
        model=build_redaction_tiled_network();model.load_state_dict(torch.load(BASE_CHECKPOINT,map_location='cpu',weights_only=True)['model'],strict=True);model.cuda().eval().requires_grad_(False)
        result=evaluate(model,rows)
    else:
        import cv2
        from agent.models import OnnxModel
        from agent.evaluate import detection_metrics
        cv2.setNumThreads(1);model=OnnxModel(source,cpu_threads=2,allow_spinning=False)
        try:
            if model.backend_version!='1.31.1' or model.net.session.get_providers()!=['CPUExecutionProvider']:raise ValueError('CPU_ORT1311_only')
            model.manifest=dict(model.manifest,confidence=.01);truth=[];predictions=[]
            for row in rows:
                found=model.predict(read_image(row));truth.extend(dict(o,sample_id=row['sample_id']) for o in row['objects'] if o['class']==row['evaluation_class'])
                predictions.extend(dict(p,sample_id=row['sample_id']) for p in found if p['class']==row['evaluation_class'])
            result={'truth':truth,'predictions':predictions,'metrics':detection_metrics(truth,predictions,['face','license_plate']),'images':len(rows),'TEST_images':0}
        finally:model.close()
    fresh_json(output,payload(role,result,rows,value['comparison_backend']))
    print(json.dumps({'status':'complete_VAL508_recorded','role':role,'backend':value['comparison_backend'],'artifact_sha256':sha256(output),'TEST_images':0}))

def compose_baseline(face,plate,rows):
    for item,label,count in [(face,'face',400),(plate,'license_plate',108)]:
        expected=[r for r in rows if r['evaluation_class']==label]
        if len(expected)!=count or item.get('role')!='baseline' or item.get('manifest_sha256')!=PIN['baseline_manifest'] or item.get('checkpoint_sha256')!=PIN['baseline_checkpoint'] or item.get('development_sha256')!=PIN['development']:raise ValueError('immutable_baseline_component_binding_required')
        if item.get('case_sample_ids')!=sorted(r['sample_id'] for r in expected):raise ValueError('complete_baseline_component_cases_required')
        assert_validation_binding(item['result'],expected)
    if face['backend']!=plate['backend'] or face.get('runtime_profile')!=plate.get('runtime_profile') or face.get('runtime_profile')!=runtime_profile(face['backend']):raise ValueError('component_backend_or_runtime_mismatch')
    return {'truth':face['result']['truth']+plate['result']['truth'],'predictions':face['result']['predictions']+plate['result']['predictions'],'images':508,'TEST_images':0}

def compose_action(face_path,plate_path):
    value=protocol();rows=validation();paths=[workspace_path(p,exists=True) for p in (face_path,plate_path)]
    if any(p.suffix.lower()!='.json' or not p.is_relative_to(ROOT/'data/training') for p in paths):raise ValueError('managed_training_JSON_components_required')
    face,plate=map(read,paths);combined=compose_baseline(face,plate,rows)
    if face['backend']!=value['comparison_backend']:raise ValueError('frozen_comparison_backend_required')
    output=payload('baseline',combined,rows,value['comparison_backend']);output['component_sha256']=[sha256(p) for p in paths]
    fresh_json(DEST/'baseline-VAL508.json',output)

def load_VAL(role,value,rows):
    path=DEST/(role+'-VAL508.json');item=read(path)
    manifest=BASE if role=='baseline' else RESCUE/'model.json';checkpoint=BASE_CHECKPOINT if role=='baseline' else RESCUE/'best.pt'
    if item['role']!=role or item['backend']!=value['comparison_backend'] or item['manifest_sha256']!=sha256(manifest) or item['checkpoint_sha256']!=sha256(checkpoint) or item['development_sha256']!=PIN['development']:raise ValueError('VAL_artifact_binding_or_backend_mismatch')
    if item.get('runtime_profile')!=value['runtime_profile']:raise ValueError('VAL_runtime_profile_mismatch')
    if item['case_sample_ids']!=sorted(r['sample_id'] for r in rows):raise ValueError('complete_unique_VAL508_cases_required')
    assert_validation_binding(item['result'],rows);return item['result'],sha256(path)

def child_model(origin,role,confidence):
    source=read(origin);parent=workspace_path(source['training_provenance'],exists=True);provenance=read(parent)
    if provenance.get('data_revoked') or provenance.get('needs_revalidation'):raise ValueError('revoked_parent_model')
    folder=workspace_path(DEST/'models'/role);folder.mkdir(parents=True,exist_ok=False);manifest=folder/'model.json'
    copied=copy.deepcopy(provenance);copied.update(models=[relative(manifest)],derivation_only=True,new_training_updates=0,parent_provenance_sha256=sha256(parent))
    for asset in copied['assets']:asset['files']=[];asset['label_refs']=[]
    fresh_json(folder/'provenance.json',copied)
    source.update(confidence=confidence,training_provenance=relative(folder/'provenance.json'))
    fresh_json(manifest,source)
    return {'manifest':relative(manifest),'manifest_sha256':sha256(manifest),'origin_manifest':relative(origin),'origin_manifest_sha256':sha256(origin),
            'origin_provenance':relative(parent),'origin_provenance_sha256':sha256(parent),'provenance_sha256':sha256(folder/'provenance.json'),
            'checkpoint':relative(BASE_CHECKPOINT if origin==BASE else RESCUE/'best.pt'),'checkpoint_sha256':sha256(BASE_CHECKPOINT if origin==BASE else RESCUE/'best.pt'),
            'weights_sha256':source['sha256'],'confidence':confidence,'dataset_release_sha256':source['dataset_release'].removeprefix('manifest-sha256:')}

def canonical_cases(rows,baseline,winner,baseline_conf,winner_conf):
    result=[]
    from agent.pipeline import intersection_over_union
    for label in ('face','license_plate'):
        by=[]
        for row in sorted(rows,key=lambda r:r['source_id']):
            if row['evaluation_class']!=label:continue
            gt=[o for o in row['objects'] if o['class']==label and o['bbox_xywh_px'][2]*o['bbox_xywh_px'][3]>=4096]
            if not gt:continue
            def correct(value,confidence):return any(p['sample_id']==row['sample_id'] and p['class']==label and p['confidence']>=confidence and any(intersection_over_union(p['bbox_xywh_px'],g['bbox_xywh_px'])>=.5 for g in gt) for p in value['predictions'])
            if correct(baseline,baseline_conf) and correct(winner,winner_conf):by.append(row);break
        if not by:raise ValueError('common_positive_VAL_parity_case_required')
        row=by[0];result.append({k:row[k] for k in ['path','sha256','source_id','expires_at']}|{'label':label})
    return result

def freeze_winner():
    value=protocol();completed_rescue();rows=validation()
    if (DEST/'winner-freeze.json').exists():raise ValueError('winner_already_frozen_no_reselection')
    baseline,base_sha=load_VAL('baseline',value,rows);rescue,rescue_sha=load_VAL('rescue_v8',value,rows)
    candidates=[{'name':'legacy_v4','training_complete':True,'validation':baseline},{'name':'rescue_v8','training_complete':True,'validation':rescue}]
    chosen=family_winner(candidates,large_recall(baseline));selected=baseline if chosen=='legacy_v4' else rescue;calibration=calibrate(selected);confidence=calibration['selected_confidence']
    baseline_conf=read(BASE)['confidence'];cases=canonical_cases(rows,baseline,selected,baseline_conf,confidence)
    base_child=child_model(BASE,'baseline',baseline_conf);winner_child=child_model(BASE if chosen=='legacy_v4' else RESCUE/'model.json','winner',confidence)
    fresh_json(DEST/'positive-VAL-cases.json',cases)
    frozen={'format':'privacy-family-v8-winner-freeze','protocol_sha256':sha256(DEST/'protocol.json'),'selected_family':chosen,'selected_confidence':confidence,
            'models':{'baseline':base_child,'winner':winner_child},'validation_predictions_sha256':{'baseline':base_sha,'rescue_v8':rescue_sha},
            'positive_VAL_cases_sha256':sha256(DEST/'positive-VAL-cases.json'),'operating_metrics':calibration['selected'],
            'largeface_baseline_fixed01':large_recall(baseline),'largeface_selected_fixed01':large_recall(selected),'selection_used_TEST':False,
            'source_expires_at':value['source_expires_at'],'P2_parent_protocol_sha256':value['P2_parent_protocol_sha256'],'P2_incomplete_excluded':True}
    fresh_json(DEST/'winner-freeze.json',frozen)
    public={k:v for k,v in frozen.items() if k!='models'};public['model_artifact_sha256']={role:{key:item[key] for key in ['manifest_sha256','checkpoint_sha256','weights_sha256','origin_manifest_sha256']} for role,item in frozen['models'].items()}
    fresh_json(ROOT/'docs/reports/PRIVACY-FAMILY-V8-WINNER-FREEZE.json',public)
    for role in ['fresh','regression']:fresh_json(DEST/(role+'-plan.json'),comparison_plan(value,frozen,role))
    print(json.dumps({'status':'winner_frozen_strict_canonical_still_required','family':chosen,'threshold':confidence,'TEST_predictions':0}))

def numeric_gates(frozen):
    from scripts.evaluate_public_refinement import strict_proof
    if sha256(DEST/'positive-VAL-cases.json')!=frozen['positive_VAL_cases_sha256']:raise ValueError('canonical_VAL_cases_changed')
    for role,digest in frozen['validation_predictions_sha256'].items():
        filename=('baseline' if role=='baseline' else 'rescue_v8')+'-VAL508.json'
        if sha256(DEST/filename)!=digest:raise ValueError('frozen_VAL_predictions_changed')
    for role,item in frozen['models'].items():
        manifest=workspace_path(item['manifest'],exists=True)
        if sha256(manifest)!=item['manifest_sha256'] or sha256(workspace_path(item['checkpoint'],exists=True))!=item['checkpoint_sha256'] or sha256(manifest.parent/'provenance.json')!=item['provenance_sha256']:raise ValueError('frozen_winner_artifact_changed')
        if sha256(workspace_path(item['origin_manifest'],exists=True))!=item['origin_manifest_sha256'] or sha256(workspace_path(item['origin_provenance'],exists=True))!=item['origin_provenance_sha256']:raise ValueError('original_parent_artifact_changed')
        for case in read(DEST/'positive-VAL-cases.json'):
            proofpath=manifest.parent/('verification-final.json' if case['label']=='face' else 'verification-plate.json')
            strict_proof(manifest,proofpath,image_sha=case['sha256'],checkpoint_sha=item['checkpoint_sha256'],positive=True)
            proof=read(proofpath)
            if proof.get('app_backend')!='onnxruntime-cpu' or proof.get('app_backend_version')!='1.31.1':raise ValueError('canonical_CPU_ORT1311_required')
            if any(proof.get('detection_postprocess_parity',{}).get(ref,{}).get('application_class_counts',{}).get(case['label'],0)<1 for ref in ['ort','torch']):raise ValueError('actual_sameclass_positive_postprocess_required')

def comparison_plan(value,frozen,role):
    baseline=frozen['models']['baseline'];winner=frozen['models']['winner'];benchmark=FRESH if role=='fresh' else REGRESSION
    return {'created_before_final_predictions':True,'models':{'baseline':baseline['manifest'],'candidate':winner['manifest']},
            'baseline_manifest_sha256':baseline['manifest_sha256'],'candidate_release_sha256':winner['dataset_release_sha256'],
            'benchmark_manifest':relative(benchmark),'benchmark_sha256':value['benchmarks'][role]['sha256'],'expected_samples':value['benchmarks'][role]['samples'],
            'role':value['benchmarks'][role]['role'],'family_freeze_sha256':sha256(DEST/'winner-freeze.json'),'test_used_for_fitting_or_selection':False,
            'raw_rtol':.001,'raw_atol':.0001,'physical_independence_verified':False,'project_acceptance_eligible':False,'production_approved':False}

def run_comparison(role,authorized=False):
    if authorized is not True:raise ValueError('explicit_final_TEST_comparison_authorization_required')
    value=protocol();frozen=read(DEST/'winner-freeze.json')
    if frozen['protocol_sha256']!=sha256(DEST/'protocol.json'):raise ValueError('winner_protocol_changed')
    numeric_gates(frozen)  # Runs before any TEST labels/pixels or attempt creation.
    if role not in ('fresh','regression'):raise ValueError('known_benchmark_role_required')
    planpath=DEST/(role+'-plan.json');plan=read(planpath);benchmark=workspace_path(plan['benchmark_manifest'],exists=True)
    if plan!=comparison_plan(value,frozen,role):raise ValueError('comparison_must_match_frozen_family_exactly')
    if plan['family_freeze_sha256']!=sha256(DEST/'winner-freeze.json') or sha256(benchmark)!=value['benchmarks'][role]['sha256']:raise ValueError('frozen_comparison_plan_changed')
    attempt=workspace_path(DEST/(role+'-attempt.json'))
    with attempt.open('x',encoding='utf-8') as stream:json.dump({'status':'started','plan_sha256':sha256(planpath),'selection_allowed':False},stream)
    try:
        from scripts.compare_detector_models import compare
        result=compare(planpath,DEST/(role+'-comparison.json'))
        if result['samples']!=plan['expected_samples']:raise ValueError('fixed_benchmark_sample_count_mismatch')
        atomic_json(attempt,{'status':'completed','plan_sha256':sha256(planpath),'selection_allowed':False,'samples':result['samples']})
    except BaseException as error:
        atomic_json(attempt,{'status':'failed','error_type':type(error).__name__,'selection_allowed':False,'automatic_retry_allowed':False});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','evaluate','compose','freeze','run']);p.add_argument('--backend',choices=['native-gpu','onnxruntime-cpu'],default='native-gpu');p.add_argument('--role',choices=['baseline','rescue_v8','fresh','regression']);p.add_argument('--component',choices=['all','face','license_plate'],default='all');p.add_argument('--face-component');p.add_argument('--plate-component');p.add_argument('--authorized',action='store_true');a=p.parse_args()
    try:
        if a.action=='prepare':prepare(a.backend)
        elif a.action=='evaluate':evaluate_VAL(a.role,a.authorized,a.component)
        elif a.action=='compose':compose_action(a.face_component,a.plate_component)
        elif a.action=='freeze':freeze_winner()
        else:run_comparison(a.role,a.authorized)
    except Exception as error:print(json.dumps({'status':'family_operation_stopped','error_type':type(error).__name__}),file=sys.stderr);raise SystemExit(1)
