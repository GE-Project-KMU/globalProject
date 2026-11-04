"""Independent complete-VAL family freeze and strict-gated one-shot benchmarks.

Preparation/selection are metadata-only. Pixel inference requires an explicit
operator flag after the Rescue run completes. No current TRAIN/TEST execution.
"""
from __future__ import annotations
import argparse,copy,json,sys,math,os
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from scripts.redaction_rescue_data import load_development,read_image
from scripts.train_redaction_rescue_v8 import calibrate,large_recall,grouped_match

DEST=ROOT/'data/training/privacy-family-v10-evaluation'
OLD_V9_DEST=ROOT/'data/training/privacy-family-v9-evaluation'
BASE=ROOT/'data/training/redaction-inference-options-v4/option-tiled/model.json'
BASE_CHECKPOINT=ROOT/'data/training/redaction-specialists-v4-efficient/best.pt'
RESCUE=ROOT/'data/training/redaction-rescue-v8'
FRESH=ROOT/'data/training/research-redaction-cycle2-v7/fresh-test.json'
REGRESSION=ROOT/'data/training/cycle2-evaluation/privacy-regression-manifest.json'
FINAL12=ROOT/'data/training/redaction-rescue-v8-final12-retry1'
OLD_DEST=ROOT/'data/training/privacy-family-v8-evaluation'
STOP=ROOT/'.cache/privacy-family-v8-operator-retry1/V9-revision-stop.json'
STOP_SHA='4d8e065302d758d790745721c6045b583bdb4dff8555762ae52696a3919e7a1b'
FAMILIES=('legacy_v4','rescue_v8','fixed_final12')
TRAINER_SUBSET={'train_face':3000,'validation_face':400,'validation_plate':108,'total':3508}
PARENTS={'docs/reports/PRIVACY-FAMILY-V9-PREDECLARATION.json':'2717f0a737aa73d72afca8cc567519d0630c3ae2ebe77b980d608a9018228117', 'scripts/evaluate_privacy_family.py': '95b5dc346e30f0ee8341afdcdc874754507ce4c937f55261bdcb71b15779ca10', 'scripts/compare_privacy_family_parallel.py': '3171d8208f47553b893358a78e5a63be17c8d203e547366dc5f0c2970cf65f31', 'docs/reports/PRIVACY-FAMILY-V8-PROTOCOL.json': '5777de565111135f3793d247a332d5bb4c5cae4635378f8cfacfcbbff3cb4f59', 'docs/reports/PRIVACY-FAMILY-V8-PARALLEL-EXECUTION-PROTOCOL.json': 'fbbcc98f6a1c0e8a21abe32387e991c44d2533e80747e33823bc5c9469906c35'}
PARENTS.update({'scripts/evaluate_privacy_family_v9.py': 'e0f567095af5efaa3f5edd2b9ea44af5a00ba435e5c3a14bc2a70294de1b2d0e', 'scripts/compare_privacy_family_v9_parallel.py': '434ce430577a145f35242fa71ed27bf9f5cb88c9a0311a8f7f124f255524cef5', 'docs/reports/PRIVACY-FAMILY-V9-PROTOCOL.json': 'af88de6497d426565635a2c16eb7779e0ec002112878429efbbd0ce28e7aa4be', 'docs/reports/PRIVACY-FAMILY-V9-PARALLEL-EXECUTION-PROTOCOL.json': '4f5ef0ef01512c4ceb680876f6b91c6c6adb40f35ca7b0ce6ff8c2e1b56eb546', 'docs/reports/PRIVACY-RESCUE-V8-FINAL12-EXPORT-PLAN.json': 'bca93d5d7163f4746c436ba2da4c277fe8b0356ac7cdfc8d0e4b91d857ad3daf', 'scripts/export_redaction_final12.py': 'e8de9f51024417d4b4b36a8791870e144786ce582fea9adca1a493b88107287a'})
PARENTS.update({'scripts/export_redaction_final12_retry1.py':'76513ae98a97dc9e6ae49d8054635f01c9683b3503435c6e54653e475acd612f','docs/reports/PRIVACY-RESCUE-V8-FINAL12-RETRY1-EXPORT-PLAN.json':'56d58768fd4814ce6e46f0b00e069230e4ad3d564a0ef93d0d602c498415c051'})
EXPORT_PLAN=ROOT/'docs/reports/PRIVACY-RESCUE-V8-FINAL12-RETRY1-EXPORT-PLAN.json'
PUBLIC=ROOT/'docs/reports/PRIVACY-FAMILY-V10-PROTOCOL.json'
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

STOP_V9=ROOT/'.cache/privacy-family-v9-operator/V10-revision-stop.json'
STOP_V9_SHA='dfd8b3769f36fe4ea31a53467c160c8078bca985ff14a64dc3a0c2b8992a415e'
FAILED_EXPORT=ROOT/'data/training/redaction-rescue-v8-final12/root-export-execution.json'
FAILED_EXPORT_SHA='279d0f60ee2e09b75c0cb669ded3bb9ebc25c136a8f0f72bc20a0cf483dc0f57'
PRESERVATION=ROOT/'data/training/redaction-rescue-v8-final12/failure-parent-preservation-audit.json'
PRESERVATION_SHA='7a4a7b84d6dafb8d11b775271031c14974c51fecf58caf4930fcfc41b6ca230c'

def revision_lineage(metadata_only=False):
    for path,pin in [(STOP_V9,STOP_V9_SHA),(FAILED_EXPORT,FAILED_EXPORT_SHA),(PRESERVATION,PRESERVATION_SHA)]:
        if sha256(path)!=pin:raise ValueError('V9_failed_export_revision_evidence_changed')
    stopped=read(STOP_V9);failed=read(FAILED_EXPORT);audit=read(PRESERVATION)
    if stopped.get('status')!='stopped_naturally_for_failed_ROOT_fixed12_export' or any(stopped.get(k)!=v for k,v in {'root_authorized':True,'exact_identity_absent':True,'phase_count':0,'owned_evaluation_children_created':0,'owned_CPU_export_operations':0,'actual_native_VAL508_evaluations':0,'winner_freezes':0,'actual_TEST_predictions':0}.items()):raise ValueError('V9_pre_VAL_TEST_stop_required')
    if failed.get('status')!='failed' or failed.get('sources_unchanged') is not True or failed.get('own_child_running_after_cleanup') is not False:raise ValueError('preserved_failed_export_source_and_child_exit_required')
    # Original failed wrapper flag remains false; the independent audit supplies missing evidence.
    if audit.get('failed_execution_sha256')!=FAILED_EXPORT_SHA or audit.get('all_original_inputs_unchanged') is not True or audit.get('original_input_artifacts_verified')!=9 or len(audit.get('input_sha256',{}))!=9:raise ValueError('independent_failed_export_parent_preservation_required')
    for name,pin in audit['input_sha256'].items():
        path=workspace_path(name.replace('\\','/'),exists=True)
        if metadata_only and path.suffix=='.pt':continue
        if sha256(path)!=pin:raise ValueError('original_failed_export_parent_changed')
    return {'V9_stop_sha256':STOP_V9_SHA,'failed_export_sha256':FAILED_EXPORT_SHA,'independent_parent_preservation_sha256':PRESERVATION_SHA,'old_failed_wrapper_parent_verification_was_not_reached':True}

def immutable_parents():
    for name,pin in PARENTS.items():
        if sha256(ROOT/name)!=pin:raise ValueError('original_V8_source_or_protocol_changed')

def before_selection():
    no_prior_TEST()
    for role in ('fresh','regression'):
        for name in (role+'-attempt.json',role+'-comparison.json','parallel-'+role+'-v10'):
            if os.path.lexists(DEST/name):raise ValueError('V10_TEST_started_selection_closed')

def no_prior_TEST():
    revision_lineage(metadata_only=True)
    for role in ('fresh','regression'):
        for name in (role+'-attempt.json',role+'-comparison.json','parallel-'+role+'-v9'):
            if os.path.lexists(OLD_V9_DEST/name):raise ValueError('prior_V9_TEST_attempt_precludes_V10')
    for role in ('fresh','regression'):
        for name in (role+'-attempt.json',role+'-comparison.json','parallel-'+role+'-v8'):
            if os.path.lexists(OLD_DEST/name):raise ValueError('prior_V8_TEST_attempt_precludes_V10')
    if sha256(STOP)!=STOP_SHA:raise ValueError('original_queue_stop_receipt_changed')
    stopped=read(STOP)
    required={'root_authorized':True,'exact_identity_absent':True,'own_children_before_stop':0,'phase_count':0,'actual_VAL508_evaluations':0,'winner_freezes':0,'actual_TEST_predictions':0,'models_weights_and_learning_sources_changed':False}
    if stopped.get('status')!='stopped_before_V8_evaluation_for_preTEST_V9_revision' or any(stopped.get(k)!=v for k,v in required.items()):raise ValueError('pre_V8_phase_stop_required')

def artifact(role):
    if role=='baseline':return BASE,BASE_CHECKPOINT
    if role=='rescue_v8':return RESCUE/'model.json',RESCUE/'best.pt'
    if role=='fixed_final12':return FINAL12/'model.json',FINAL12/'best.pt'
    raise ValueError('known_family_required')

def family_winner(candidates,reference_large):
    if {c['name'] for c in candidates}!=set(FAMILIES) or len(candidates)!=3 or not all(c.get('training_complete') is True for c in candidates):raise ValueError('three_complete_predeclared_families_required')
    eligible=[(c,calibrate(c['validation'])['selected']) for c in candidates if large_recall(c['validation'])>=reference_large-.01]
    feasible=[p for p in eligible if p[1]['both_recall95']]
    def key(pair):
        c,point=pair
        metrics=(point['macro_precision'],point['min_recall']) if feasible else (point['min_recall'],point['macro_precision'])
        return (*metrics,-FAMILIES.index(c['name']))
    if not eligible:raise ValueError('no_largeface_eligible_family')
    return max(feasible or eligible,key=key)[0]['name']

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
    if backend!='native-gpu':raise ValueError('same_native_GPU_VAL_profile_required')
    before_selection();immutable_parents()
    export=read(EXPORT_PLAN)
    if export.get('original_trainer_subset')!=TRAINER_SUBSET:raise ValueError('exact_original_trainer_3508_subset_required')
    if export.get('export_source_sha256')!=sha256(ROOT/'scripts/export_redaction_final12_retry1.py'):raise ValueError('fixed_exporter_policy_required')
    value={'original_trainer_subset':TRAINER_SUBSET,'revision_lineage':revision_lineage(metadata_only=True),'revision_reason':'Exact original trainer provenance predicate TRAIN face3000 plus ALL VAL508; packaging-only correction after preserved V9 export failure, same epoch12 EMA and all selection math', 'parent_V9_protocol_sha256':'af88de6497d426565635a2c16eb7779e0ec002112878429efbbd0ce28e7aa4be','parent_V9_parallel_protocol_sha256':'4f5ef0ef01512c4ceb680876f6b91c6c6adb40f35ca7b0ce6ff8c2e1b56eb546','export_plan_sha256':sha256(EXPORT_PLAN),'export_source_sha256':export['export_source_sha256'],'format':'privacy-family-v10-predeclared-protocol-v1','source_sha256':sha256(Path(__file__)),
        'predeclaration_sha256':'2717f0a737aa73d72afca8cc567519d0630c3ae2ebe77b980d608a9018228117','family_aliases':{'rescue_v8_AP_selected':'rescue_v8'},'parent_V8_protocol_sha256':'5777de565111135f3793d247a332d5bb4c5cae4635378f8cfacfcbbff3cb4f59',
        'parent_V8_source_sha256':'95b5dc346e30f0ee8341afdcdc874754507ce4c937f55261bdcb71b15779ca10',
        'trainer_source_sha256':'7bc192db04ff79ff0cad9c3b440aa8357c99f387eecf4a7093ce3f59d9475f65',
        'trainer_plan_sha256':'6bbd59191a2c352a6a2bf7dc0fbd15f25ced0f500be6010fa3011a4e0f94164d',
        'old_operator_stop_sha256':STOP_SHA,'development_sha256':PIN['development'],'comparison_backend':backend,
        'P2_parent_protocol_sha256':sha256(P2_PARENT),'runtime_profile':runtime_profile(backend),
        'baseline_manifest_sha256':PIN['baseline_manifest'],'baseline_checkpoint_sha256':PIN['baseline_checkpoint'],
        'VAL_images':508,'VAL_face':400,'VAL_plate':108,'families':list(FAMILIES),
        'candidate_choice_predeclaration_timing':'V9 epoch12 EMA choice fixed after six development epochs, before final weights and all V8/V9/V10 TEST inference; not before all VAL','revision_timing':'V10 packaging/provenance correction after original training completed and V9 export failed; same predeclared epoch12 EMA, no reselection or new training',
        'learning_unchanged':True,'extra_gradient_updates':0,'fixed_final12_epoch':12,'fixed_final12_state':'EMA',
        'backend_fairness':'Complete native FP32 baseline508, original AP-selected native508 metadata import, final12 EMA native508; same frozen evaluator and runtime',
        'confidence_grid':[.01,.03,.05,.1,.2,.35,.5],
        'largeface_gate':'sqrtGTarea>=64 fixed confidence.1; recall>=baseline-.01',
        'family_winner':'both-class Recall95 feasible: macroPrecision then minRecall; otherwise minRecall then macroPrecision; exact tie legacy_v4 then rescue_v8 then fixed_final12',
        'canonical_positive_cases':'Same sorted common correctly detected positive face and plate VAL cases, baseline/winner four strict CPU proofs',
        'strict_before_TEST':{'raw_rtol':.001,'raw_atol':.0001,'CPU_ORT':'1.31.1','no_tolerance_relaxation':True},
        'benchmarks':{'fresh':{'sha256':PIN['fresh'],'samples':200,'role':'new blind source; physical identity independence unverified'},'regression':{'sha256':PIN['regression'],'samples':1400,'role':'previously examined regression only'}},
        'one_shot_attempt_receipt':True,'TEST_predictions_before_freeze':0,'source_expires_at':'2026-11-10T16:55:19.285523+00:00',
        'research_only':True,'production_approved':False,'user_recordings_used':False}
    DEST.mkdir(parents=True,exist_ok=True);fresh_json(DEST/'protocol.json',value);fresh_json(PUBLIC,value)
    print(json.dumps({'status':'V10_family_protocol_prepared_no_inference','protocol_sha256':sha256(DEST/'protocol.json'),'VAL_inferences':0,'TEST_images':0,'GPU_used':False}))
    return value

def protocol(metadata_only=False):
    no_prior_TEST();immutable_parents()
    revision_lineage(metadata_only=metadata_only)
    value=read(DEST/'protocol.json')
    if read(PUBLIC)!=value or value['source_sha256']!=sha256(Path(__file__)):raise ValueError('frozen_family_protocol_or_source_changed')
    if value['trainer_source_sha256']!=sha256(ROOT/'scripts/train_redaction_rescue_v8.py') or value['trainer_plan_sha256']!=sha256(RESCUE/'plan.json'):raise ValueError('frozen_trainer_changed')
    for path,key in [(BASE,'baseline_manifest'),(BASE_CHECKPOINT,'baseline_checkpoint'),(FRESH,'fresh'),(REGRESSION,'regression')]:
        if metadata_only and key=='baseline_checkpoint':continue # No weights read during policy preparation.
        if sha256(path)!=PIN[key]:raise ValueError('original_frozen_artifact_changed')
    if datetime.fromisoformat(value['source_expires_at'])<=datetime.now(timezone.utc):raise ValueError('VAL_source_expired')
    if value['export_plan_sha256']!=sha256(EXPORT_PLAN) or value['export_source_sha256']!=sha256(ROOT/'scripts/export_redaction_final12_retry1.py'):raise ValueError('exporter_or_policy_changed')
    if value['families']!=list(FAMILIES) or value['comparison_backend']!='native-gpu' or value['old_operator_stop_sha256']!=STOP_SHA:raise ValueError('fixed_three_family_policy_required')
    return value

def completed_rescue():
    status=read(RESCUE/'training-status.json');manifest=read(RESCUE/'model.json');freeze=read(RESCUE/'candidate-freeze.json')
    if status.get('status')!='completed' or status.get('epochs')!=12 or manifest.get('training_complete') is not True:raise ValueError('complete_rescue_training_required')
    if freeze['manifest_sha256']!=sha256(RESCUE/'model.json') or freeze['checkpoint_sha256']!=sha256(RESCUE/'best.pt'):raise ValueError('completed_rescue_artifact_changed')
    if sha256(RESCUE/'training-config.json')!=sha256(RESCUE/'plan.json'):raise ValueError('approved_training_configuration_required')
    return manifest

def completed_final12():
    completed_rescue()
    manifest=read(FINAL12/'model.json');receipt=read(FINAL12/'export-receipt.json')
    planpath=ROOT/'docs/reports/PRIVACY-RESCUE-V8-FINAL12-RETRY1-EXPORT-PLAN.json';plan=read(planpath)
    required={'status':'completed','epoch':12,'EMA':True,'TEST_images':0,'training_complete':True}
    if any(receipt.get(k)!=v for k,v in required.items()):raise ValueError('complete_fixed_epoch12_EMA_export_required')
    for key,path in [('manifest_sha256',FINAL12/'model.json'),('checkpoint_sha256',FINAL12/'best.pt'),('onnx_sha256',FINAL12/'model.onnx')]:
        if receipt.get(key)!=sha256(path):raise ValueError('fixed12_export_binding_changed')
    if receipt.get('source_plan_sha256')!=sha256(RESCUE/'plan.json') or receipt.get('export_plan_sha256')!=sha256(planpath) or receipt.get('export_source_sha256')!=plan.get('export_source_sha256'):raise ValueError('frozen_final12_export_policy_required')
    if manifest.get('sha256')!=receipt['onnx_sha256'] or manifest.get('trained_checkpoint_sha256')!=receipt['checkpoint_sha256'] or manifest.get('training_complete') is not True:raise ValueError('fixed12_manifest_binding_required')
    freeze=read(FINAL12/'candidate-freeze.json')
    if receipt.get('candidate_freeze_sha256')!=sha256(FINAL12/'candidate-freeze.json'):raise ValueError('fixed12_candidate_freeze_changed')
    for key in ('manifest_sha256','checkpoint_sha256','onnx_sha256','export_plan_sha256','export_source_sha256'):
        if freeze.get(key)!=receipt.get(key):raise ValueError('fixed12_candidate_freeze_binding_required')
    if freeze.get('epoch')!=12 or freeze.get('fixed_final12_EMA') is not True or freeze.get('TEST_images')!=0 or freeze.get('training_complete') is not True or freeze.get('confidence')!=.01 or receipt.get('legacy_raw_prefix_matches_frozen_v4') is not True:raise ValueError('fixed12_frozen_epoch_legacy_contract_required')
    if receipt.get('original_trainer_subset')!=TRAINER_SUBSET or plan.get('original_trainer_subset')!=TRAINER_SUBSET or receipt.get('same_original_final12_EMA') is not True or receipt.get('parent_export_policy_sha256')!='bca93d5d7163f4746c436ba2da4c277fe8b0356ac7cdfc8d0e4b91d857ad3daf':raise ValueError('same_fixed12_exact_original_training_subset_required')
    pointer=plan['provenance_membership_audit'];audit_path=workspace_path(pointer['path'],exists=True)
    if sha256(audit_path)!=pointer['sha256'] or receipt.get('membership_audit_sha256')!=pointer['sha256']:raise ValueError('exact_subset_membership_audit_required')
    audit=read(audit_path)
    if audit.get('passed') is not True or audit.get('selected_training_and_validation_rows')!=3508 or audit.get('parent_assets')!=3508 or audit.get('unused_plate_TRAIN_rows_excluded')!=1993 or receipt.get('provenance_membership_sha256')!=audit.get('complete_identity_set_sha256') or audit.get('development_sha256')!=PIN['development']:raise ValueError('exact_original_training_provenance_membership_required')
    return manifest

def payload(role,result,rows,backend):
    assert_validation_binding(result,rows)
    manifest,checkpoint=artifact(role)
    return {'format':'privacy-family-complete-VAL-v1','role':role,'backend':backend,'development_sha256':PIN['development'],
            'runtime_profile':runtime_profile(backend),
            'manifest_sha256':sha256(manifest),'checkpoint_sha256':sha256(checkpoint),'source_expires_at':min(r['expires_at'] for r in rows),
            'case_sample_ids':sorted(r['sample_id'] for r in rows),'result':result,'TEST_images':0}

def evaluate_VAL(role,authorized=False,component='all'):
    if authorized is not True:raise ValueError('explicit_posttraining_VAL_inference_authorization_required')
    before_selection()
    value=protocol();completed_rescue();completed_final12();rows=validation()
    if role not in ('baseline','rescue_v8','fixed_final12'):raise ValueError('known_family_required')
    if component not in ('all','face','license_plate') or component!='all' and role!='baseline':raise ValueError('baseline_components_only')
    if component!='all':rows=[r for r in rows if r['evaluation_class']==component]
    output=DEST/(role+'-VAL508.json' if component=='all' else 'baseline-'+component+'-VAL.json')
    if output.exists():raise ValueError('VAL_artifact_already_exists')
    source,checkpoint=artifact(role)
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
        if role=='baseline':model=build_redaction_tiled_network()
        else:
            from agent.redaction_rescue_network import build_redaction_rescue_network
            model=build_redaction_rescue_network()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True);model.cuda().eval().requires_grad_(False)
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
    before_selection()
    value=protocol();rows=validation();paths=[workspace_path(p,exists=True) for p in (face_path,plate_path)]
    if any(p.suffix.lower()!='.json' or not p.is_relative_to(ROOT/'data/training') for p in paths):raise ValueError('managed_training_JSON_components_required')
    face,plate=map(read,paths);combined=compose_baseline(face,plate,rows)
    if face['backend']!=value['comparison_backend']:raise ValueError('frozen_comparison_backend_required')
    output=payload('baseline',combined,rows,value['comparison_backend']);output['component_sha256']=[sha256(p) for p in paths]
    fresh_json(DEST/'baseline-VAL508.json',output)

def load_VAL(role,value,rows):
    path=DEST/(role+'-VAL508.json');item=read(path)
    manifest,checkpoint=artifact(role)
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
            'checkpoint':relative(BASE_CHECKPOINT if origin==BASE else FINAL12/'best.pt' if origin==FINAL12/'model.json' else RESCUE/'best.pt'),'checkpoint_sha256':sha256(BASE_CHECKPOINT if origin==BASE else FINAL12/'best.pt' if origin==FINAL12/'model.json' else RESCUE/'best.pt'),
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

def candidate_artifacts():
    result={}
    for role in ('baseline','rescue_v8','fixed_final12'):
        manifest,checkpoint=artifact(role);info=read(manifest)
        provenance=workspace_path(info['training_provenance'],exists=True)
        result[role]={'manifest':relative(manifest),'manifest_sha256':sha256(manifest),'checkpoint':relative(checkpoint),'checkpoint_sha256':sha256(checkpoint),'weights_sha256':info['sha256'],'provenance':relative(provenance),'provenance_sha256':sha256(provenance)}
    return result

def freeze_winner():
    before_selection()
    value=protocol();completed_rescue();completed_final12();rows=validation()
    if (DEST/'winner-freeze.json').exists():raise ValueError('winner_already_frozen_no_reselection')
    baseline,base_sha=load_VAL('baseline',value,rows);rescue,rescue_sha=load_VAL('rescue_v8',value,rows);final12,final_sha=load_VAL('fixed_final12',value,rows)
    candidates=[{'name':'legacy_v4','training_complete':True,'validation':baseline},{'name':'rescue_v8','training_complete':True,'validation':rescue},{'name':'fixed_final12','training_complete':True,'validation':final12}]
    chosen=family_winner(candidates,large_recall(baseline));selected={'legacy_v4':baseline,'rescue_v8':rescue,'fixed_final12':final12}[chosen];calibration=calibrate(selected);confidence=calibration['selected_confidence']
    baseline_conf=read(BASE)['confidence'];cases=canonical_cases(rows,baseline,selected,baseline_conf,confidence)
    base_child=child_model(BASE,'baseline',baseline_conf);winner_child=child_model({'legacy_v4':BASE,'rescue_v8':RESCUE/'model.json','fixed_final12':FINAL12/'model.json'}[chosen],'winner',confidence)
    fresh_json(DEST/'positive-VAL-cases.json',cases)
    frozen={'format':'privacy-family-v10-winner-freeze','protocol_sha256':sha256(DEST/'protocol.json'),'selected_family':chosen,'selected_confidence':confidence,
            'candidate_artifacts':candidate_artifacts(),'revision_timing':value['revision_timing'],'models':{'baseline':base_child,'winner':winner_child},'validation_predictions_sha256':{'baseline':base_sha,'rescue_v8':rescue_sha,'fixed_final12':final_sha},
            'positive_VAL_cases_sha256':sha256(DEST/'positive-VAL-cases.json'),'operating_metrics':calibration['selected'],
            'largeface_baseline_fixed01':large_recall(baseline),'largeface_selected_fixed01':large_recall(selected),'selection_used_TEST':False,
            'source_expires_at':value['source_expires_at'],'P2_parent_protocol_sha256':value['P2_parent_protocol_sha256'],'P2_incomplete_excluded':True}
    fresh_json(DEST/'winner-freeze.json',frozen)
    public={k:v for k,v in frozen.items() if k not in {'models','candidate_artifacts'}};public['candidate_artifact_sha256']={role:{key:child[key] for key in ('manifest_sha256','checkpoint_sha256','weights_sha256','provenance_sha256')} for role,child in frozen['candidate_artifacts'].items()};public['model_artifact_sha256']={role:{key:item[key] for key in ['manifest_sha256','checkpoint_sha256','weights_sha256','origin_manifest_sha256']} for role,item in frozen['models'].items()}
    fresh_json(ROOT/'docs/reports/PRIVACY-FAMILY-V10-WINNER-FREEZE.json',public)
    for role in ['fresh','regression']:fresh_json(DEST/(role+'-plan.json'),comparison_plan(value,frozen,role))
    print(json.dumps({'status':'winner_frozen_strict_canonical_still_required','family':chosen,'threshold':confidence,'TEST_predictions':0}))

def numeric_gates(frozen):
    no_prior_TEST()
    if frozen.get('candidate_artifacts')!=candidate_artifacts():raise ValueError('frozen_family_candidate_changed')
    from scripts.evaluate_public_refinement import strict_proof
    if sha256(DEST/'positive-VAL-cases.json')!=frozen['positive_VAL_cases_sha256']:raise ValueError('canonical_VAL_cases_changed')
    for role,digest in frozen['validation_predictions_sha256'].items():
        filename=role+'-VAL508.json'
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
    with attempt.open('x',encoding='utf-8') as stream:
        json.dump({'status':'started','plan_sha256':sha256(planpath),'selection_allowed':False},stream);stream.flush();os.fsync(stream.fileno())
    try:
        from scripts.compare_detector_models import compare
        result=compare(planpath,DEST/(role+'-comparison.json'))
        if result['samples']!=plan['expected_samples']:raise ValueError('fixed_benchmark_sample_count_mismatch')
        atomic_json(attempt,{'status':'completed','plan_sha256':sha256(planpath),'selection_allowed':False,'samples':result['samples']})
    except BaseException as error:
        atomic_json(attempt,{'status':'failed','error_type':type(error).__name__,'selection_allowed':False,'automatic_retry_allowed':False});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','evaluate','compose','freeze','run']);p.add_argument('--backend',choices=['native-gpu','onnxruntime-cpu'],default='native-gpu');p.add_argument('--role',choices=['baseline','rescue_v8','fixed_final12','fresh','regression']);p.add_argument('--component',choices=['all','face','license_plate'],default='all');p.add_argument('--face-component');p.add_argument('--plate-component');p.add_argument('--authorized',action='store_true');a=p.parse_args()
    try:
        if a.action=='prepare':prepare(a.backend)
        elif a.action=='evaluate':evaluate_VAL(a.role,a.authorized,a.component)
        elif a.action=='compose':compose_action(a.face_component,a.plate_component)
        elif a.action=='freeze':freeze_winner()
        else:run_comparison(a.role,a.authorized)
    except Exception as error:print(json.dumps({'status':'family_operation_stopped','error_type':type(error).__name__}),file=sys.stderr);raise SystemExit(1)
