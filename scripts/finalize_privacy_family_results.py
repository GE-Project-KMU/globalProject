"""Validate completed one-shot controls and publish whitelist aggregates only.

No image/NPY opening, inference, training, threshold search or source mutation.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,ntpath,os,re,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256
from agent.pipeline import intersection_over_union
from scripts import evaluate_privacy_family as family
from scripts import compare_privacy_family_parallel as parallel
from scripts.improve_redaction_finalize import assert_public

CLASSES=['face','license_plate'];COUNTS={'fresh':200,'regression':1400}
OPERATOR=ROOT/'.cache/privacy-family-v8-operator-retry1'
OUTPUT=ROOT/'docs/reports/PRIVACY-FAMILY-V8-RESULTS.json'
OPERATOR_SHA='f5cd212a401816eb49de2a9e9e9b604cf4f29978aa9a69c094d956272e265e87'
RUNTIME={'Python':'3.11.9','NumPy':'1.26.4','OpenCV':'4.10.0','ORT':'1.31.1','canonical_common_CV_threads':2,'parallel_CV_threads':1,'thread_counts_are_frozen_source_settings':True}
ARCHITECTURES={'two YOLOX-Nano specialists global plus four tiles','frozen legacy YOLOX-Nano specialists plus four tile P2 rescue'}
PHASES=['baseline-VAL508','rescue-VAL508-import','winner-freeze','strict-baseline-face','strict-baseline-license_plate','strict-winner-face','strict-winner-license_plate','fresh','regression']

def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def same(a,b):return encoded(a)==encoded(b)
def demand(condition,reason):
 if not condition:raise ValueError(reason)
def lexical_control_path(value):
 """Reject outside spellings using strings only, before resolve/stat/link I/O."""
 demand(isinstance(value,(str,Path)),'plain_control_path_required')
 raw=os.fspath(value);demand(bool(raw) and '\x00' not in raw and '://' not in raw,'plain_control_path_required')
 root=os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(ROOT))))
 drive,_=ntpath.splitdrive(raw);root_drive,_=ntpath.splitdrive(root)
 if drive:demand(bool(root_drive) and drive.casefold()==root_drive.casefold() and ntpath.isabs(raw),'outside_control_path_lexically_refused')
 candidate=os.path.normcase(os.path.normpath(os.path.abspath(os.path.join(root,raw))))
 try:inside=os.path.commonpath([root,candidate])==root
 except ValueError:inside=False
 demand(inside,'outside_control_path_lexically_refused');return candidate
class Controls:
 """Bound reads and freeze all bytes read/hashed; never admit a pixel format."""
 def __init__(self):self.hashes={}
 def path(self,value):
  lexical=lexical_control_path(value)
  p=workspace_path(lexical,exists=True)
  demand(p.is_file() and not p.is_symlink() and not getattr(p,'is_junction',lambda:False)(),'plain_control_file_required')
  demand(p.suffix.lower() in ('.json','.py','.onnx','.pt','.pth','.safetensors'),'pixel_or_NPY_read_forbidden')
  return p
 def digest(self,value,expected=None):
  p=self.path(value);digest=sha256(p)
  if expected is not None:demand(digest==expected,'control_or_weight_SHA_mismatch')
  if p in self.hashes:demand(self.hashes[p]==digest,'control_changed_during_finalization')
  self.hashes[p]=digest;return digest
 def read(self,value):
  p=self.path(value);demand(p.suffix=='.json','only_control_JSON_may_be_decoded');before=self.digest(p)
  with p.open('rb') as f:raw=f.read(512_000_001)
  demand(len(raw)<=512_000_000,'control_JSON_size_bound_exceeded')
  demand(hashlib.sha256(raw).hexdigest()==before,'control_changed_during_read')
  def duplicate_safe(pairs):
   result={}
   for k,v in pairs:
    demand(k not in result,'duplicate_control_JSON_key');result[k]=v
   return result
  def finite_float(text):
   value=float(text);demand(math.isfinite(value),'nonfinite_JSON');return value
  return json.loads(raw,object_pairs_hook=duplicate_safe,parse_float=finite_float,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite_JSON')))
 def close(self):
  for path,digest in list(self.hashes.items()):self.digest(path,digest)

def finite(value):return type(value) in (int,float) and math.isfinite(value)
def indexed_scores(truth,predictions,confidences):
 """Canonical AP101 math with GT lookup indexed by sample; order/ties unchanged."""
 output={c:{'classes':{}} for c in confidences}
 for label in CLASSES:
  gt=[r for r in truth if r['class']==label];by=defaultdict(list)
  for i,g in enumerate(gt):by[g['sample_id']].append((i,g))
  found=sorted((p for p in predictions if p['class']==label),key=lambda p:p['confidence'],reverse=True)
  matched=set();hits=[]
  for p in found:
   overlap,index=max(((intersection_over_union(p['bbox_xywh_px'],g['bbox_xywh_px']),i) for i,g in by[p['sample_id']] if i not in matched),default=(0,None))
   hit=overlap>=.5
   if hit:matched.add(index)
   hits.append(int(hit))
  for confidence in confidences:
   selected=[hit for p,hit in zip(found,hits) if p['confidence']>=confidence]
   tp=np.cumsum(selected);fp=np.cumsum(1-np.asarray(selected));recall=tp/len(gt) if gt else np.zeros(len(selected));precision=tp/np.maximum(tp+fp,1)
   ap=float(np.mean([max(precision[recall>=r],default=0.) for r in np.linspace(0,1,101)])) if gt else None
   count=sum(selected);n=len(selected)
   output[confidence]['classes'][label]={'gt_count':len(gt),'prediction_count':n,'tp':count,'fp':n-count,'fn':len(gt)-count,'ap50':ap,'recall':count/len(gt) if gt else None,'precision':count/n if n else 0.}
 for confidence,value in output.items():
  aps=[r['ap50'] for r in value['classes'].values() if r['ap50'] is not None]
  value.update(map50=sum(aps)/len(aps) if aps else None,matching='one_to_one_IoU0.5_AP101',confidence=confidence)
 return output

def records_valid(truth,predictions,order):
 allowed=set(order)
 for rows,is_prediction in [(truth,False),(predictions,True)]:
  for row in rows:
   demand(row.get('sample_id') in allowed and row.get('class') in CLASSES,'unbound_record_or_unknown_class')
   box=row.get('bbox_xywh_px');demand(isinstance(box,list) and len(box)==4 and all(finite(v) for v in box) and min(box[2:])>0,'finite_positive_bbox_required')
   if is_prediction:demand(finite(row.get('confidence')) and .01<=row['confidence']<=1,'frozen_score_floor_required')
def counts_valid(value):
 for k in ['gt_count','prediction_count','tp','fp','fn']:demand(type(value.get(k)) is int and value[k]>=0,'strict_integer_counts_required')
 demand(value['gt_count']>0 and value['tp']+value['fn']==value['gt_count'] and value['tp']+value['fp']==value['prediction_count'],'balanced_positive_support_required')
 for k in ['ap50','recall','precision']:demand(finite(value.get(k)) and 0<=value[k]<=1,'finite_unit_metric_required')
 demand(value['recall']==value['tp']/value['gt_count'] and value['precision']==(value['tp']/value['prediction_count'] if value['prediction_count'] else 0.),'count_metric_disagreement')
def validate_queue(queue,phases):
 demand(queue.get('status')=='completed' and queue.get('operator_source_sha256')==OPERATOR_SHA and queue.get('CPU_execution_runtime')==RUNTIME,'completed_approved_operator_required')
 demand(queue.get('automatic_retry_allowed') is False and queue.get('user_recordings_used') is False,'no_retry_or_user_recordings_required')
 demand(set(phases)==set(PHASES),'all_operator_phases_required')
 for name,phase in phases.items():
  demand(phase.get('phase')==name and phase.get('status')=='completed' and type(phase.get('child_exit_code')) is int and phase['child_exit_code']==0 and phase.get('own_child_running_after_cleanup') is False,'active_phase_failure_or_unclean_child')
  demand(phase.get('CPU_execution_runtime')==RUNTIME and phase.get('automatic_retry_allowed') is False and phase.get('root_authorized') is True,'approved_same_CPU_runtime_required')
  if name=='baseline-VAL508':
   demand(phase.get('observer_child_stopped') is True and phase.get('observer_thread_stopped') is True and phase.get('device_memory',{}).get('over_budget_observed') is False,'clean_GPU_observer_required')

def validate_role(role,plan,attempt,execution,receipt,workers,per_image,report,plan_sha,execution_sha):
 n=COUNTS[role]
 demand(attempt.get('status')=='completed' and attempt.get('writer')=='two_process_CPU' and attempt.get('samples')==n and attempt.get('selection_allowed') is False,'completed_permanent_attempt_required')
 demand(attempt.get('plan_sha256')==plan_sha and attempt.get('parallel_protocol_sha256')==execution['parallel_protocol_sha256'],'attempt_control_binding_failed')
 demand(execution.get('comparison_role')==role and execution.get('parent_plan_sha256')==plan_sha and execution.get('expected_samples')==n,'complete_frozen_execution_plan_required')
 demand(all(k in execution and same(execution[k],v) for k,v in plan.items()),'execution_parent_plan_fields_changed')
 demand(receipt.get('status')=='completed' and receipt.get('parent_plan_sha256')==plan_sha and receipt.get('execution_plan_sha256')==execution_sha and receipt.get('parallel_protocol_sha256')==execution['parallel_protocol_sha256'] and receipt.get('model_processes')==2 and receipt.get('GPU_operations')==0 and receipt.get('model_or_threshold_selection_allowed') is False and receipt.get('timing_is_shared_parallel_context') is True,'complete_two_CPU_worker_receipt_required')
 demand(finite(receipt.get('wall_seconds')) and receipt['wall_seconds']>=0,'finite_execution_walltime_required')
 demand(set(workers)=={'baseline','candidate'} and set(report.get('results',{}))==set(workers),'exact_pair_required')
 order=workers['baseline']['sample_order'];demand(len(order)==n and len(set(order))==n and order==sorted(order),'complete_unique_sorted_cohort_required')
 truth=per_image['truth'];records_valid(truth,[],order)
 demand(set(per_image.get('predictions',{}))==set(workers),'same_private_per_image_pair_required')
 metadata=execution['execution_models']
 for name,w in workers.items():
  parallel.validate_worker_output(w,name,execution,execution_sha,order)
  demand(all(finite(t) and t>=0 for t in w['timings']),'finite_numeric_worker_times_required')
  demand(same(w['predictions'],per_image['predictions'][name]),'worker_and_per_image_prediction_order_disagreement')
  records_valid(truth,w['predictions'],order);conf=metadata[name]['operating_confidence'];demand(finite(conf) and .01<=conf<=1,'frozen_confidence_required')
  recomputed=indexed_scores(truth,w['predictions'],[.01,conf,.35]);model=report['results'][name]
  for key,confidence in [('ap_metrics',.01),('operating_metrics',conf),('fixed_035_reference',.35)]:
   for label in CLASSES:counts_valid(model[key]['classes'][label])
   demand(same(model[key],recomputed[confidence]),'canonical_AP_or_counts_or_order_disagreement')
  demand(all(same(model[k],v) for k,v in metadata[name].items()),'comparison_model_metadata_disagreement')
  demand(model['latency_seconds']=={'median':float(np.median(w['timings'])),'p95':float(np.percentile(w['timings'],95))},'comparison_latency_disagreement')
 demand(report.get('samples')==n and report.get('plan_sha256')==plan_sha and report.get('benchmark_sha256')==plan['benchmark_sha256'] and report.get('scope')==plan['role'],'comparison_plan_binding_failed')
 demand(report.get('test_used_for_selection') is False and report.get('known_class_conditional') is True and report.get('independent_physical_identity_or_route_verified') is False and report.get('project_acceptance_passed') is False,'conditional_research_disclosures_required')
 demand(report.get('map50_change')==report['results']['candidate']['ap_metrics']['map50']-report['results']['baseline']['ap_metrics']['map50'],'comparison_change_disagreement')
 # Copy ONLY original completed aggregate values. Recomputed values are validation only.
 clean={}
 for name,item in report['results'].items():
  clean[name]={'operating_confidence':item['operating_confidence'],'map50':item['ap_metrics']['map50'],'latency_seconds':copy.deepcopy(item['latency_seconds']),
   'classes':{label:{'GT_support':item['ap_metrics']['classes'][label]['gt_count'],'AP50':item['ap_metrics']['classes'][label]['ap50'],
    **{key:copy.deepcopy(item['operating_metrics']['classes'][label][key]) for key in ['tp','fp','fn','prediction_count','precision','recall']}} for label in CLASSES}}
 minimum=min(v['recall'] for v in clean['candidate']['classes'].values())
 return {'images':n,'models':clean,'candidate_min_class_recall':minimum,'candidate_both_class_recall95_goal_met':minimum>=.95,'parallel_worker_wall_seconds':receipt['wall_seconds']}

def gather():
 controls=Controls();parallel.check_protocol();family.protocol();family.completed_rescue()
 queue=controls.read(OPERATOR/'queue-receipt.json');phases={name:controls.read(OPERATOR/(name+'-receipt.json')) for name in PHASES};validate_queue(queue,phases)
 controls.digest(ROOT/'.cache/privacy_family_v8_retry1_authorized_queue.py',OPERATOR_SHA)
 # Only active receipts count as failures. The explicitly superseded failed
 # preflight remains preserved and bound inside the successful retry controls.
 for phase in phases.values():
  for path,digest in phase['binding']['source_sha256'].items():controls.digest(ROOT/path,digest)
  for path,digest in phase['binding']['artifacts_sha256'].items():controls.digest(ROOT/path,digest)
  for path,digest in phase['output_sha256'].items():controls.digest(ROOT/path,digest)
 frozen=controls.read(family.DEST/'winner-freeze.json');family.numeric_gates(frozen)
 demand(frozen.get('selected_family') in ('legacy_v4','rescue_v8') and frozen.get('selection_used_TEST') is False and frozen.get('P2_incomplete_excluded') is True,'frozen_VAL_only_family_required')
 demand(frozen.get('protocol_sha256')==controls.digest(family.DEST/'protocol.json',parallel.FAMILY_SHA),'family_protocol_changed')
 value=family.protocol();val={}
 controls.digest(family.PUBLIC);controls.digest(family.DEST/'positive-VAL-cases.json',frozen['positive_VAL_cases_sha256'])
 for role in ['baseline','rescue_v8']:
  path=family.DEST/(role+'-VAL508.json');controls.digest(path,frozen['validation_predictions_sha256'][role]);item=controls.read(path)
  demand(item.get('role')==role and item.get('backend')=='native-gpu' and item.get('development_sha256')==family.PIN['development'] and item.get('runtime_profile')==value['runtime_profile'] and item.get('result',{}).get('images')==508 and item.get('TEST_images')==0,'complete_native_VAL_control_required')
  demand(len(item['case_sample_ids'])==508 and item['case_sample_ids']==sorted(set(item['case_sample_ids'])),'complete_unique_VAL_controls_required');val[role]=item
 demand(val['baseline']['case_sample_ids']==val['rescue_v8']['case_sample_ids'] and same(val['baseline']['result']['truth'],val['rescue_v8']['result']['truth']),'same_VAL_GT_required')
 selected=family.family_winner([{'name':'legacy_v4','training_complete':True,'validation':val['baseline']['result']},{'name':'rescue_v8','training_complete':True,'validation':val['rescue_v8']['result']}],family.large_recall(val['baseline']['result']))
 calibration=family.calibrate(val['baseline' if selected=='legacy_v4' else 'rescue_v8']['result'])
 demand(selected==frozen['selected_family'] and calibration['selected_confidence']==frozen['selected_confidence'] and same(calibration['selected'],frozen['operating_metrics']),'VAL_family_or_confidence_selection_changed')
 artifacts={};canonical={}
 for role,item in frozen['models'].items():
  manifest=controls.read(ROOT/item['manifest']);demand(manifest['confidence']==item['confidence'] and manifest.get('architecture') in ARCHITECTURES,'frozen_model_contract_required')
  controls.digest(ROOT/item['manifest'],item['manifest_sha256']);controls.digest(ROOT/item['checkpoint'],item['checkpoint_sha256']);controls.digest(ROOT/manifest['weights'],item['weights_sha256'])
  controls.digest(ROOT/item['origin_manifest'],item['origin_manifest_sha256']);controls.digest(ROOT/item['origin_provenance'],item['origin_provenance_sha256'])
  artifacts[role]={k:item[k] for k in ['manifest_sha256','weights_sha256','checkpoint_sha256','origin_manifest_sha256','provenance_sha256','dataset_release_sha256']}
  canonical[role]={}
  for label,filename in [('face','verification-final.json'),('license_plate','verification-plate.json')]:
   proofpath=(ROOT/item['manifest']).parent/filename;proof=controls.read(proofpath)
   demand(proof.get('runtime_python')=='3.11.9' and proof.get('opencv_preprocessing_version')=='4.10.0','same_direct_CPU_canonical_runtime_required')
   canonical[role][label]=controls.digest(proofpath)
 cohorts={};worker_hashes={}
 for role in COUNTS:
  parent=family.DEST/(role+'-plan.json');plan=controls.read(parent);demand(plan==family.comparison_plan(value,frozen,role),'exact_original_benchmark_plan_required')
  directory=family.DEST/('parallel-'+role+'-v8');ep=directory/'execution-plan.json';execution=controls.read(ep);parallel.assert_closure(execution)
  demand(execution['parent_pid']==phases[role]['child_pid'] and execution['parent_plan']==parent.relative_to(ROOT).as_posix(),'direct_operator_execution_parent_binding_failed')
  started={'status':'started','plan_sha256':controls.digest(parent),'parallel_protocol_sha256':controls.digest(parallel.PROTOCOL),'writer':'two_process_CPU','selection_allowed':False,'automatic_retry_allowed':False}
  demand(execution['attempt_sha256']==hashlib.sha256(json.dumps(started,sort_keys=True).encode()).hexdigest(),'original_exclusive_attempt_snapshot_changed')
  for refs in execution['model_closure'].values():
   for ref in refs.values():controls.digest(ROOT/ref['path'],ref['sha256'])
  controls.digest(ROOT/execution['benchmark_manifest'],execution['benchmark_sha256'])
  workers={name:controls.read(directory/(name+'.json')) for name in ['baseline','candidate']};reportpath=family.DEST/(role+'-comparison.json');report=controls.read(reportpath)
  controls.digest(reportpath,queue['results_sha256'][role]);worker_hashes[role]={name:controls.digest(directory/(name+'.json')) for name in workers}
  cohorts[role]=validate_role(role,plan,controls.read(family.DEST/(role+'-attempt.json')),execution,controls.read(directory/'execution-receipt.json'),workers,controls.read(family.DEST/(role+'-comparison-per-image.json')),report,controls.digest(parent),controls.digest(ep))
 result={'format':'privacy-family-v8-final-results-v1','selected_family':frozen['selected_family'],'selected_confidence':frozen['selected_confidence'],
  'family_protocol_sha256':parallel.FAMILY_SHA,'family_source_sha256':parallel.FAMILY_SOURCE_SHA,'parallel_protocol_sha256':controls.digest(parallel.PROTOCOL),'parallel_worker_source_sha256':controls.digest(ROOT/'scripts/compare_privacy_family_parallel.py'),
  'winner_freeze_sha256':controls.digest(family.DEST/'winner-freeze.json'),'publisher_source_sha256':sha256(Path(__file__)),'artifact_sha256':artifacts,'canonical_default_strict_evidence_sha256':canonical,
  'worker_result_sha256':worker_hashes,'operator_control_sha256':controls.digest(OPERATOR/'queue-receipt.json'),'runtime':copy.deepcopy(RUNTIME),'cohorts':cohorts,
  'research_only':True,'production_approved':False,'human_privacy_reviewed':False,'physical_person_or_route_independence_verified':False,'TEST_used_for_selection':False,
  'limits':['Fresh200 is a once-only paired benchmark; old1400 is previously seen regression, never TRAIN or selection.',
   'Publisher-known class coverage only; unknown other class is not a negative. Full privacy protection is unverified.',
   'Model family and thresholds were frozen on VAL508 before TEST; incomplete P2 is diagnostic only.',
   'All four face/plate canonical proofs use unchanged default raw/postprocess gates; numerical agreement is not accuracy.',
   'Two concurrent CPU2/inter1 workers use CV1; unchanged canonical verifier uses CV2 in the same Python/NumPy/OpenCV/ORT environment.',
   'Shared-host per-model and worker wall latency is not isolated speed, whole-pipeline10FPS, camera or field acceptance.',
   'Physical person, vehicle and route independence remains unverified; source grouping is a proxy.',
   'Original expiry and non-commercial research/redistribution limits remain; no raw photos or learned weights are published.',
   'Earlier failed preflight remains historical evidence; every active retry operator phase and permanent attempt must have completed.']}
 controls.close();assert_public(result);return result,controls

def markdown(report):
 lines=['# 개인정보 모델 가족 최종 CPU 평가','','VAL508에서 모델 가족과 임계값을 먼저 동결했다. 아래 값은 완료된 비교 보고서의 집계를 그대로 옮긴 것이다.','','선택 가족: `'+report['selected_family']+'`, confidence '+str(report['selected_confidence'])+'.','','|범위|클래스|모델|GT|AP50|TP|FP|FN|정밀도|재현율|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for role in COUNTS:
  for name in ['baseline','candidate']:
   for label in CLASSES:
    v=report['cohorts'][role]['models'][name]['classes'][label]
    lines.append(f"|{role}|{label}|{name}|{v['GT_support']}|{v['AP50']*100:.4f}%|{v['tp']}|{v['fp']}|{v['fn']}|{v['precision']*100:.4f}%|{v['recall']*100:.4f}%|")
 for role in COUNTS:
  v=report['cohorts'][role];lines.extend(['',f"{role}: {v['images']}장, 후보 최소 클래스 Recall {v['candidate_min_class_recall']*100:.4f}%, 양 클래스 Recall95% 목표 {'충족' if v['candidate_both_class_recall95_goal_met'] else '미달'}. 시간이 공유 호스트 경합을 포함하므로10FPS 근거로 사용하지 않는다."])
 lines.extend(['','## 해석 범위','',*['- '+v for v in report['limits']],'']);return '\n'.join(lines).encode('utf-8')
def public_whitelist(report):
 allowed={'format','selected_family','selected_confidence','family_protocol_sha256','family_source_sha256','parallel_protocol_sha256','parallel_worker_source_sha256','winner_freeze_sha256','publisher_source_sha256','artifact_sha256','canonical_default_strict_evidence_sha256','worker_result_sha256','operator_control_sha256','runtime','cohorts','research_only','production_approved','human_privacy_reviewed','physical_person_or_route_independence_verified','TEST_used_for_selection','limits'}
 demand(set(report)==allowed,'complete_public_root_whitelist_required')
 demand(report['format']=='privacy-family-v8-final-results-v1' and report['selected_family'] in ('legacy_v4','rescue_v8') and finite(report['selected_confidence']) and .01<=report['selected_confidence']<=1,'public_family_contract_required')
 demand(set(report['cohorts'])==set(COUNTS),'exact_public_cohorts_required')
 for role,cohort in report['cohorts'].items():
  demand(set(cohort)=={'images','models','candidate_min_class_recall','candidate_both_class_recall95_goal_met','parallel_worker_wall_seconds'} and type(cohort['images']) is int and cohort['images']==COUNTS[role],'public_cohort_whitelist_required')
  demand(set(cohort['models'])=={'baseline','candidate'},'public_model_pair_required')
  for model in cohort['models'].values():
   demand(set(model)=={'operating_confidence','map50','latency_seconds','classes'} and set(model['classes'])==set(CLASSES) and set(model['latency_seconds'])=={'median','p95'},'public_model_whitelist_required')
   demand(finite(model['operating_confidence']) and .01<=model['operating_confidence']<=1 and all(finite(v) and v>=0 for v in model['latency_seconds'].values()),'public_numeric_runtime_required')
   for values in model['classes'].values():
    demand(set(values)=={'GT_support','AP50','tp','fp','fn','prediction_count','precision','recall'},'public_class_whitelist_required')
    counts_valid({'gt_count':values['GT_support'],'ap50':values['AP50'],**{k:values[k] for k in ['tp','fp','fn','prediction_count','precision','recall']}})
   demand(model['map50']==sum(v['AP50'] for v in model['classes'].values())/2,'public_macro_AP_disagreement')
  minimum=min(v['recall'] for v in cohort['models']['candidate']['classes'].values())
  demand(cohort['candidate_min_class_recall']==minimum and type(cohort['candidate_both_class_recall95_goal_met']) is bool and cohort['candidate_both_class_recall95_goal_met']==(minimum>=.95) and finite(cohort['parallel_worker_wall_seconds']) and cohort['parallel_worker_wall_seconds']>=0,'public_goal_or_timing_disagreement')
 for key,value in report.items():
  if key.endswith('_sha256') and isinstance(value,str):demand(bool(re.fullmatch('[a-f0-9]{64}',value)),'aggregate_SHA_required')
 def hashes(value):
  if isinstance(value,dict):
   for item in value.values():hashes(item)
  else:demand(isinstance(value,str) and bool(re.fullmatch('[a-f0-9]{64}',value)),'aggregate_SHA_only')
 if 'artifact_sha256' in report:
  demand(set(report['artifact_sha256'])=={'baseline','winner'},'artifact_role_whitelist_required')
  for value in report['artifact_sha256'].values():demand(set(value)=={'manifest_sha256','weights_sha256','checkpoint_sha256','origin_manifest_sha256','provenance_sha256','dataset_release_sha256'},'artifact_hash_whitelist_required')
  hashes(report['artifact_sha256'])
 if 'canonical_default_strict_evidence_sha256' in report:
  value=report['canonical_default_strict_evidence_sha256'];demand(set(value)=={'baseline','winner'} and all(set(v)==set(CLASSES) for v in value.values()),'canonical_hash_whitelist_required');hashes(value)
 if 'worker_result_sha256' in report:
  value=report['worker_result_sha256'];demand(set(value)==set(COUNTS) and all(set(v)=={'baseline','candidate'} for v in value.values()),'worker_hash_whitelist_required');hashes(value)
 if 'runtime' in report:demand(report['runtime']==RUNTIME,'public_runtime_whitelist_required')
 for key,wanted in [('research_only',True),('production_approved',False),('human_privacy_reviewed',False),('physical_person_or_route_independence_verified',False),('TEST_used_for_selection',False)]:demand(report[key] is wanted,'public_research_scope_required')
 for value in walk_strings(report):
  demand(not re.search(r'(?i)\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b',value),'individual_UUID_in_public_report')
 assert_public(report)
def walk_strings(value):
 if isinstance(value,str):yield value
 elif isinstance(value,dict):
  for k,v in value.items():yield str(k);yield from walk_strings(v)
 elif isinstance(value,list):
  for v in value:yield from walk_strings(v)
def publish_pair(report,destination):
 public_whitelist(report);j=encoded(report);m=markdown(report);destination=workspace_path(lexical_control_path(destination));other=destination.with_suffix('.md')
 demand(destination.suffix=='.json' and destination.parent==ROOT/'docs/reports','public_report_directory_required')
 if destination.exists() or other.exists():
  demand(destination.is_file() and other.is_file() and destination.read_bytes()==j and other.read_bytes()==m,'existing_public_results_differ_or_partial');return {'status':'unchanged','json_sha256':hashlib.sha256(j).hexdigest(),'MD_sha256':hashlib.sha256(m).hexdigest()}
 # Publish JSON last: an MD write failure can never leave a successful JSON.
 # A partial MD is preserved and any next invocation refuses to overwrite it.
 for path,raw in [(other,m),(destination,j)]:
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'status':'published','json_sha256':hashlib.sha256(j).hexdigest(),'MD_sha256':hashlib.sha256(m).hexdigest()}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--authorized',action='store_true');a=p.parse_args()
 try:
  demand(a.authorized,'explicit_postcompletion_publication_authorization_required')
  import cv2,onnxruntime
  demand({'Python':sys.version.split()[0],'NumPy':np.__version__,'OpenCV':cv2.__version__,'ORT':onnxruntime.__version__}=={k:RUNTIME[k] for k in ['Python','NumPy','OpenCV','ORT']},'same_direct_CPU_metadata_runtime_required')
  report,controls=gather();controls.close();print(json.dumps(publish_pair(report,OUTPUT)));return 0
 except Exception as error:print(json.dumps({'status':'privacy_publication_refused','error_type':type(error).__name__}));return 1
if __name__=='__main__':raise SystemExit(main())
