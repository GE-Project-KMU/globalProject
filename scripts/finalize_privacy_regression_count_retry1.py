"""Metadata-only publication of preserved fresh200 and corrected same-manifest reg600.
Original failed operator/1400-plan evidence remains failed and unchanged.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256
from scripts import finalize_privacy_family_v10_results as base
from scripts import compare_privacy_regression_count_retry1 as retry
from scripts.improve_redaction_finalize import assert_public
family=retry.family;parallel=retry.original
Controls=base.Controls;encoded=base.encoded;same=base.same;demand=base.demand;finite=base.finite
indexed_scores=base.indexed_scores;records_valid=base.records_valid;counts_valid=base.counts_valid;walk_strings=base.walk_strings;lexical_control_path=base.lexical_control_path
CLASSES=list(base.CLASSES);COUNTS={'fresh':200,'regression':600};RUNTIME=copy.deepcopy(base.RUNTIME);ARCHITECTURES=set(base.ARCHITECTURES);REVISION=copy.deepcopy(base.REVISION)
OPERATOR=base.OPERATOR;PHASES=list(base.PHASES);PARENT_PUBLISHER_SHA=base.PARENT_PUBLISHER_SHA;PREDECL=base.PREDECL;PREDECL_SHA=base.PREDECL_SHA
OUTPUT=ROOT/'docs/reports/PRIVACY-FAMILY-V10-COUNT-RETRY1-RESULTS.json'
BASE_SHA='6ea83550fe1abbcb518a9fd241257a0838d195f03afc7b2ed2512cc9224f2f0a'
RUNNER_SHA='a2c3a5e460a50a4ec30c902edcd4626802ae9f3d5061a334e5d6bf4376297bd0'
PROTOCOL_SHA='92fbd695334314db83e38416bf17bdd1777d70dafeca1490c272177b239b94a1'
COUNT_CORRECTION={'original_expected_images':1400,'actual_unchanged_manifest_images':600,'face_images':300,'plate_images':300,'benchmark_sha256':'03dbb7a8e60b0316e4c06508bcb76175097f594a29d0a25d40a2129998ae4c8a','original_plan_sha256':'38938b525f8525adf30897a9832e4492278f28a0f415de1de9207a0045d6986d','original_failed_attempt_sha256':'2b4d57ab2e6ab23954a5b11349938e52ea4bd24c8e484c2361d5ed0e0426d3d5','corrected_plan_sha256':'df803eef81daa7034d16b042ba7b268075dd5f919cd52b73f4b5a69c6f0feb88','new_protocol_sha256':PROTOCOL_SHA,'new_runner_source_sha256':RUNNER_SHA,'original_operator_status':'failed','original_regression_status':'failed','original_regression_workers':0,'original_regression_image_hash_or_decode_calls':0,'fresh_images_reexecuted':0,'model_or_threshold_changed':False,'changed_plan_fields':['expected_samples']}

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


def validate_failed_operator(queue,phases,operator_sha):
 demand(queue.get('status')=='failed' and queue.get('reason')=='own_child_nonzero_exit' and queue.get('operator_source_sha256')==operator_sha and queue.get('CPU_execution_runtime')==RUNTIME,'preserved_original_failed_operator_required')
 demand(queue.get('automatic_retry_allowed') is False and queue.get('user_recordings_used') is False and set(phases)==set(PHASES),'complete_original_controls_required')
 for name,phase in phases.items():
  expected='failed' if name=='regression' else 'completed'
  demand(phase.get('phase')==name and phase.get('status')==expected and type(phase.get('child_exit_code')) is int and (phase['child_exit_code']!=0 if name=='regression' else phase['child_exit_code']==0) and phase.get('own_child_running_after_cleanup') is False,'original_failure_or_completed_phase_mismatch')
  demand(phase.get('CPU_execution_runtime')==RUNTIME and phase.get('automatic_retry_allowed') is False and phase.get('root_authorized') is True,'unchanged_original_runtime_required')
  if name in ('baseline-VAL508','fixed12-VAL508'):demand(phase.get('observer_child_stopped') is True and phase.get('observer_thread_stopped') is True and phase.get('device_memory',{}).get('over_budget_observed') is False,'clean_native_GPU_release_required')

def count_lineage(protocol,old_plan,new_plan,failed,new_receipt,new_report):
 demand(protocol.get('old_expected_samples')==1400 and protocol.get('expected_samples')==600 and protocol.get('face_samples')==300 and protocol.get('plate_samples')==300 and protocol.get('benchmark_sha256')==COUNT_CORRECTION['benchmark_sha256'] and protocol.get('fresh_reexecuted') is False and protocol.get('changed_plan_fields')==['expected_samples'] and protocol.get('model_or_threshold_selection_allowed') is False,'truthful_count_only_correction_required')
 demand(old_plan.get('expected_samples')==1400 and same(new_plan,dict(old_plan,expected_samples=600)),'only_expected_count_may_change')
 demand(failed.get('status')=='failed' and failed.get('automatic_retry_allowed') is False,'original_failure_must_remain_failed')
 expected={'original_expected':1400,'actual_same_manifest':600,'original_plan_sha256':COUNT_CORRECTION['original_plan_sha256'],'original_failed_attempt_sha256':COUNT_CORRECTION['original_failed_attempt_sha256'],'protocol_sha256':PROTOCOL_SHA,'fresh_reexecuted':False,'cohort_or_model_selection_changed':False}
 demand(same(new_report.get('count_correction'),expected),'completed_report_count_disclosure_mismatch')
 demand(new_receipt.get('original_plan_sha256')==COUNT_CORRECTION['original_plan_sha256'] and new_receipt.get('original_failed_attempt_sha256')==COUNT_CORRECTION['original_failed_attempt_sha256'] and new_receipt.get('worker_source_sha256')==RUNNER_SHA,'new_execution_parent_failure_binding_required')

def gather(operator_sha):
 controls=Controls();controls.digest(ROOT/'scripts/finalize_privacy_family_v10_results.py',BASE_SHA);controls.digest(ROOT/'scripts/compare_privacy_regression_count_retry1.py',RUNNER_SHA);controls.digest(retry.PUBLIC,PROTOCOL_SHA)
 protocol=retry.check_protocol();_,value,frozen,new_parent,new_plan=retry.admit();family.completed_rescue();family.completed_final12()
 queue=controls.read(OPERATOR/'queue-receipt.json');phases={n:controls.read(OPERATOR/(n+'-receipt.json')) for n in PHASES};validate_failed_operator(queue,phases,operator_sha)
 controls.digest(ROOT/'.cache/privacy_family_v10_authorized_queue.py',operator_sha)
 for phase in phases.values():
  for path,digest in phase['binding']['source_sha256'].items():controls.digest(ROOT/path,digest)
  for path,digest in phase['binding']['artifacts_sha256'].items():controls.digest(ROOT/path,digest)
  for path,digest in phase.get('output_sha256',{}).items():controls.digest(ROOT/path,digest)
 controls.digest(retry.BOUNDARY,retry.BOUNDARY_SHA);controls.digest(family.DEST/'regression-plan.json',retry.ORIGINAL_PLAN_SHA);controls.digest(family.DEST/'regression-attempt.json',retry.FAILED_SHA)
 frozen=controls.read(family.DEST/'winner-freeze.json');family.numeric_gates(frozen)
 demand(frozen.get('selected_family') in ('legacy_v4','rescue_v8','fixed_final12') and frozen.get('selection_used_TEST') is False and frozen.get('P2_incomplete_excluded') is True,'frozen_VAL_only_family_required')
 demand(frozen.get('protocol_sha256')==controls.digest(family.DEST/'protocol.json',parallel.family_sha()),'family_protocol_changed')
 value=family.protocol();val={}
 controls.digest(family.PUBLIC);controls.digest(family.DEST/'positive-VAL-cases.json',frozen['positive_VAL_cases_sha256'])
 for role in ['baseline','rescue_v8','fixed_final12']:
  path=family.DEST/(role+'-VAL508.json');controls.digest(path,frozen['validation_predictions_sha256'][role]);item=controls.read(path)
  demand(item.get('role')==role and item.get('backend')=='native-gpu' and item.get('development_sha256')==family.PIN['development'] and item.get('runtime_profile')==value['runtime_profile'] and item.get('result',{}).get('images')==508 and item.get('TEST_images')==0,'complete_native_VAL_control_required')
  demand(len(item['case_sample_ids'])==508 and item['case_sample_ids']==sorted(set(item['case_sample_ids'])),'complete_unique_VAL_controls_required');val[role]=item
 demand(all(val['baseline']['case_sample_ids']==val[role]['case_sample_ids'] and same(val['baseline']['result']['truth'],val[role]['result']['truth']) for role in ['rescue_v8','fixed_final12']),'same_three_family_VAL_GT_required')
 selected=family.family_winner([{'name':'legacy_v4','training_complete':True,'validation':val['baseline']['result']},{'name':'rescue_v8','training_complete':True,'validation':val['rescue_v8']['result']},{'name':'fixed_final12','training_complete':True,'validation':val['fixed_final12']['result']}],family.large_recall(val['baseline']['result']))
 calibration=family.calibrate(val[{'legacy_v4':'baseline','rescue_v8':'rescue_v8','fixed_final12':'fixed_final12'}[selected]]['result'])
 demand(selected==frozen['selected_family'] and calibration['selected_confidence']==frozen['selected_confidence'] and same(calibration['selected'],frozen['operating_metrics']),'VAL_family_or_confidence_selection_changed')
 demand(frozen.get('revision_timing')==value['revision_timing'],'unchanged_development_informed_revision_required')
 candidate_hashes={}
 demand(set(frozen.get('candidate_artifacts',{}))=={'baseline','rescue_v8','fixed_final12'},'all_three_candidate_artifacts_required')
 for role,item in frozen['candidate_artifacts'].items():
  manifest=controls.read(ROOT/item['manifest'])
  for key,reference in [('manifest_sha256',item['manifest']),('checkpoint_sha256',item['checkpoint']),('provenance_sha256',item['provenance']),('weights_sha256',manifest['weights'])]:controls.digest(ROOT/reference,item[key])
  demand(val[role]['manifest_sha256']==item['manifest_sha256'] and val[role]['checkpoint_sha256']==item['checkpoint_sha256'],'three_VAL_candidate_binding_failed')
  candidate_hashes[role]={k:item[k] for k in ['manifest_sha256','checkpoint_sha256','weights_sha256','provenance_sha256']}
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
  parent=family.DEST/'fresh-plan.json' if role=='fresh' else retry.PLAN
  directory=family.DEST/'parallel-fresh-v10' if role=='fresh' else retry.OUT
  reportpath=family.DEST/'fresh-comparison.json' if role=='fresh' else directory/'comparison.json'
  perpath=family.DEST/'fresh-comparison-per-image.json' if role=='fresh' else directory/'comparison-per-image.json'
  attemptpath=family.DEST/'fresh-attempt.json' if role=='fresh' else retry.ATTEMPT
  plan=controls.read(parent);ep=directory/'execution-plan.json';execution=controls.read(ep);receipt=controls.read(directory/'execution-receipt.json')
  (parallel if role=='fresh' else retry).assert_closure(execution)
  demand(execution['parent_plan']==parent.relative_to(ROOT).as_posix(),'exact_execution_parent_path_required')
  started={'status':'started','plan_sha256':controls.digest(parent),'parallel_protocol_sha256':controls.digest(parallel.PROTOCOL if role=='fresh' else retry.PUBLIC),'writer':'two_process_CPU','selection_allowed':False,'automatic_retry_allowed':False}
  demand(execution['attempt_sha256']==hashlib.sha256(json.dumps(started,sort_keys=True).encode()).hexdigest(),'exclusive_attempt_snapshot_changed')
  if role=='fresh':
   demand(plan==family.comparison_plan(value,frozen,role),'original_fresh_plan_changed');controls.digest(reportpath,protocol['fresh_comparison_sha256']);controls.digest(attemptpath,protocol['fresh_attempt_sha256']);demand(execution['parent_pid']==phases['fresh']['child_pid'],'original_fresh_parent_changed')
  else:
   controls.digest(parent,protocol['corrected_plan_sha256']);count_lineage(protocol,controls.read(family.DEST/'regression-plan.json'),plan,controls.read(family.DEST/'regression-attempt.json'),receipt,controls.read(reportpath))
  for refs in execution['model_closure'].values():
   for ref in refs.values():controls.digest(ROOT/ref['path'],ref['sha256'])
  controls.digest(ROOT/execution['benchmark_manifest'],execution['benchmark_sha256'])
  workers={name:controls.read(directory/(name+'.json')) for name in ['baseline','candidate']}
  worker_hashes[role]={name:controls.digest(directory/(name+'.json')) for name in workers}
  cohorts[role]=validate_role(role,plan,controls.read(attemptpath),execution,receipt,workers,controls.read(perpath),controls.read(reportpath),controls.digest(parent),controls.digest(ep))
 result={'format':'privacy-family-v10-count-retry1-results-v1','count_correction':copy.deepcopy(COUNT_CORRECTION),'selected_family':frozen['selected_family'],'selected_confidence':frozen['selected_confidence'],'revision':copy.deepcopy(REVISION),'candidate_artifact_sha256':candidate_hashes,
  'family_protocol_sha256':parallel.family_sha(),'family_source_sha256':parallel.family_source_sha(),'parallel_protocol_sha256':controls.digest(parallel.PROTOCOL),'parallel_worker_source_sha256':controls.digest(Path(parallel.__file__)),
  'winner_freeze_sha256':controls.digest(family.DEST/'winner-freeze.json'),'publisher_source_sha256':sha256(Path(__file__)),'artifact_sha256':artifacts,'canonical_default_strict_evidence_sha256':canonical,'worker_result_sha256':worker_hashes,'operator_control_sha256':controls.digest(OPERATOR/'queue-receipt.json'),'runtime':copy.deepcopy(RUNTIME),'cohorts':cohorts,
  'research_only':True,'production_approved':False,'human_privacy_reviewed':False,'physical_person_or_route_independence_verified':False,'TEST_used_for_selection':False,
  'limits':['Fresh200 is the original completed once-only comparison; it was not rerun.','The original plan incorrectly expected1400 for an unchanged600-row regression manifest; original failed operator/attempt remains failed.','Corrected regression600 contains300face and300plate original images; no new rows, model or threshold selection.','Known publisher class coverage only; unknown other class is not negative and full privacy protection is unverified.','Model family/grid/large-face gate were frozen on VAL508 before TEST; all four default strict canonical proofs passed.','Per-model/worker CPU latency includes shared-host contention and is not field/camera/full-pipeline10FPS proof.','Physical person/vehicle/route independence remains unverified; research/non-commercial/redistribution limits and original expiries remain.']}
 controls.close();public_whitelist(result);return result,controls

def public_whitelist(report):
 allowed={'count_correction','revision','candidate_artifact_sha256','format','selected_family','selected_confidence','family_protocol_sha256','family_source_sha256','parallel_protocol_sha256','parallel_worker_source_sha256','winner_freeze_sha256','publisher_source_sha256','artifact_sha256','canonical_default_strict_evidence_sha256','worker_result_sha256','operator_control_sha256','runtime','cohorts','research_only','production_approved','human_privacy_reviewed','physical_person_or_route_independence_verified','TEST_used_for_selection','limits'}
 demand(set(report)==allowed,'complete_public_root_whitelist_required')
 demand(report['format']=='privacy-family-v10-count-retry1-results-v1' and report['selected_family'] in ('legacy_v4','rescue_v8','fixed_final12') and finite(report['selected_confidence']) and .01<=report['selected_confidence']<=1,'public_family_contract_required')
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
 demand(report['count_correction']==COUNT_CORRECTION,'public_count_correction_disclosure_required')
 demand(report['revision']==REVISION,'public_revision_disclosure_required')
 demand(set(report['candidate_artifact_sha256'])=={'baseline','rescue_v8','fixed_final12'},'public_three_candidates_required')
 for item in report['candidate_artifact_sha256'].values():demand(set(item)=={'manifest_sha256','checkpoint_sha256','weights_sha256','provenance_sha256'},'public_candidate_hash_whitelist_required')
 hashes(report['candidate_artifact_sha256'])
 for value in walk_strings(report):
  demand(not re.search(r'(?i)\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b',value),'individual_UUID_in_public_report')
 assert_public(report)


def markdown(report):
 return (base.markdown(report).decode('utf-8')+'\n## 회귀 표본 수 계획 오류와 교정\n\n원 계획의1400은 잘못된 count metadata였다. 원 manifest bytes는 그대로600행(얼굴300/번호판300)이다. 원 operator/attempt 실패는 그대로 보존하며, 이미지·모델·threshold를 바꾸지 않은 새 count-only 평가600을 구분한다. 완료된 fresh200은 재실행하지 않았다. 전체1400을 평가했다고 주장하지 않는다.\n').encode('utf-8')

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
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--authorized',action='store_true');p.add_argument('--operator-source-sha256',required=True);a=p.parse_args()
 try:
  demand(a.authorized,'explicit_postcompletion_publication_authorization_required');demand(bool(re.fullmatch('[a-f0-9]{64}',a.operator_source_sha256)),'approved_operator_SHA_required')
  import cv2,onnxruntime
  demand({'Python':sys.version.split()[0],'NumPy':np.__version__,'OpenCV':cv2.__version__,'ORT':onnxruntime.__version__}=={k:RUNTIME[k] for k in ['Python','NumPy','OpenCV','ORT']},'same_direct_CPU_metadata_runtime_required')
  report,controls=gather(a.operator_source_sha256);controls.close();print(json.dumps(publish_pair(report,OUTPUT)));return 0
 except Exception as error:print(json.dumps({'status':'count_retry_publication_refused','error_type':type(error).__name__}));return 1


if __name__=='__main__':raise SystemExit(main())
