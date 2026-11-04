"""V9 metadata-only publication: three pre-TEST families, original V8 preserved."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256
from scripts import finalize_privacy_family_results as prior
from scripts import evaluate_privacy_family_v9 as family
from scripts import compare_privacy_family_v9_parallel as parallel
from scripts.improve_redaction_finalize import assert_public
Controls=prior.Controls;encoded=prior.encoded;same=prior.same;demand=prior.demand;finite=prior.finite
indexed_scores=prior.indexed_scores;records_valid=prior.records_valid;counts_valid=prior.counts_valid
validate_role=prior.validate_role;walk_strings=prior.walk_strings;lexical_control_path=prior.lexical_control_path
CLASSES=list(prior.CLASSES);COUNTS=dict(prior.COUNTS);RUNTIME=copy.deepcopy(prior.RUNTIME);ARCHITECTURES=set(prior.ARCHITECTURES)
PHASES=['baseline-VAL508','rescue-VAL508-import','fixed12-VAL508','winner-freeze','strict-baseline-face','strict-baseline-license_plate','strict-winner-face','strict-winner-license_plate','fresh','regression']
OPERATOR=ROOT/'.cache/privacy-family-v9-operator'
OUTPUT=ROOT/'docs/reports/PRIVACY-FAMILY-V9-RESULTS.json'
PARENT_PUBLISHER_SHA='12d9a6cee2aa9a32176f9802ca7b5e08d44f4072f64f000dd0914ff918ba6a8b'
PREDECL=ROOT/'docs/reports/PRIVACY-FAMILY-V9-PREDECLARATION.json'
PREDECL_SHA='2717f0a737aa73d72afca8cc567519d0630c3ae2ebe77b980d608a9018228117'
REVISION={'development_informed_revision':True,'completed_face_development_epochs_observed':6,'declared_before_final12_weights':True,'declared_before_TEST':True,'original_V8_VAL_or_TEST_performed':False,'predeclaration_sha256':PREDECL_SHA,'original_V8_stop_sha256':'4d8e065302d758d790745721c6045b583bdb4dff8555762ae52696a3919e7a1b','families':['legacy_v4','rescue_v8','fixed_final12'],'AP_selected_alias':'rescue_v8_AP_selected is descriptive alias of actual rescue_v8 role'}
REVISION.update(original_V8_family_protocol_sha256='5777de565111135f3793d247a332d5bb4c5cae4635378f8cfacfcbbff3cb4f59',original_V8_family_source_sha256='95b5dc346e30f0ee8341afdcdc874754507ce4c937f55261bdcb71b15779ca10',original_V8_parallel_protocol_sha256='fbbcc98f6a1c0e8a21abe32387e991c44d2533e80747e33823bc5c9469906c35',original_V8_parallel_source_sha256='3171d8208f47553b893358a78e5a63be17c8d203e547366dc5f0c2970cf65f31',parent_publisher_sha256=PARENT_PUBLISHER_SHA)

def validate_queue(queue,phases,operator_sha):
 demand(queue.get('status')=='completed' and queue.get('operator_source_sha256')==operator_sha and queue.get('CPU_execution_runtime')==RUNTIME,'completed_approved_operator_required')
 demand(queue.get('automatic_retry_allowed') is False and queue.get('user_recordings_used') is False,'no_retry_or_user_recordings_required')
 demand(set(phases)==set(PHASES),'all_operator_phases_required')
 for name,phase in phases.items():
  demand(phase.get('phase')==name and phase.get('status')=='completed' and type(phase.get('child_exit_code')) is int and phase['child_exit_code']==0 and phase.get('own_child_running_after_cleanup') is False,'active_phase_failure_or_unclean_child')
  demand(phase.get('CPU_execution_runtime')==RUNTIME and phase.get('automatic_retry_allowed') is False and phase.get('root_authorized') is True,'approved_same_CPU_runtime_required')
  if name in ('baseline-VAL508','fixed12-VAL508'):
   demand(phase.get('observer_child_stopped') is True and phase.get('observer_thread_stopped') is True and phase.get('device_memory',{}).get('over_budget_observed') is False,'clean_GPU_observer_required')


def gather(operator_sha):
 controls=Controls();controls.digest(ROOT/'scripts/finalize_privacy_family_results.py',PARENT_PUBLISHER_SHA);controls.digest(PREDECL,PREDECL_SHA);parallel.check_protocol();family.protocol();family.completed_rescue();family.completed_final12()
 from scripts.supervise_final12_cpu_export import completed_execution
 completed_execution()  # Metadata+hashes only; failed/running root export forbids publication.
 controls.digest(family.FINAL12/'root-export-execution.json');controls.digest(family.FINAL12/'export-receipt.json');controls.digest(family.FINAL12/'candidate-freeze.json')
 queue=controls.read(OPERATOR/'queue-receipt.json');phases={name:controls.read(OPERATOR/(name+'-receipt.json')) for name in PHASES};validate_queue(queue,phases,operator_sha)
 controls.digest(ROOT/'.cache/privacy_family_v9_authorized_queue.py',operator_sha)
 # Only active receipts count as failures. The explicitly superseded failed
 # preflight remains preserved and bound inside the successful retry controls.
 for phase in phases.values():
  for path,digest in phase['binding']['source_sha256'].items():controls.digest(ROOT/path,digest)
  for path,digest in phase['binding']['artifacts_sha256'].items():controls.digest(ROOT/path,digest)
  for path,digest in phase['output_sha256'].items():controls.digest(ROOT/path,digest)
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
  parent=family.DEST/(role+'-plan.json');plan=controls.read(parent);demand(plan==family.comparison_plan(value,frozen,role),'exact_original_benchmark_plan_required')
  directory=family.DEST/('parallel-'+role+'-v9');ep=directory/'execution-plan.json';execution=controls.read(ep);parallel.assert_closure(execution)
  demand(execution['parent_pid']==phases[role]['child_pid'] and execution['parent_plan']==parent.relative_to(ROOT).as_posix(),'direct_operator_execution_parent_binding_failed')
  started={'status':'started','plan_sha256':controls.digest(parent),'parallel_protocol_sha256':controls.digest(parallel.PROTOCOL),'writer':'two_process_CPU','selection_allowed':False,'automatic_retry_allowed':False}
  demand(execution['attempt_sha256']==hashlib.sha256(json.dumps(started,sort_keys=True).encode()).hexdigest(),'original_exclusive_attempt_snapshot_changed')
  for refs in execution['model_closure'].values():
   for ref in refs.values():controls.digest(ROOT/ref['path'],ref['sha256'])
  controls.digest(ROOT/execution['benchmark_manifest'],execution['benchmark_sha256'])
  workers={name:controls.read(directory/(name+'.json')) for name in ['baseline','candidate']};reportpath=family.DEST/(role+'-comparison.json');report=controls.read(reportpath)
  controls.digest(reportpath,queue['results_sha256'][role]);worker_hashes[role]={name:controls.digest(directory/(name+'.json')) for name in workers}
  cohorts[role]=validate_role(role,plan,controls.read(family.DEST/(role+'-attempt.json')),execution,controls.read(directory/'execution-receipt.json'),workers,controls.read(family.DEST/(role+'-comparison-per-image.json')),report,controls.digest(parent),controls.digest(ep))
 result={'format':'privacy-family-v9-final-results-v1','revision':copy.deepcopy(REVISION),'candidate_artifact_sha256':candidate_hashes,'selected_family':frozen['selected_family'],'selected_confidence':frozen['selected_confidence'],
  'family_protocol_sha256':parallel.family_sha(),'family_source_sha256':parallel.family_source_sha(),'parallel_protocol_sha256':controls.digest(parallel.PROTOCOL),'parallel_worker_source_sha256':controls.digest(ROOT/'scripts/compare_privacy_family_v9_parallel.py'),
  'winner_freeze_sha256':controls.digest(family.DEST/'winner-freeze.json'),'publisher_source_sha256':sha256(Path(__file__)),'artifact_sha256':artifacts,'canonical_default_strict_evidence_sha256':canonical,
  'worker_result_sha256':worker_hashes,'operator_control_sha256':controls.digest(OPERATOR/'queue-receipt.json'),'runtime':copy.deepcopy(RUNTIME),'cohorts':cohorts,
  'research_only':True,'production_approved':False,'human_privacy_reviewed':False,'physical_person_or_route_independence_verified':False,'TEST_used_for_selection':False,
  'limits':['Fresh200 is a once-only paired benchmark; old1400 is previously seen regression, never TRAIN or selection.',
   'Publisher-known class coverage only; unknown other class is not a negative. Full privacy protection is unverified.',
   'Three families were fixed after six observed development epochs but before final12 weights and TEST; this is an exploratory development-informed revision, not preregistration before any VAL.',
   'All four face/plate canonical proofs use unchanged default raw/postprocess gates; numerical agreement is not accuracy.',
   'Two concurrent CPU2/inter1 workers use CV1; unchanged canonical verifier uses CV2 in the same Python/NumPy/OpenCV/ORT environment.',
   'Shared-host per-model and worker wall latency is not isolated speed, whole-pipeline10FPS, camera or field acceptance.',
   'Physical person, vehicle and route independence remains unverified; source grouping is a proxy.',
   'Original expiry and non-commercial research/redistribution limits remain; no raw photos or learned weights are published.',
   'Earlier failed preflight remains historical evidence; every active retry operator phase and permanent attempt must have completed.']}
 controls.close();assert_public(result);return result,controls


def public_whitelist(report):
 allowed={'revision','candidate_artifact_sha256','format','selected_family','selected_confidence','family_protocol_sha256','family_source_sha256','parallel_protocol_sha256','parallel_worker_source_sha256','winner_freeze_sha256','publisher_source_sha256','artifact_sha256','canonical_default_strict_evidence_sha256','worker_result_sha256','operator_control_sha256','runtime','cohorts','research_only','production_approved','human_privacy_reviewed','physical_person_or_route_independence_verified','TEST_used_for_selection','limits'}
 demand(set(report)==allowed,'complete_public_root_whitelist_required')
 demand(report['format']=='privacy-family-v9-final-results-v1' and report['selected_family'] in ('legacy_v4','rescue_v8','fixed_final12') and finite(report['selected_confidence']) and .01<=report['selected_confidence']<=1,'public_family_contract_required')
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
 demand(report['revision']==REVISION,'public_revision_disclosure_required')
 demand(set(report['candidate_artifact_sha256'])=={'baseline','rescue_v8','fixed_final12'},'public_three_candidates_required')
 for item in report['candidate_artifact_sha256'].values():demand(set(item)=={'manifest_sha256','checkpoint_sha256','weights_sha256','provenance_sha256'},'public_candidate_hash_whitelist_required')
 hashes(report['candidate_artifact_sha256'])
 for value in walk_strings(report):
  demand(not re.search(r'(?i)\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b',value),'individual_UUID_in_public_report')
 assert_public(report)


def markdown(report):
 body=prior.markdown(report).decode('utf-8')
 return (body+'\n## V9 개발 관측 후 정책 변경\n\n완료된 얼굴 VAL 6epoch 관측 후 original AP 선택본과 고정 epoch12 EMA를 별도 후보로 비교하도록 변경했다. 최종12 가중치 생성 및 TEST 전에 선언했으며, 모든 VAL 이전의 사전등록으로 주장하지 않는다. 원 V8 평가와 TEST는 실행하지 않았다.\n\n세 후보는 legacy_v4, rescue_v8(설명용 alias rescue_v8_AP_selected), fixed_final12다. 학습 source·12epoch recipe·큰 얼굴 gate·confidence grid·TEST cohort를 바꾸지 않았다. TEST 뒤 모델·임계값을 재선정하지 않는다.\n').encode('utf-8')

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
 except Exception as error:print(json.dumps({'status':'privacy_V9_publication_refused','error_type':type(error).__name__}));return 1
if __name__=='__main__':raise SystemExit(main())
