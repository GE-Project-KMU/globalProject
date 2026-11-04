"""Predeclare cycle2 comparisons and run once after real strict VAL proofs.

No training/calibration, user recordings, or relaxed numerical fallback. Individual
metadata/predictions are restricted to the private data/training cycle2 root.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
from scripts.public_refinement_data import freeze
from scripts.evaluate_public_refinement import strict_proof
from scripts.compare_lane_models import compare as compare_lane
from scripts.compare_detector_models import compare as compare_detector
DEST=ROOT/'data/training/cycle2-evaluation'
CONFIG={
'lane':{'release':'data/public/a2d2-cycle2-v5/release.json','dev':'data/public/a2d2-cycle2-v5/development.json','fresh':'data/public/a2d2-cycle2-v5/fresh-test.json','baseline':'data/training/lane-refinement-v4/lane.json','candidate':'data/training/lane-cycle2-v5/lane.json','checkpoint':'lane_best.pt','regression':['data/public/a2d2-improvement-v2/quarantined-test/manifest.json','data/public/a2d2-refinement-v3/fresh-test.json']},
'privacy':{'release':'data/training/research-redaction-cycle2-v7/release.json','dev':'data/training/research-redaction-cycle2-v7/development.json','fresh':'data/training/research-redaction-cycle2-v7/fresh-test.json','baseline':'data/training/redaction-inference-options-v4/option-tiled/model.json','candidate':'data/training/redaction-p2-cycle2-v7/model.json','checkpoint':'best.pt','regression':['data/training/research-redaction-v4/release.json','data/training/research-redaction-v6/fresh-test.json']}}
def read(path):return json.loads(workspace_path(path,exists=True).read_text(encoding='utf-8'))
def rel(path):return path.relative_to(ROOT).as_posix()
def prepare(kind):
 cfg=CONFIG[kind];dev=read(cfg['dev']);fresh=read(cfg['fresh']);reg=[];seen=set()
 for source in cfg['regression']:
  for row in read(source):
   if row['split']!='test':continue
   if row['source_hash'] in seen:raise ValueError('historical_regression_duplicate_source')
   seen.add(row['source_hash']);reg.append(row)
 if {r['source_hash'] for r in reg}&{r['source_hash'] for r in fresh}:raise ValueError('fresh_regression_overlap')
 freeze(DEST/f'{kind}-regression-manifest.json',reg)
 val=sorted((r for r in dev if r['split']=='validation'),key=lambda r:r['source_id'])
 if kind=='lane':
  groups=sorted({r['recording'] for r in val});positive=[r for r in val if r.get('positive_pixels',0)>0];chosen=[]
  for group in groups:
   eligible=[r for r in positive if r['recording']==group]
   if not eligible:raise ValueError('positive_VAL_case_per_recording_required')
   chosen.append(eligible[0])
  chosen.append(next(r for r in positive if r['sample_id'] not in {v['sample_id'] for v in chosen}))
 else:
  chosen=[next(r for r in val if r['evaluation_class']==label and any(o['class']==label and o['bbox_xywh_px'][2]*o['bbox_xywh_px'][3]>=64**2 for o in r['objects'])) for label in ['face','license_plate']]
 cases=[{'path':r['path'],'sha256':r['sha256'],'source_id':r['source_id'],'expires_at':r['expires_at'],'label':r.get('evaluation_class','paint')} for r in chosen]
 freeze(DEST/f'{kind}-positive-VAL-cases.json',cases)
 numeric={'format':'cycle2-numeric-gate-plan-v1','task':kind,'positive_cases_sha256':sha256(DEST/f'{kind}-positive-VAL-cases.json'),'positive_VAL_cases':len(cases),'all_lane_VAL_count':len(val) if kind=='lane' else None,'raw_rtol':.001,'raw_atol':.0001,'exact_postprocess_required':True,'CPU_ORT':'1.31.1','old_GPU_proof_reused':False,'TEST_used':False}
 freeze(ROOT/f'docs/reports/{kind.upper()}-CYCLE2-NUMERIC-PLAN.json',numeric)
 for role,bench in [('fresh',workspace_path(cfg['fresh'])),('regression',DEST/f'{kind}-regression-manifest.json')]:
  plan={'created_before_final_predictions':True,'models':{k:cfg[k] for k in ['baseline','candidate']},'baseline_manifest_sha256':sha256(workspace_path(cfg['baseline'])),'candidate_release_sha256':sha256(workspace_path(cfg['release'])),'benchmark_manifest':rel(bench),'benchmark_sha256':sha256(bench),'role':'new_unseen_source_recording_only_not_physical_independence' if role=='fresh' else 'previously_examined_regression_only','test_used_for_fitting_or_selection':False,'physical_independence_verified':False,'raw_rtol':.001,'raw_atol':.0001,'project_acceptance_eligible':False,'production_approved':False,'numeric_plan_sha256':sha256(ROOT/f'docs/reports/{kind.upper()}-CYCLE2-NUMERIC-PLAN.json')}
  freeze(ROOT/f'docs/reports/{kind.upper()}-CYCLE2-{role.upper()}-PLAN.json',plan)
 print(json.dumps({'task':kind,'fresh_samples':len(fresh),'regression_samples':len(reg),'positive_VAL_cases':len(cases),'final_model_predictions':0}),flush=True)

def full_gate(full,manifest_sha,checkpoint_sha,expected):
 if type(expected) is not int or expected<=0:raise ValueError('positive_expected_VAL_count_required')
 if full.get('manifest_sha256')!=manifest_sha or full.get('checkpoint_sha256')!=checkpoint_sha:raise ValueError('all_VAL_artifact_binding_required')
 cases=full.get('cases',[])
 if len(cases)!=expected or any(type(c.get('index')) is not int for c in cases) or {c['index'] for c in cases}!=set(range(expected)):raise ValueError('complete_unique_all_VAL_required')
 if any(full.get(k) is not True for k in ['complete','all_raw_passed','all_post_passed']):raise ValueError('all_VAL_failure_cannot_be_hidden')
 if full.get('rtol')!=.001 or full.get('atol')!=.0001:raise ValueError('default_VAL_tolerance_required')
 reference=full.get('reference_runtime',{})
 if reference.get('ort')!='1.31.1' or reference.get('providers')!=['CPUExecutionProvider'] or reference.get('GPU_initialized') is not False or full.get('app_runtime')!={'engine':'onnxruntime-cpu','version':'1.31.1'}:raise ValueError('restricted_CPU_ORT1311_required')
 names={'app_vs_original_FP32_torch','app_vs_reference_ORT131','reference_ORT131_vs_original_FP32_torch'}
 for case in cases:
  if case.get('raw_passed') is not True or case.get('post_passed') is not True or set(case.get('raw',{}))!=names or set(case.get('post',{}))!={'app','torch','ort'}:raise ValueError('every_VAL_raw_and_post_required')
  if any(v.get('passed') is not True or v.get('rtol')!=.001 or v.get('atol')!=.0001 for v in case['raw'].values()):raise ValueError('VAL_raw_failed_or_relaxed')
  if any(v.get('exact_match') is not True or v.get('mismatching_pixels')!=0 for v in case['post'].values()):raise ValueError('VAL_exact_post_failed')

def gates(kind,plan):
 cfg=CONFIG[kind];numericpath=ROOT/f'docs/reports/{kind.upper()}-CYCLE2-NUMERIC-PLAN.json';numeric=read(numericpath)
 if sha256(numericpath)!=plan['numeric_plan_sha256']:raise ValueError('numeric_plan_changed')
 casepath=DEST/f'{kind}-positive-VAL-cases.json'
 if sha256(casepath)!=numeric['positive_cases_sha256']:raise ValueError('positive_VAL_cases_changed')
 cases=read(casepath);manifest=workspace_path(cfg['candidate'],exists=True);checkpoint=manifest.parent/cfg['checkpoint']
 for i,case in enumerate(cases):
  filename='verification-final.json' if i==0 else (f'verification-val-{i}.json' if kind=='lane' else 'verification-plate.json')
  strict_proof(manifest,manifest.parent/filename,image_sha=case['sha256'],checkpoint_sha=sha256(checkpoint),positive=True)
  if kind=='privacy':
   proof=read(manifest.parent/filename)
   if any(proof.get('detection_postprocess_parity',{}).get(ref,{}).get('application_class_counts',{}).get(case['label'],0)<1 for ref in ['ort','torch']):raise ValueError('known_class_positive_prediction_required')
 if kind=='lane':
  report=read(manifest.parent/'full-validation-131.json');full_gate(report,sha256(manifest),sha256(checkpoint),numeric['all_lane_VAL_count'])
  fullplan=manifest.parent/'full-validation-131-plan.json'
  if sha256(fullplan)!=report['plan_sha256']:raise ValueError('full_VAL_case_plan_changed')
  valplan=read(fullplan);expected=sorted((r for r in read(cfg['dev']) if r['split']=='validation'),key=lambda r:r['source_id'])
  if len(valplan.get('cases',[]))!=len(expected) or any(a['path']!=b['path'] or a['sha256']!=b['sha256'] for a,b in zip(valplan['cases'],expected)):raise ValueError('complete_original_VAL_cases_required')
  if any(c['image_sha256']!=valplan['cases'][c['index']]['sha256'] for c in report['cases']):raise ValueError('full_VAL_image_binding_failed')

def publish(kind):
 cohorts={}
 for role in ['fresh','regression']:
  path=DEST/f'{kind}-{role}.json'
  if not path.exists():continue
  actual=read(path);models={}
  for name,result in actual['results'].items():
   keys=['manifest_sha256','weights_sha256','paint_threshold','input_size','backend','backend_version','confusion','paint_iou','paint_precision','paint_recall','paint_dice','latency_seconds'] if kind=='lane' else ['manifest_sha256','weights_sha256','operating_confidence','architecture','input_size','backend','backend_version','ap_metrics','operating_metrics','fixed_035_reference','latency_seconds']
   models[name]={k:result[k] for k in keys if k in result}
  cohorts[role]={'samples':actual['samples'],'scope':actual['scope'],'models':models}
 atomic_json(ROOT/f'docs/reports/{kind.upper()}-CYCLE2-RESULTS.json',{'format':'public-cycle2-results-v1','task':kind,'cohorts':cohorts,'old_TEST_is_regression_only':True,'TEST_used_for_selection':False,'physical_person_or_route_independence_verified':False,'user_recordings_used':False,'source_expiries_extended':False,'research_only':True,'production_approved':False,'project_acceptance_eligible':False,'limits':['Publisher proxy/known-class conditional annotations, not wear/complete privacy ground truth','Source/recording holdout, not physical person/route independence','Shared-host CPU single-model costs, not10FPS or pipeline acceptance']})

def run(kind,role):
 planpath=ROOT/f'docs/reports/{kind.upper()}-CYCLE2-{role.upper()}-PLAN.json';plan=read(planpath);gates(kind,plan)
 attempt=DEST/f'{kind}-{role}-attempt.json'
 if attempt.exists():raise ValueError('one_shot_comparison_already_started_no_automatic_retry')
 atomic_json(attempt,{'plan_sha256':sha256(planpath),'status':'started','selection_allowed':False})
 result=(compare_lane if kind=='lane' else compare_detector)(planpath,DEST/f'{kind}-{role}.json');publish(kind)
 atomic_json(attempt,{'plan_sha256':sha256(planpath),'status':'completed','selection_allowed':False,'samples':result['samples']})
 print(json.dumps({'task':kind,'cohort':role,'samples':result['samples']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('kind',choices=['lane','privacy']);p.add_argument('role',nargs='?',choices=['fresh','regression']);a=p.parse_args();prepare(a.kind) if a.action=='prepare' else run(a.kind,a.role)
