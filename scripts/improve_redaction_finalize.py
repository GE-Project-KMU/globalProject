"""Metadata orchestration only: frozen VAL parity cases and predeclared comparisons.

Never reads native model.json/summary/TEST scores to choose an inference option.
All raw photographs, sample identities and per-image reports remain local.
"""
import argparse
import json
import math
import re
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256,workspace_path
from scripts.setup_training import environment

OPTIONS=ROOT/'data/training/redaction-inference-options-v4'
TRAINING=ROOT/'data/training/redaction-specialists-v4-efficient'
VAL=ROOT/'data/training/research-redaction-v4/validation-only.json'
PROTOCOL=ROOT/'docs/reports/REDACTION-APP-COMPARISON-PROTOCOL.json'
CLASSES=('face','license_plate')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rel(path):return path.relative_to(ROOT).as_posix()

def frozen():
    fp=OPTIONS/'option-selection-freeze.json';freeze=read(fp)
    if freeze['test_images_or_metrics_read'] is not False:raise ValueError('test_blind_freeze_required')
    for item in freeze['options'].values():
        path=workspace_path(item['manifest'],exists=True);manifest=read(path)
        if sha256(path)!=item['model_manifest_sha256'] or manifest['confidence']!=item['confidence'] or sha256(workspace_path(manifest['weights'],exists=True))!=item['onnx_sha256']:raise ValueError('frozen_option_changed')
    if sha256(TRAINING/'best.pt')!=freeze['shared_checkpoint_sha256']:raise ValueError('checkpoint_changed')
    if sha256(VAL)!=freeze['validation_manifest_sha256']:raise ValueError('validation_changed')
    return freeze,fp

def strict_report(report,item,checkpoint_sha,case):
    if report.get('manifest_sha256')!=item['model_manifest_sha256'] or report.get('weight_sha256')!=item['onnx_sha256'] or report.get('checkpoint_sha256')!=checkpoint_sha or report.get('image_sha256')!=case['image_sha256']:raise ValueError('strict_report_artifact_or_image_binding_failed')
    required=('numerical_parity_passed','postprocess_parity_passed','artifact_contract_verified','preprocess_contract_verified','finite_output','torch_checkpoint_compared')
    if any(report.get(k) is not True for k in required) or report.get('synthetic_input') is not False:raise ValueError('real_image_raw_and_postprocess_gate_required')
    for key in ('app_vs_ort','app_vs_torch','ort_vs_torch','app_repeatability'):
        value=report.get(key,{})
        if value.get('passed') is not True or value.get('rtol')!=.001 or value.get('atol')!=.0001:raise ValueError('default_strict_tolerances_required')
    if case.get('gt_count',0)<1 or case.get('predicted_known_class_count',0)<1 or report.get('adapter_result_count',0)<1:raise ValueError('positive_known_class_parity_coverage_required')

def gate_reports(freeze,fp):
    item=freeze['options'][freeze['selected_option']];manifest=workspace_path(item['manifest'],exists=True)
    cases=read(manifest.parent/'parity-cases.json')
    if cases['option_selection_freeze_sha256']!=sha256(fp) or cases['manifest_sha256']!=item['model_manifest_sha256'] or cases['checkpoint_sha256']!=freeze['shared_checkpoint_sha256']:raise ValueError('private_case_receipt_binding_failed')
    reports={name:manifest.parent/filename for name,filename in [('face','verification-final.json'),('license_plate','verification-plate.json')]}
    for label,path in reports.items():strict_report(read(path),item,freeze['shared_checkpoint_sha256'],cases['cases'][label])
    return reports,cases

def case_coverage():
    """Check the ALREADY fixed parity samples; never replace a failed sample."""
    from agent.pipeline import intersection_over_union
    freeze,fp=frozen();reports,cases=gate_reports(freeze,fp)
    details_path=OPTIONS/'validation-details.json'
    if sha256(details_path)!=freeze['validation_details_sha256']:raise ValueError('validation_detail_changed')
    details=read(details_path);coverage={}
    for label,case in cases['cases'].items():
        gt=[r for r in details['truth'] if r['sample_id']==case['sample_id'] and r['class']==label]
        predicted=sorted([r for r in details['predictions'][freeze['selected_option']] if r['sample_id']==case['sample_id'] and r['class']==label and r['confidence']>=freeze['options'][freeze['selected_option']]['confidence']],key=lambda r:r['confidence'],reverse=True)
        matched=set()
        for prediction in predicted:
            overlap,index=max(((intersection_over_union(prediction['bbox_xywh_px'],row['bbox_xywh_px']),i) for i,row in enumerate(gt) if i not in matched),default=(0,None))
            if overlap>=.5:matched.add(index)
        if len(gt)!=case['gt_count'] or len(predicted)!=case['predicted_known_class_count']:raise ValueError('fixed_case_count_changed')
        coverage[label]={'gt_count':len(gt),'pred_count':len(predicted),'matched_count':len(matched)}
    return coverage

def record_coverage():
    freeze,fp=frozen();directory=workspace_path(freeze['options'][freeze['selected_option']]['manifest']).parent;destination=directory/'parity-known-class-detections.json'
    if destination.exists():raise ValueError('coverage_evidence_already_exists')
    coverage=case_coverage();atomic_json(destination,{'selection_rule_unchanged':True,'new_sample_selection':False,'option_selection_freeze_sha256':sha256(fp),'private_case_receipt_sha256':sha256(directory/'parity-cases.json'),'validation_details_sha256':freeze['validation_details_sha256'],'matching':'confidence_descending_one_to_one_same_class_IoU0.5','coverage':coverage})
    print(json.dumps({'fixed_case_coverage':coverage,'both_fixed_classes_have_actual_true_positive':all(v['matched_count']>=1 for v in coverage.values()),'evidence_sha256':sha256(destination)}),flush=True)

def prepare():
    freeze,fp=frozen();selected=freeze['selected_option'];item=freeze['options'][selected]
    manifest=workspace_path(item['manifest'],exists=True);directory=manifest.parent
    destination=directory/'parity-cases.json'
    if destination.exists():raise ValueError('cases_already_frozen')
    detail=OPTIONS/'validation-details.json'
    if sha256(detail)!=freeze['validation_details_sha256']:raise ValueError('validation_detail_changed')
    details=read(detail);rows={r['sample_id']:r for r in read(VAL)};cases={}
    for label in CLASSES:
        gt_ids={x['sample_id'] for x in details['truth'] if x['class']==label}
        predictions=[x for x in details['predictions'][selected] if x['class']==label and x['confidence']>=item['confidence']]
        chosen=sorted(gt_ids&{x['sample_id'] for x in predictions})[0];row=rows[chosen]
        if row['split']!='validation' or row['evaluation_class']!=label:raise ValueError('positive_known_class_validation_required')
        if sha256(workspace_path(row['path'],exists=True))!=row['sha256']:raise ValueError('case_image_changed')
        cases[label]={'sample_id':chosen,'image_path':row['path'],'image_sha256':row['sha256'],'gt_count':sum(x['sample_id']==chosen and x['class']==label for x in details['truth']),'predicted_known_class_count':sum(x['sample_id']==chosen for x in predictions)}
    receipt={'option_selection_freeze_sha256':sha256(fp),'protocol_sha256':sha256(PROTOCOL),'selected_option':selected,'manifest':item['manifest'],'manifest_sha256':item['model_manifest_sha256'],'checkpoint_sha256':freeze['shared_checkpoint_sha256'],'selection_rule':'For each known class, lexicographically first VAL sample_id with positive publisher GT and at least one same-class APP prediction at frozen confidence; parity coverage only, no correctness/metric criterion.','cases':cases}
    atomic_json(destination,receipt)
    print(json.dumps({'selected_option':selected,'manifest_sha256':item['model_manifest_sha256'],'cases':{c:{k:v for k,v in row.items() if k in ('gt_count','predicted_known_class_count')} for c,row in cases.items()},'private_case_receipt_sha256':sha256(destination)}),flush=True)

def verify(label):
    freeze,fp=frozen();manifest=workspace_path(freeze['options'][freeze['selected_option']]['manifest'],exists=True)
    cases=read(manifest.parent/'parity-cases.json');case=cases['cases'][label]
    if cases['option_selection_freeze_sha256']!=sha256(fp):raise ValueError('case_freeze_changed')
    if sha256(workspace_path(case['image_path'],exists=True))!=case['image_sha256']:raise ValueError('case_image_changed')
    name='verification-final.json' if label=='face' else 'verification-plate.json';output=manifest.parent/name
    if output.exists():raise ValueError('verification_already_exists_do_not_reselect')
    command=[sys.executable,str(ROOT/'scripts/verify_trained_models.py'),str(manifest),'--checkpoint',str(TRAINING/'best.pt'),'--image',str(workspace_path(case['image_path'],exists=True)),'--output',str(output)]
    env=environment();env['CUDA_VISIBLE_DEVICES']=''
    result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
    (manifest.parent/(label+'-verification-command.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    report=read(output) if output.exists() else {}
    passed=result.returncode==0 and report.get('numerical_parity_passed') is True and report.get('postprocess_parity_passed') is True and report.get('artifact_contract_verified') is True
    print(json.dumps({'class':label,'exit_code':result.returncode,'passed':passed,'status':report.get('status'),'raw_parity':report.get('numerical_parity_passed'),'postprocess_parity':report.get('postprocess_parity_passed'),'adapter_result_count':report.get('adapter_result_count'),'known_class_prediction_count':case['predicted_known_class_count'],'report_sha256':sha256(output) if output.exists() else None}),flush=True)
    if not passed:raise SystemExit(2)

def plans():
    freeze,fp=frozen();item=freeze['options'][freeze['selected_option']];manifest=workspace_path(item['manifest'],exists=True)
    reports,cases=gate_reports(freeze,fp)
    protocol=read(PROTOCOL)
    if sha256(PROTOCOL)!=cases['protocol_sha256'] or protocol['recorded_before_fresh_test_scores_access'] is not True:raise ValueError('predeclared_protocol_changed')
    if sha256(workspace_path(protocol['baseline_manifest'],exists=True))!=protocol['baseline_manifest_sha256']:raise ValueError('baseline_changed')
    for name,benchmark in protocol['benchmarks'].items():
        path=ROOT/f'docs/reports/REDACTION-APP-{name.upper()}-COMPARISON-PLAN.json'
        if path.exists():raise ValueError('comparison_plan_already_materialized')
        plan={'models':{'baseline':protocol['baseline_manifest'],'candidate':item['manifest']},'baseline_manifest_sha256':protocol['baseline_manifest_sha256'],'candidate_release_sha256':protocol['candidate_release_sha256'],'benchmark_manifest':benchmark['benchmark_manifest'],'benchmark_sha256':benchmark['benchmark_sha256'],'role':benchmark['role'],'samples':benchmark['samples'],'protocol_sha256':sha256(PROTOCOL),'option_selection_freeze_sha256':sha256(fp),'candidate_manifest_sha256':item['model_manifest_sha256'],'verification_report_sha256':{key:sha256(value) for key,value in reports.items()},'app_floor_confidence':protocol['app_floor_confidence'],'fixed_reference_confidence':protocol['fixed_reference_confidence'],'nms_iou':protocol['nms_iou'],'test_used_for_selection':False,'production_approved':False}
        atomic_json(path,plan)
    print(json.dumps({'comparison_plans_materialized':True,'protocol_sha256':sha256(PROTOCOL),'selected_option':freeze['selected_option'],'test_scores_read':False}),flush=True)

PRIVATE_KEYS={'sample_id','source_id','image_id','image_sha256','source_image_sha256','source_hash','source_member','sample_hash','bbox_xywh_px','bbox','bboxes','pixel_coordinates','paint_mask_rle','polygon','segmentation','detections','path','image_path','mask_path','source_path','source_filename','filename','file_name','predictions','truth','pixels','image_bytes','mask_bytes','image_base64','raw_image','original_image','thumbnail','weights','weights_path','checkpoint_path','model_path','artifact_path','manifest','benchmark_manifest','source_release_path'}
PRIVATE_KEY_TOKENS={''.join(c for c in key if c.isalnum()) for key in PRIVATE_KEYS}
def assert_public(value):
    """Refuse individual records/paths rather than silently sanitizing evidence."""
    if isinstance(value,dict):
        for key,item in value.items():
            if ''.join(c for c in str(key).lower() if c.isalnum()) in PRIVATE_KEY_TOKENS:raise ValueError('private_field_in_public_aggregate')
            assert_public(item)
    elif isinstance(value,list):
        for item in value:assert_public(item)
    elif isinstance(value,str):
        lowered=value.lower().replace('\\','/')
        suffixes=('.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff','.gif','.npy','.npz','.pth','.pt','.ckpt','.safetensors','.onnx','.zip','.tar','.gz')
        if re.search(r'(?<![a-z])[a-z]:/|(?:^|[\s\"\'`(<])(?:data|\.cache)/|data:image/|data:application/octet-stream',lowered) or any(lowered.endswith(x) for x in suffixes):raise ValueError('private_path_in_public_aggregate')

def count_metrics(value):
    for key in ('gt_count','prediction_count','tp','fp','fn'):
        if type(value.get(key)) is not int or value[key]<0:raise ValueError('invalid_aggregate_count')
    if value['gt_count']<1 or value['tp']+value['fn']!=value['gt_count'] or value['tp']+value['fp']!=value['prediction_count']:raise ValueError('aggregate_counts_do_not_balance')
    expected={'recall':value['tp']/value['gt_count'],'precision':value['tp']/value['prediction_count'] if value['prediction_count'] else 0.}
    for key,number in expected.items():
        if type(value.get(key)) not in (int,float) or not math.isfinite(value[key]) or abs(value[key]-number)>1e-12:raise ValueError('metric_count_mismatch')
    if type(value.get('ap50')) not in (int,float) or not 0<=value['ap50']<=1:raise ValueError('invalid_ap50')

def aggregate_comparison(report,plan,plan_sha,benchmark,baseline_sha,candidate_sha):
    assert_public(report)
    if report.get('samples')!=benchmark['samples'] or report.get('benchmark_sha256')!=benchmark['benchmark_sha256'] or report.get('plan_sha256')!=plan_sha or report.get('scope')!=benchmark['role']:raise ValueError('comparison_plan_or_benchmark_binding_failed')
    if set(report.get('results',{}))!={'baseline','candidate'}:raise ValueError('exactly_one_frozen_pair_required')
    if report.get('test_used_for_selection') is not False or report.get('project_acceptance_passed') is not False or report.get('known_class_conditional') is not True:raise ValueError('conditional_research_disclosure_required')
    for name,expected_sha in [('baseline',baseline_sha),('candidate',candidate_sha)]:
        model=report['results'][name]
        if model['manifest_sha256']!=expected_sha:raise ValueError('comparison_model_binding_failed')
        for group in ('ap_metrics','operating_metrics','fixed_035_reference'):
            if set(model[group]['classes'])!=set(CLASSES):raise ValueError('both_class_metrics_required')
            for label,values in model[group]['classes'].items():
                count_metrics(values)
                if values['gt_count']!=model['ap_metrics']['classes'][label]['gt_count']:raise ValueError('metric_group_support_mismatch')
    changes={}
    for label in CLASSES:
        a,b=(report['results'][name] for name in ('baseline','candidate'))
        if a['ap_metrics']['classes'][label]['gt_count']!=b['ap_metrics']['classes'][label]['gt_count']:raise ValueError('same_data_support_mismatch')
        changes[label]={key+'_percentage_point_change':100*(b['ap_metrics' if key=='ap50' else 'operating_metrics']['classes'][label][key]-a['ap_metrics' if key=='ap50' else 'operating_metrics']['classes'][label][key]) for key in ('ap50','recall','precision')}
    return {'samples':report['samples'],'role':benchmark['role'],'benchmark_sha256':report['benchmark_sha256'],'plan_sha256':plan_sha,'results':report['results'],'changes':changes,'candidate_min_class_recall':min(v['recall'] for v in report['results']['candidate']['operating_metrics']['classes'].values()),'candidate_both_class_recall95_goal_met':all(v['recall']>=.95 for v in report['results']['candidate']['operating_metrics']['classes'].values())}

def publish():
    freeze,fp=frozen();reports,cases=gate_reports(freeze,fp);protocol=read(PROTOCOL)
    if sha256(PROTOCOL)!=cases['protocol_sha256']:raise ValueError('protocol_changed')
    item=freeze['options'][freeze['selected_option']];comparisons={};evidence={}
    for name,benchmark in protocol['benchmarks'].items():
        pp=ROOT/f'docs/reports/REDACTION-APP-{name.upper()}-COMPARISON-PLAN.json';plan=read(pp)
        if plan['protocol_sha256']!=sha256(PROTOCOL) or plan['option_selection_freeze_sha256']!=sha256(fp):raise ValueError('materialized_plan_receipt_changed')
        path=OPTIONS/f'app-{name}-comparison.json'
        comparisons[name]=aggregate_comparison(read(path),plan,sha256(pp),benchmark,protocol['baseline_manifest_sha256'],item['model_manifest_sha256']);evidence[name+'_comparison_sha256']=sha256(path)
    validation=read(OPTIONS/'validation-aggregate.json')
    if sha256(OPTIONS/'validation-aggregate.json')!=freeze['validation_aggregate_sha256'] or validation['selected_option']!=freeze['selected_option']:raise ValueError('validation_aggregate_binding_failed')
    report={'format':'redaction-improvement-results-v1','selected_option':freeze['selected_option'],'architecture':comparisons['fresh']['results']['candidate']['architecture'],'input_size':comparisons['fresh']['results']['candidate']['input_size'],'operating_confidence':item['confidence'],'checkpoint_sha256':freeze['shared_checkpoint_sha256'],'selected_manifest_sha256':item['model_manifest_sha256'],'selected_onnx_sha256':item['onnx_sha256'],'protocol_sha256':sha256(PROTOCOL),'option_selection_freeze_sha256':sha256(fp),'validation_selection':{'counts':validation['validation_counts'],'calibrations':validation['calibrations'],'predict_seconds':validation['timing_seconds'],'test_images_or_metrics_read':False},'cpu_parity':{'face_and_plate_default_strict_passed':True,'rtol':.001,'atol':.0001,'evidence_sha256':{label:sha256(path) for label,path in reports.items()},'fixed_val_case_detection_coverage':case_coverage(),'case_selection_rule':'Lexicographic first positive-GT VAL case with same-class prediction at frozen confidence; no correctness criterion, then independent IoU0.5 match check without replacement.'},'comparisons':comparisons,'comparison_evidence_sha256':evidence,'source_licenses':['CC-BY-NC-ND-4.0','MIT'],'research_only':True,'production_approved':False,'project_acceptance_passed':False,'human_privacy_reviewed':False,'test_used_for_selection':False,'limits':['Known publisher class per original frame only; unknown other class excluded from negative scoring.','Fresh source holdout400 is the primary paired comparison; previously examined400 is regression only.','Higher AP50/recall does not imply higher precision; false positives and latency are reported without reselection.','95% minimum class recall goal unchanged; any failure remains failure.','CPU predict timing includes preprocessing/inference/postprocessing, excludes image decode and complete vehicle pipeline. No field FPS or ZED2i agency acceptance.','WIDER event/CCPD identity/source-SHA/dHash guards do not prove independent physical persons or routes. Negative-only driving frame FPR and exhaustive privacy coverage are not established.','Private offline non-commercial research only; no external photograph or learned-weight redistribution clearance. Original source expiries retained.','Native standard training scores are historical context only; all main results use the frozen selected option in the actual application.']}
    assert_public(report);dest=ROOT/'docs/reports/REDACTION-IMPROVEMENT-RESULTS.json'
    if dest.exists() or dest.with_suffix('.md').exists():raise ValueError('public_results_already_exist')
    atomic_json(dest,report)
    lines=['# 얼굴·번호판 검출 개선: 실제 앱 CPU 비교','','선택은 봉인된 시험 점수를 열기 전에 VAL508장으로 고정했다. 선택한 후보는 두 YOLOX-Nano 전문모델이며 얼굴 global+4타일, 번호판1회 추론을 수행한다. 단일 Nano가 아니다. 두 클래스의 실제 양성 VAL 영상에서 기본 strict 수치·후처리 정합성을 모두 통과했다.','','## 동일 새 시험 원본400장: 주 비교','','|대상|AP50 이전→선택|재현율 이전→선택|정밀도 이전→선택|','|---|---:|---:|---:|']
    def table_rows(comp):
        output=[]
        for label,title in [('face','얼굴'),('license_plate','번호판')]:
            a,b=(comp['results'][name] for name in ('baseline','candidate'))
            values=[]
            for key,group in [('ap50','ap_metrics'),('recall','operating_metrics'),('precision','operating_metrics')]:values.append(f"{a[group]['classes'][label][key]*100:.2f}% → {b[group]['classes'][label][key]*100:.2f}%")
            output.append('|'+title+'|'+'|'.join(values)+'|')
        return output
    lines+=table_rows(comparisons['fresh'])
    lines+=['',f"선택 confidence는 {item['confidence']}이며 시험 결과로 바꾸지 않았다. 최소 클래스 재현율95% 목표: **{'달성' if comparisons['fresh']['candidate_both_class_recall95_goal_met'] else '미달'}**. AP50·재현율 상승과 정밀도 저하를 함께 해석해야 한다.",'','고정 parity 사례는 양성 GT와 동일 클래스 예측이 존재하는 VAL 표본의 ID 사전순 첫 건이었다. 정확한 검출 여부로 표본을 골라 바꾸지 않았다. 고정 후 별도 IoU0.5 일대일 확인 결과의 GT/예측/정답 대응 수만 집계 JSON에 보관했다. 수치 정합성은 전체 검출 성능의 증거와 구분한다.','','## 이미 확인한 과거 원본400장: 회귀 비교','','|대상|AP50 이전→선택|재현율 이전→선택|정밀도 이전→선택|','|---|---:|---:|---:|']+table_rows(comparisons['regression'])
    latency=comparisons['fresh']['results'];a,b=(latency[name]['latency_seconds'] for name in ('baseline','candidate'))
    lines+=['','## 비용과 한계','',f"새400장 actual app CPU predict 중앙값은 {a['median']*1000:.1f}ms → {b['median']*1000:.1f}ms, P95는 {a['p95']*1000:.1f}ms → {b['p95']*1000:.1f}ms였다. 전처리·추론·후처리를 포함하고 이미지 decode와 차량 전체 파이프라인은 제외한다. 실차 FPS 인수 결과가 아니다.",'','알려진 클래스만 주석된 연구 자료의 조건부 지표이며 다른 클래스 미주석을 음성으로 세지 않았다. 독립 인물·노선, 음성 주행 영상의 FPR, 완전한 개인정보 비식별화 성공을 검증하지 않았다. ZED2i 실차·기관 인수는 별도이며 자동 운영 활성화하지 않는다.','','WIDER CC-BY-NC-ND-4.0/CCPD MIT 근거와 최초 보관 기한을 유지한다. 원본·개별 예측·표본 ID·가중치는 로컬 비상업 연구에만 보관하며 외부 배포나 개인정보 승인을 주장하지 않는다.','','원래 표준 학습 보고서는 이력으로 보존한다. 본 주 비교는 VAL에서 고정한 선택 옵션을 실제 앱으로 실행한 결과이다.','']
    text='\n'.join(lines);assert_public(text);dest.with_suffix('.md').write_text(text,encoding='utf-8')
    print(json.dumps({'public_results_sha256':sha256(dest),'fresh_both_class_recall95_goal_met':comparisons['fresh']['candidate_both_class_recall95_goal_met'],'selected_option':freeze['selected_option']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','face','license_plate','plans','coverage','publish']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='plans':plans()
    elif args.action=='publish':publish()
    elif args.action=='coverage':record_coverage()
    else:verify(args.action)
