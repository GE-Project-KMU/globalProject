"""One final paired actual-app test: frozen MWPD positives plus unseen no-D40 negatives."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release
from scripts.compare_detector_models import compare,score
BASE='data/training/pothole-nano-rdd2020-v4/model.json'
DEST=ROOT/'data/training/pothole-mwpd-v6'
PUBLIC=ROOT/'docs/reports/POTHOLE-MWPD-FINAL-PLAN.json'


def read(path):return json.loads(workspace_path(path,exists=True).read_text(encoding='utf-8'))


def freeze():
    benchmark=DEST/'final-benchmark.json'
    if benchmark.exists() or PUBLIC.exists():raise ValueError('immutable_final_benchmark_already_frozen')
    positive=[r for r in read('data/training/research-pothole-mwpd-v6/release.json') if r['split']=='test']
    negative=read('data/training/research-pothole-clean-v6/release.json')
    if len(positive)!=97 or len(negative)<30 or any(r['objects'] for r in negative):raise ValueError('fixed_positive_and_negative_cohorts_required')
    rows=positive+negative
    if not validate_research_release(rows)['valid']:raise ValueError('invalid_frozen_test_cohorts')
    development=[r for r in read(DEST/'release.json') if r['split']!='test']
    if {r['sha256'] for r in negative}&{r['sha256'] for r in development}:raise ValueError('negative_development_leakage')
    atomic_json(benchmark,rows)
    atomic_json(PUBLIC,{'format':'pothole-mwpd-final-plan-v1','created_before_final_predictions':True,
        'models':{'baseline':BASE,'candidate':'data/training/pothole-mwpd-v6/model.json'},
        'baseline_manifest_sha256':sha256(ROOT/BASE),'candidate_release_sha256':sha256(DEST/'release.json'),
        'benchmark_manifest':benchmark.relative_to(ROOT).as_posix(),'benchmark_sha256':sha256(benchmark),
        'role':'new_unused_source_image_and_filename_block_cohorts_not_verified_physical_routes',
        'positive_images':len(positive),'negative_images':len(negative),'negative_source_country':'Czech',
        'metrics':['combined AP50 at .01 floor','MWPD positive-only AP50','frozen .1 operating precision/recall','no-D40 negative FP/image and any-FP fraction'],
        'candidate_selection':'frozen 12epoch mean(old VAL,new VAL) with old VAL<=3pp absolute drop; never final TEST',
        'test_used_for_selection':False,'physical_route_independence_verified':False,'production_approved':False,
        'script_sha256':sha256(Path(__file__))})
    print(json.dumps({'frozen_positive':len(positive),'frozen_negative':len(negative),'model_predictions':0}),flush=True)


def run():
    plan=read(PUBLIC)
    if plan['script_sha256']!=sha256(Path(__file__)):raise ValueError('final_evaluator_changed_since_freeze')
    snapshot=read(DEST/'selection-freeze.json');candidate=read(DEST/'model.json')
    if sha256(DEST/'model.json')!=snapshot['manifest_sha256'] or candidate['sha256']!=snapshot['onnx_sha256'] or sha256(DEST/'best.pt')!=snapshot['checkpoint_sha256']:
        raise ValueError('candidate_changed_after_validation_selection')
    output=DEST/'final-comparison.json'
    result=compare(PUBLIC,output)
    details=read(output.with_name(output.stem+'-per-image.json'));rows=read(DEST/'final-benchmark.json')
    positive_ids={r['sample_id'] for r in rows if r['objects']};negative_ids={r['sample_id'] for r in rows if not r['objects']}
    truth=details['truth'];metrics={}
    for name,predictions in details['predictions'].items():
        operating=result['results'][name]['operating_confidence']
        false=[p for p in predictions if p['sample_id'] in negative_ids and p['confidence']>=operating]
        metrics[name]={'combined':result['results'][name],
            'positive_only_ap50':score([t for t in truth if t['sample_id'] in positive_ids],[p for p in predictions if p['sample_id'] in positive_ids],['pothole'],.01)['map50'],
            'negative_fp_boxes':len(false),'negative_fp_per_image':len(false)/len(negative_ids),
            'negative_images_with_fp':len({p['sample_id'] for p in false}),
            'negative_any_fp_fraction':len({p['sample_id'] for p in false})/len(negative_ids)}
    report={'format':'pothole-mwpd-final-results-v1','plan_sha256':sha256(PUBLIC),'comparison_report_sha256':sha256(output),
            'samples':len(rows),'positive_images':len(positive_ids),'negative_images':len(negative_ids),'results':metrics,
            'combined_ap50_change':result['map50_change'],'training_epochs':12,'selected_epoch':snapshot['selected_epoch'],
            'optimizer_updates':snapshot['optimizer_updates'],'validation':{'baseline':snapshot['baseline_validation'],'selected':snapshot['selected_validation']},
            'test_used_for_model_or_threshold_selection':False,'physical_route_independence_verified':False,'project_acceptance_passed':False,
            'source_licenses':['CC-BY-4.0','CC-BY-NC-3.0'],'research_only':True,'weights_uploaded':False,'user_recordings_used':False,
            'limits':['Small fixed source-image cohorts, not domestic ZED driving acceptance','No-D40 publisher labels were not exhaustively human-reannotated','Negative cohort has48 Czech images; FP/image is not FP/km','Original-domain validation noninferiority gate is development-only; old validation is not a new final test','No post-TEST model, threshold or sampling changes']}
    atomic_json(ROOT/'docs/reports/POTHOLE-MWPD-RESULTS.json',report)
    print(json.dumps({'combined_ap50':{k:v['combined']['ap_metrics']['map50'] for k,v in metrics.items()},'change':result['map50_change']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);args=p.parse_args()
    (freeze if args.action=='freeze' else run)()
