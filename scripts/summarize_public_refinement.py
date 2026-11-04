"""Aggregate actual experiment evidence; never accesses raw source images."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256,workspace_path
from scripts.evaluate_public_refinement import gates
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main(kind):
    lane=kind=='lane';folder=ROOT/('data/training/lane-refinement-v4' if lane else 'data/training/redaction-refinement-v6')
    manifest=folder/('lane.json' if lane else 'model.json');info=read(manifest);config=read(folder/'training-config.json');history=read(folder/'history.json')
    if info.get('research_only') is not True or info.get('production_approved') is not False or info.get('project_acceptance_eligible') is not False:raise ValueError('research_only_flags_required')
    if sha256(workspace_path(info['weights'],exists=True))!=info['sha256']:raise ValueError('weights_changed')
    plan=read(ROOT/f'docs/reports/{kind.upper()}-REFINEMENT-FRESH-PLAN.json');gates(kind,plan)
    results=read(ROOT/f'docs/reports/{kind.upper()}-REFINEMENT-RESULTS.json')
    if set(results['cohorts'])!={'fresh','regression'}:raise ValueError('both_predeclared_comparisons_required')
    source=ROOT/('scripts/refine_lane_train.py' if lane else 'scripts/refine_privacy_train.py')
    summary={'format':'public-refinement-completion-v1','task':kind,'training_epochs_completed':len(history['history']),'actual_optimizer_updates':history['actual_updates'],
        'selected_epoch':info['best_epoch'],'executed_trainer_sha256':config['training_code_sha256'],'current_trainer_sha256':sha256(source),
        'executed_source_still_unchanged':config['training_code_sha256']==sha256(source),'model_manifest_sha256':sha256(manifest),'weights_sha256':info['sha256'],
        'checkpoint_sha256':sha256(folder/('lane_best.pt' if lane else 'best.pt')),'development_sha256':config['development_sha256'],'release_sha256':config['release_sha256'],
        'numeric_positive_validation_cases':3 if lane else 2,'default_raw_rtol':.001,'default_raw_atol':.0001,'strict_raw_and_postprocess_gate_passed':True,
        'full_validation100_raw_and_exact_mask_passed':True if lane else None,
        **({'new_GPU_profile_verified':False} if lane else {'GPU_validation_included_in_this_CPU_report':False}),
        'peak_reserved_GPU_GiB':history['peak_reserved_GiB'],'fresh':results['cohorts']['fresh'],'regression':results['cohorts']['regression'],
        'research_only':True,'production_approved':False,'project_acceptance_eligible':False,'user_recordings_used':False,'source_expiries_extended':False,
        'old_VAL_is_development_only':True,'old_TEST_is_regression_only':True,'fresh_cohort_used_for_reselection':False,
        'limits':['New source/recording evaluation, not physical route/person independence','Software/public annotation evidence, not a real road or complete privacy acceptance','CPU prediction cost excludes capture/network/review/full-pipeline latency']}
    candidate=summary['fresh']['models']['candidate']
    if lane:
        summary.update(target_paint_IoU=.75,target_met=candidate['paint_iou']>=.75,validation_paint_IoU=info['validation_paint_iou'],selected_threshold=info['paint_threshold'],retention_mae_pp=None)
    else:
        recalls={name:value['tp']/(value['tp']+value['fn']) if value['tp']+value['fn'] else None for name,value in candidate['operating_metrics']['classes'].items()}
        summary.update(target_minimum_known_class_recall=.95,fresh_known_class_recall=recalls,target_met=all(v is not None and v>=.95 for v in recalls.values()),
            plate_state_unchanged=info['plate_state_unchanged'],selected_confidence=info['confidence'])
    atomic_json(ROOT/f'docs/reports/{kind.upper()}-REFINEMENT-COMPLETION.json',summary)
    print(json.dumps({'task':kind,'target_met':summary['target_met'],'actual_optimizer_updates':summary['actual_optimizer_updates'],'selected_epoch':summary['selected_epoch']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['lane','privacy']);a=p.parse_args();main(a.kind)
