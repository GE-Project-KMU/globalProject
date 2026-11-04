"""Metadata-only fresh/regression plans before final predictions; no TEST pixels."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256,workspace_path
from scripts.public_refinement_data import freeze
DEST=ROOT/'data/training/refinement-evaluation'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rel(path):return path.relative_to(ROOT).as_posix()
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'data/training/lane-a2d2-supported-ort131-v3/lane.json';baseline_dir=ROOT/'data/training/lane-refinement-baseline-v4';baseline=baseline_dir/'lane.json'
    original=read(parent);provenance=read(workspace_path(original['training_provenance'],exists=True));provenance['models']=[rel(baseline)]
    freeze(baseline_dir/'provenance.json',provenance)
    clone=dict(original,training_provenance=rel(baseline_dir/'provenance.json'),parent_manifest_sha256=sha256(parent),clone_scope='unchanged baseline graph/state/threshold/runtime profile; new verification receipt location only')
    freeze(baseline,clone)
    lane_dev=read(ROOT/'data/public/a2d2-refinement-v3/development.json');privacy_dev=read(ROOT/'data/training/research-redaction-v6/development.json')
    cases={'lane':[], 'privacy':{}}
    for row in sorted((r for r in lane_dev if r['split']=='validation' and r.get('positive_pixels',0)>0),key=lambda r:r['source_id'])[:3]:
        cases['lane'].append({key:row[key] for key in ['source_id','path','sha256','expires_at']}|{'positive_GT_pixels':row['positive_pixels']})
    for kind in ['face','license_plate']:
        row=sorted((r for r in privacy_dev if r['split']=='validation' and r['evaluation_class']==kind and r['objects']),key=lambda r:r['source_id'])[0]
        cases['privacy'][kind]={key:row[key] for key in ['source_id','path','sha256','expires_at']}|{'GT_count':len(row['objects'])}
    freeze(DEST/'positive-validation-cases.json',cases)
    freeze(ROOT/'docs/reports/REFINEMENT-NUMERIC-GATE-PLAN.json',{'created_before_final_predictions':True,'positive_validation_cases_manifest_sha256':sha256(DEST/'positive-validation-cases.json'),
        'lane_predeclared_positive_cases':3,'lane_full_VAL_images':100,'privacy_predeclared_positive_classes':['face','license_plate'],
        'raw_rtol':.001,'raw_atol':.0001,'postprocess_mask_exact_required':True,'test_pixels_for_numeric_gate':0,'new_GPU_support_claimed':False})
    for kind,baseline,candidate,release,fresh,regression in [
        ('LANE',baseline,ROOT/'data/training/lane-refinement-v4/lane.json',ROOT/'data/public/a2d2-refinement-v3/release.json',ROOT/'data/public/a2d2-refinement-v3/fresh-test.json',ROOT/'data/public/a2d2-improvement-v2/quarantined-test/manifest.json'),
        ('PRIVACY',ROOT/'data/training/redaction-inference-options-v4/option-tiled/model.json',ROOT/'data/training/redaction-refinement-v6/model.json',ROOT/'data/training/research-redaction-v6/release.json',ROOT/'data/training/research-redaction-v6/fresh-test.json',ROOT/'data/training/research-redaction-v4/release.json')]:
        for role,benchmark in [('FRESH',fresh),('REGRESSION',regression)]:
            plan={'created_before_final_predictions':True,'models':{'baseline':rel(baseline),'candidate':rel(candidate)},'baseline_manifest_sha256':sha256(baseline),
                'candidate_release_sha256':sha256(release),'benchmark_manifest':rel(benchmark),'benchmark_sha256':sha256(benchmark),
                'role':('new_unseen_recording_or_source_only_no_physical_person_route_independence' if role=='FRESH' else 'previously_examined_regression_only_no_independent_accuracy_claim'),
                'checkpoint_threshold_selection':'development VAL only; no post-TEST tuning','test_used_for_fitting_or_selection':False,'physical_independence_verified':False,
                'required_candidate_positive_VAL_proofs':3 if kind=='LANE' else 2,'raw_rtol':.001,'raw_atol':.0001,'project_acceptance_eligible':False,'production_approved':False}
            freeze(ROOT/f'docs/reports/{kind}-REFINEMENT-{role}-PLAN.json',plan)
    print(json.dumps({'cases_manifest_sha256':sha256(DEST/'positive-validation-cases.json'),'baseline_lane_manifest_sha256':sha256(ROOT/'data/training/lane-refinement-baseline-v4/lane.json'),'TEST_pixels_read':0}))
if __name__=='__main__':main()
