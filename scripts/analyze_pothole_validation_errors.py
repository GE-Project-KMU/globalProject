"""Metadata-only error decomposition of the existing v6 VAL diagnostic."""
import json,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from agent.pipeline import intersection_over_union

def decompose(truth,predictions):
    matched=set();fp=[]
    for p in sorted(predictions,key=lambda x:x['confidence'],reverse=True):
        overlap,index=max(((intersection_over_union(p['bbox_xywh_px'],g['bbox_xywh_px']),i) for i,g in enumerate(truth) if i not in matched),default=(0,None))
        if overlap>=.5:matched.add(index)
        else:fp.append(p)
    def category(value):return 'below_0_1' if value<.1 else '0_1_to_0_5' if value<.5 else 'at_least_0_5'
    fn=Counter(category(max((intersection_over_union(g['bbox_xywh_px'],p['bbox_xywh_px']) for p in predictions),default=0)) for i,g in enumerate(truth) if i not in matched)
    errors=Counter(category(max((intersection_over_union(g['bbox_xywh_px'],p['bbox_xywh_px']) for g in truth),default=0)) for p in fp)
    return {'tp':len(matched),'fn':len(truth)-len(matched),'fp':len(fp),'fn_best_prediction_IoU':dict(fn),'fp_best_GT_IoU':dict(errors)}

def run():
    folder=workspace_path('data/training/pothole-validation-diagnostic-v7',exists=True)
    path=folder/'validation-private-predictions.json';private=json.loads(path.read_text(encoding='utf-8'));base=json.loads((folder/'aggregate.json').read_text(encoding='utf-8'))
    result={domain:Counter() for domain in ['original','mwpd']};fn={d:Counter() for d in result};fp={d:Counter() for d in result}
    if len(private['samples'])!=652:raise ValueError('fixed_validation_only652_required')
    for sample in private['samples']:
        domain=sample['domain'];pred=[p for p in sample['predictions'] if p['class']=='pothole' and p['confidence']>=base['operating_confidence']]
        stats=decompose(sample['truth'],pred);result[domain].update({k:stats[k] for k in ['tp','fn','fp']})
        result[domain].update(images=1,images_without_GT=int(not sample['truth']),FP_on_images_without_GT=len(pred) if not sample['truth'] else 0)
        fn[domain].update(stats['fn_best_prediction_IoU']);fp[domain].update(stats['fp_best_GT_IoU'])
    for domain in result:
        original=base['domains'][domain]['operating']['overall']
        if any(result[domain][k]!=original[k] for k in ['tp','fn','fp']):raise ValueError('decomposition_does_not_match_frozen_diagnostic')
    report={'format':'pothole-v6-VAL-error-decomposition-v1','source_sha256':sha256(Path(__file__)),
            'parent_aggregate_sha256':sha256(folder/'aggregate.json'),'private_predictions_sha256':sha256(path),
            'validation_images':652,'additional_model_inferences':0,'additional_image_reads':0,'test_images':0,'gpu_used':False,
            'operating_confidence':base['operating_confidence'],'source_expires_at':private['expires_at'],
            'domains':{d:{**dict(result[d]),'FN_best_sameclass_prediction_IoU':dict(fn[d]),'FP_best_sameclass_GT_IoU':dict(fp[d])} for d in result},
            'interpretation':{'below_0_1':'No close operating-point prediction/GT; background/confidence failure is possible but cause not proven.',
                              '0_1_to_0_5':'Some box overlap, but below IoU0.5; localization or annotation disagreement is possible, not established.',
                              'at_least_0_5':'One-to-one matching collision or duplicate prediction/annotation is possible; human audit not performed.'},
            'limits':['Metadata-only existing validation diagnostic, not independent evaluation.','Publisher annotations may be incomplete; background false positives are label-relative.','No operating threshold or weights were changed.']}
    atomic_json(folder/'error-decomposition.json',report);atomic_json(workspace_path('docs/reports/POTHOLE-V6-VALIDATION-ERROR-DECOMPOSITION.json'),report)
    print(json.dumps(report))

if __name__=='__main__':run()
