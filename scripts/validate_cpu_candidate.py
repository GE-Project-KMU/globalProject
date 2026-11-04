"""Frozen CPU setting vs baseline on pinned public VAL; no speed selection here."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scripts.benchmark_inference_cpu import MANIFESTS,build,measure,parity,sha,scoped

SOURCES={'lane':'data/public/a2d2-improvement-v2/development-manifest.json',
    'pothole':'data/training/research-pothole-rdd2020-v4/release.json',
    'redaction':'data/training/research-redaction-v4/validation-only.json'}


def select(rows,count,required_sha=None):
    rows=sorted(rows,key=lambda row:(row.get('source_id',''),row['sample_id']))
    required=[row for row in rows if row['sha256']==required_sha] if required_sha else []
    if required_sha and len(required)!=1:raise ValueError('canonical_val_case_missing_or_duplicated')
    result=required+[row for row in rows if row not in required][:count-len(required)]
    if len(result)!=count:raise ValueError('insufficient_validation_rows')
    return result


def run(output,production_profile=False):
    output=scoped(output)
    if output.exists():raise ValueError('new_validation_directory_required')
    output.mkdir(parents=True)
    rows={task:[r for r in json.loads(scoped(path).read_text(encoding='utf-8')) if r['split']=='validation'] for task,path in SOURCES.items()}
    case=json.loads(scoped('data/training/pothole-nano-rdd2020-v4/verification-case.json').read_text())
    pothole_sha=sha(case['image'])
    privacy_root=ROOT/'data/training/redaction-inference-options-v4/option-tiled'
    required={label:json.loads((privacy_root/file).read_text())['image_sha256'] for label,file in [('face','verification-final.json'),('license_plate','verification-plate.json')]}
    selected={'lane':select(rows['lane'],100),'pothole':select([r for r in rows['pothole'] if r.get('objects')],20,pothole_sha),
        'redaction':sum([select([r for r in rows['redaction'] if r['evaluation_class']==label],20,required[label]) for label in required],[])}
    plan={'format':'cpu-eight-no-spin-public-val-proof-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'candidate_setting':'eight_no_spin','production_profile':production_profile,'baseline_setting':'baseline','speed_selection_already_completed':True,
        'accuracy_or_threshold_selection_performed':False,'new_test_or_user_images_read':0,
        'adapter_source_sha256':sha(ROOT/'agent/models.py'),'source_manifest_sha256':{task:sha(path) for task,path in SOURCES.items()},
        'model_manifest_sha256':{task:sha(path) for task,path in MANIFESTS.items()},
        'raw_tolerance':{'rtol':.001,'atol':.0001},'cases':{task:[{k:r[k] for k in ('sample_id','path','sha256','expires_at')} for r in values] for task,values in selected.items()}}
    (output/'private-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    baseline=build(MANIFESTS,'baseline')
    if production_profile:
        from agent.inference_runner import create_models
        from scripts.benchmark_inference_cpu import Capture
        candidate=create_models(MANIFESTS,'fast')
        for model in candidate.values():model.net=Capture(model.net)
    else:candidate=build(MANIFESTS,'eight_no_spin')
    results={}
    for task,cases in selected.items():
        checks=[]
        for index,row in enumerate(cases):
            if datetime.fromisoformat(row['expires_at'])<=datetime.now(timezone.utc):raise ValueError('public_validation_source_expired')
            raw_bytes=scoped(row['path']).read_bytes()
            import hashlib
            if hashlib.sha256(raw_bytes).hexdigest()!=row['sha256']:raise ValueError('validation_source_changed')
            image=cv2.imdecode(np.frombuffer(raw_bytes,np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError('public_validation_decode_failed')
            old,_=measure(baseline[task],image);old_raw=baseline[task].net.last.copy()
            new,_=measure(candidate[task],image)
            check=parity(task,old_raw,old,candidate[task].net.last,new)
            checks.append({'image_sha256':row['sha256'],**check})
            if (index+1)%20==0:print(json.dumps({'task':task,'completed':index+1,'total':len(cases)}),flush=True)
        results[task]={'cases':len(checks),'passed':all(r['passed'] for r in checks),'failed_cases':sum(not r['passed'] for r in checks),
            'max_raw_abs_error':max(r['raw']['max_abs_error'] for r in checks),
            'raw_failed_cases':sum(not r['raw']['passed'] for r in checks),'post_failed_cases':sum(not r['post']['passed'] for r in checks)}
        (output/(task+'-private-checks.json')).write_text(json.dumps(checks,indent=2)+'\n')
    assert sha(ROOT/'agent/models.py')==plan['adapter_source_sha256']
    for task,path in MANIFESTS.items():assert sha(path)==plan['model_manifest_sha256'][task]
    aggregate={'format':plan['format'],'candidate_setting':'eight_no_spin','counts':{task:len(v) for task,v in selected.items()},
        'results':results,'passed':all(v['passed'] for v in results.values()),'source_model_unchanged':True,
        'new_test_or_user_images_read':0,'accuracy_or_threshold_selection_performed':False,'raw_tolerance':plan['raw_tolerance'],
        'speed_tuning_on_these_cases':False,'plan_sha256':sha(output/'private-plan.json')}
    (output/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n');print(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');p.add_argument('--production-profile',action='store_true');a=p.parse_args();run(a.output,a.production_profile)
