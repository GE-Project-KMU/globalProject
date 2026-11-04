"""Frozen CPU-only TRAIN mining. No VAL/TEST inference or quota alteration."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from scripts.analyze_pothole_validation import match

MODEL='data/training/pothole-mwpd-v6/model.json'
BASE_RELEASE='data/training/pothole-mwpd-v6/release.json'
SOURCES={'RDD':'data/training/research-pothole-rdd2020-v4/release.json','MWPD':'data/training/research-pothole-mwpd-v6/release.json'}
OUTPUT='data/training/pothole-train-hardness-v8'
COUNTS={'RDD_positive':2676,'RDD_negative':2676,'MWPD_positive':2188}
DEPENDENCIES=['agent/models.py','agent/pipeline.py','scripts/analyze_pothole_validation.py']

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def aware(value):
    date=datetime.fromisoformat(value.replace('Z','+00:00'))
    if date.tzinfo is None:raise ValueError('aware_source_expiry_required')
    return date.astimezone(timezone.utc)

def hardness(gt,fn,fp):
    if any(type(x) is not int or x<0 for x in (gt,fn,fp)) or fn>gt:raise ValueError('valid_matching_counts_required')
    return 1.+2.*min(1.,(fn+fp)/(gt+3) if gt else fp/3)

def membership(rows):
    if any(r['split']!='train' or r['hardness_stratum'] not in COUNTS for r in rows):raise ValueError('TRAIN_only_membership_required')
    records=sorted((r['sample_id'],r['sha256'],r['source_hash'],r['leakage_group_id'],r['hardness_stratum']) for r in rows)
    return hashlib.sha256(json.dumps(records,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def assert_train_separation(rows,protected):
    if any(r['split']!='train' for r in rows):raise ValueError('TRAIN_only_rows_required')
    for key in ['sample_id','sha256','source_hash','leakage_group_id']:
        if {r[key] for r in rows}&{r[key] for r in protected}:raise ValueError('train_validation_or_test_identity_overlap')

def training_rows():
    rows=[];proof={};protected=[]
    for domain,reference in SOURCES.items():
        path=workspace_path(reference,exists=True);proof[domain]=sha256(path);corpus=read(path)
        protected.extend(r for r in corpus if r['split']!='train')
        for row in corpus:
            if row['split']!='train':continue
            if any(o.get('class')!='pothole' for o in row['objects']):raise ValueError('single_pothole_class_required')
            if row.get('human_privacy_reviewed') is not False or row.get('not_for_deployment') is not True or row.get('release_scope')!='offline_research':raise ValueError('restricted_research_training_required')
            rows.append(dict(row,hardness_stratum=domain+('_positive' if row['objects'] else '_negative')))
    if Counter(r['hardness_stratum'] for r in rows)!=COUNTS or len({r['sample_id'] for r in rows})!=7540:raise ValueError('fixed_original_TRAIN7540_required')
    assert_train_separation(rows,protected)
    baseline=workspace_path(BASE_RELEASE,exists=True);combined=[r for r in read(baseline) if r['split']=='train']
    if sorted((r['sample_id'],r['sha256']) for r in combined)!=sorted((r['sample_id'],r['sha256']) for r in rows):raise ValueError('baseline_train_membership_changed')
    return rows,{**proof,'baseline_combined':sha256(baseline)}

def prepare():
    output=workspace_path(OUTPUT)
    if output.exists():raise ValueError('new_hardness_output_required')
    rows,releases=training_rows();manifest=workspace_path(MODEL,exists=True);artifact=read(manifest)
    if artifact['task']!='pothole' or artifact['classes']!=['pothole'] or artifact['input_size']!=[640,640] or artifact['confidence']!=.1:raise ValueError('frozen_v6_640_confidence01_required')
    if artifact.get('production_approved') is not False or artifact.get('research_only') is not True:raise ValueError('research_model_required')
    weights=workspace_path(artifact['weights'],exists=True)
    if sha256(weights)!=artifact['sha256']:raise ValueError('frozen_weights_mismatch')
    if artifact['dataset_release']!='manifest-sha256:'+releases['baseline_combined']:raise ValueError('baseline_source_release_mismatch')
    expiry=min(aware(r['expires_at']) for r in rows)
    if expiry<=datetime.now(timezone.utc):raise ValueError('training_source_expired')
    plan={'format':'pothole-TRAIN7540-hardness-v8','source_sha256':sha256(Path(__file__)),
          'dependencies_sha256':{p:sha256(ROOT/p) for p in DEPENDENCIES},'manifest_sha256':sha256(manifest),'weights_sha256':sha256(weights),
          'source_release_sha256':releases,'training_membership_sha256':membership(rows),'train_images':7540,'strata':COUNTS,
          'matching':'confidence>=0.1; class pothole; descending-confidence greedy one-to-one IoU>=0.5; same GT order as release',
          'positive_weight':'1+2*min(1,(FN+FP)/(GT+3))','negative_weight':'1+2*min(1,FP/3)','weight_min':1,'weight_max':3,
          'source_quotas_owned_by_trainer':{'RDD_positive':.25,'RDD_negative':.25,'MWPD_positive':.5},
          'weights_do_not_change_source_quota':True,'selection_or_confidence_tuning':False,'val_images_inferred':0,'test_images_inferred':0,
          'US100_used':False,'source_expires_at':expiry.isoformat(),'expires_at':expiry.isoformat(),'cpu_threads':2,'cv2_threads':1,'allow_spinning':False,
          'runtime':'CPUExecutionProvider','gpu_used':False,'raw_photos_copied':False,'human_privacy_reviewed':False,'not_for_deployment':True}
    output.mkdir();atomic_json(output/'plan.json',plan)
    atomic_json(workspace_path('docs/reports/POTHOLE-TRAIN-HARDNESS-V8-PLAN.json'),plan)
    print(json.dumps({'status':'TRAIN_mining_plan_frozen','plan_sha256':sha256(output/'plan.json'),'source_sha256':plan['source_sha256'],'train_images':7540,'strata':COUNTS,'actual_mining_started':False}))

def mine():
    import cv2,numpy as np
    from agent.models import OnnxModel
    cv2.setNumThreads(1);output=workspace_path(OUTPUT,exists=True);plan=read(output/'plan.json');plan_sha=sha256(output/'plan.json')
    if sha256(Path(__file__))!=plan['source_sha256'] or any(sha256(ROOT/p)!=digest for p,digest in plan['dependencies_sha256'].items()):raise ValueError('frozen_mining_source_changed')
    rows,releases=training_rows()
    if releases!=plan['source_release_sha256'] or membership(rows)!=plan['training_membership_sha256']:raise ValueError('frozen_TRAIN_membership_changed')
    expiry=aware(plan['source_expires_at'])
    if expiry<=datetime.now(timezone.utc):raise ValueError('original_training_source_expired')
    manifest=workspace_path(MODEL,exists=True)
    if sha256(manifest)!=plan['manifest_sha256']:raise ValueError('frozen_baseline_manifest_changed')
    artifact=read(manifest);weights=workspace_path(artifact['weights'],exists=True)
    if sha256(weights)!=plan['weights_sha256']:raise ValueError('frozen_baseline_weights_changed')
    receipt=output/'run-receipt.json'
    with receipt.open('x',encoding='utf-8') as stream:json.dump({'status':'running','plan_sha256':plan_sha,'started_at':datetime.now(timezone.utc).isoformat(),'expires_at':expiry.isoformat()},stream)
    model=None;results=[];predictions=[];times=[]
    try:
        model=OnnxModel(manifest,cpu_threads=2,allow_spinning=False)
        if model.net.session.get_providers()!=['CPUExecutionProvider']:raise ValueError('CPU_execution_only_required')
        for row in sorted(rows,key=lambda r:r['sample_id']):
            if datetime.now(timezone.utc)>=expiry:raise ValueError('source_expired_during_TRAIN_mining')
            image_path=workspace_path(row['path'],exists=True)
            if sha256(image_path)!=row['sha256']:raise ValueError('TRAIN_image_changed')
            image=cv2.imread(str(image_path))
            if image is None:raise ValueError('TRAIN_image_decode_failed')
            start=time.perf_counter();pred=[p for p in model.predict(image) if p['class']=='pothole' and p['confidence']>=.1];times.append(time.perf_counter()-start)
            truth=row['objects'];found,fp=match(truth,pred);gt=len(truth);fn=gt-len(found)
            results.append({'sample_id':row['sample_id'],'sha256':row['sha256'],'stratum':row['hardness_stratum'],'weight':hardness(gt,fn,fp),'gt_count':gt,'tp':len(found),'fp':fp,'fn':fn})
            predictions.append({'sample_id':row['sample_id'],'source_hash':row['sha256'],'predictions':pred,'expires_at':row['expires_at']})
            if len(results)%500==0:print(json.dumps({'TRAIN_images_completed':len(results),'VAL_images':0,'TEST_images':0}),flush=True)
        if sha256(manifest)!=plan['manifest_sha256'] or sha256(weights)!=plan['weights_sha256']:raise ValueError('baseline_model_changed_during_mining')
        _,current=training_rows()
        if current!=releases:raise ValueError('source_release_changed_during_mining')
        summaries={}
        for stratum in COUNTS:
            subset=[r for r in results if r['stratum']==stratum];values=[r['weight'] for r in subset]
            summaries[stratum]={'images':len(subset),'weight_min':min(values),'weight_max':max(values),'weight_mean':float(np.mean(values)),
                'weight_median':float(np.median(values)),'weight_p95':float(np.percentile(values,95)),
                'weight_equal1':sum(v==1 for v in values),'weight_between1and3':sum(1<v<3 for v in values),'weight_equal3':sum(v==3 for v in values),
                'gt':sum(r['gt_count'] for r in subset),'tp':sum(r['tp'] for r in subset),'fp':sum(r['fp'] for r in subset),'fn':sum(r['fn'] for r in subset)}
        header={'format':plan['format'],'plan_sha256':plan_sha,'source_release_sha256':releases,'training_membership_sha256':plan['training_membership_sha256'],
                'manifest_sha256':plan['manifest_sha256'],'weights_sha256':plan['weights_sha256'],'expires_at':expiry.isoformat(),'source_expires_at':expiry.isoformat(),
                'train_only':True,'val_images':0,'test_images':0,'US100_used':False,'source_quotas_changed':False}
        atomic_json(output/'predictions-private.json',{'expires_at':expiry.isoformat(),'samples':predictions})
        atomic_json(output/'training-weights.json',{**header,'samples':results})
        aggregate={**header,'status':'completed','source_sha256':plan['source_sha256'],'train_images':7540,'strata':summaries,
                   'training_weights_file_sha256':sha256(output/'training-weights.json'),'gpu_used':False,'CPU_intra_op_threads':2,
                   'latency_seconds':{'median':float(np.median(times)),'p95':float(np.percentile(times,95))},
                   'limits':['TRAIN hardness only; not validation or independent accuracy.','Individual weights are bounded1..3; trainer must enforce source quotas separately.','Single-model CPU latency under shared training host load; not field/pipeline FPS.','Publisher-label relative FP/FN; no human privacy or deployment approval.']}
        atomic_json(output/'aggregate.json',aggregate);atomic_json(workspace_path('docs/reports/POTHOLE-TRAIN-HARDNESS-V8-RESULTS.json'),aggregate)
        atomic_json(receipt,{'status':'completed','plan_sha256':plan_sha,'finished_at':datetime.now(timezone.utc).isoformat(),'expires_at':expiry.isoformat(),'samples':7540})
        print(json.dumps(aggregate));return aggregate
    except BaseException as error:
        atomic_json(receipt,{'status':'failed','error_type':type(error).__name__,'plan_sha256':plan_sha,'expires_at':expiry.isoformat(),'successful_weights_not_claimed':True})
        raise
    finally:
        if model is not None:model.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','mine']);args=parser.parse_args()
    try:(prepare if args.action=='plan' else mine)()
    except BaseException as error:
        print(json.dumps({'status':'TRAIN_mining_stopped','error_type':type(error).__name__}),file=sys.stderr);sys.exit(1)
