"""CPU-only v6 validation diagnostic; no TEST images or predictions are read."""
from __future__ import annotations
import json,math,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from agent.pipeline import intersection_over_union

MODEL='data/training/pothole-mwpd-v6/model.json'
SOURCES={'original':'data/training/research-pothole-rdd2020-v4/release.json','mwpd':'data/training/research-pothole-mwpd-v6/release.json'}
DEST='data/training/pothole-validation-diagnostic-v7'
BINS=('<16','16_to_32','32_to_64','64_plus')

def size_bin(box,width,height):
    scaled=math.sqrt(box[2]*box[3])*min(640/width,640/height)
    return BINS[0] if scaled<16 else BINS[1] if scaled<32 else BINS[2] if scaled<64 else BINS[3]

def match(truth,predictions):
    found=set();fp=0
    for p in sorted(predictions,key=lambda x:x['confidence'],reverse=True):
        overlap,index=max(((intersection_over_union(p['bbox_xywh_px'],g['bbox_xywh_px']),i) for i,g in enumerate(truth) if i not in found),default=(0,None))
        if overlap>=.5:found.add(index)
        else:fp+=1
    return found,fp

def run():
    import cv2,numpy as np
    from agent.models import OnnxModel
    from agent.dataset import validate_research_release
    cv2.setNumThreads(1);output=workspace_path(DEST)
    if output.exists():raise ValueError('fresh_validation_diagnostic_output_required')
    rows=[];release_proof={}
    for domain,name in SOURCES.items():
        path=workspace_path(name,exists=True);release_proof[domain]=sha256(path)
        rows.extend(dict(r,diagnostic_domain=domain) for r in json.loads(path.read_text(encoding='utf-8')) if r['split']=='validation')
    if Counter(r['diagnostic_domain'] for r in rows)!={'original':402,'mwpd':250}:raise ValueError('exact_652_validation_only_required')
    if not validate_research_release(rows)['valid']:raise ValueError('expired_or_invalid_validation_source')
    expiry=min(datetime.fromisoformat(r['expires_at'].replace('Z','+00:00')) for r in rows)
    manifest=workspace_path(MODEL,exists=True);manifest_sha=sha256(manifest)
    model=OnnxModel(manifest,cpu_threads=2,allow_spinning=False)
    try:
        artifact=dict(model.manifest);confidence=artifact.get('confidence',.35)
        if artifact['task']!='pothole' or artifact['classes']!=['pothole'] or artifact['input_size']!=[640,640]:raise ValueError('fixed_pothole_v6_640_required')
        if model.net.session.get_providers()!=['CPUExecutionProvider']:raise ValueError('CPU_execution_only_required')
        weights=workspace_path(artifact['weights'],exists=True);weights_sha=sha256(weights)
        plan={'format':'pothole-v6-validation-diagnostic-v1','scope':'existing validation only; not a fresh benchmark',
              'source_sha256':sha256(Path(__file__)),'manifest_sha256':manifest_sha,'weights_sha256':weights_sha,
              'release_sha256':release_proof,'images':652,'source_expires_at':expiry.isoformat(),'input_size':[640,640],
              'operating_confidence':confidence,'supplemental_confidence_floor':.01,'IoU':.5,
              'bins':'sqrt(GT box area) after aspect-preserving640 letterbox: <16,16..<32,32..<64,>=64',
              'runtime':'CPUExecutionProvider','cpu_threads':2,'cv2_threads':1,'gpu_used':False,'test_images':0,'no_model_or_threshold_modification':True}
        output.mkdir();atomic_json(output/'plan.json',plan)
        model.manifest=dict(artifact,confidence=.01)
        accum={domain:{name:{b:Counter() for b in BINS} for name in ['operating','floor_001']} for domain in SOURCES}
        totals={domain:{name:Counter() for name in ['operating','floor_001']} for domain in SOURCES}
        private=[];times=[]
        for row in sorted(rows,key=lambda r:r['sample_id']):
            if datetime.now(timezone.utc)>=expiry:raise ValueError('source_expired_during_validation_diagnostic')
            path=workspace_path(row['path'],exists=True)
            if sha256(path)!=row['sha256']:raise ValueError('validation_image_changed')
            image=cv2.imread(str(path))
            if image is None:raise ValueError('validation_image_decode_failed')
            truth=[o for o in row['objects'] if o['class']=='pothole'];start=time.perf_counter();pred=model.predict(image);times.append(time.perf_counter()-start)
            domain=row['diagnostic_domain']
            for name,threshold in [('operating',confidence),('floor_001',.01)]:
                found,fp=match(truth,[p for p in pred if p['class']=='pothole' and p['confidence']>=threshold])
                totals[domain][name].update(gt=len(truth),tp=len(found),fn=len(truth)-len(found),fp=fp,images=1)
                for i,obj in enumerate(truth):accum[domain][name][size_bin(obj['bbox_xywh_px'],image.shape[1],image.shape[0])].update(gt=1,tp=int(i in found),fn=int(i not in found))
            private.append({'sample_id':row['sample_id'],'source_hash':row['sha256'],'domain':domain,'truth':truth,'predictions':pred,'expires_at':row['expires_at']})
            if len(private)%100==0:print(json.dumps({'validation_images_completed':len(private),'test_images':0}),flush=True)
        if sha256(manifest)!=manifest_sha or sha256(weights)!=weights_sha:raise ValueError('frozen_model_changed')
        if any(sha256(workspace_path(SOURCES[k],exists=True))!=digest for k,digest in release_proof.items()):raise ValueError('validation_release_changed')
        def metrics(c):
            result={'gt':c['gt'],'tp':c['tp'],'fn':c['fn'],**dict(c)};result['recall']=c['tp']/c['gt'] if c['gt'] else None
            if 'fp' in c:result['precision']=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else None
            return result
        result={'format':plan['format'],'scope':plan['scope'],'plan_sha256':sha256(output/'plan.json'),'manifest_sha256':manifest_sha,
                'weights_sha256':weights_sha,'images':652,'test_images':0,'gpu_used':False,'operating_confidence':confidence,
                'source_expires_at':expiry.isoformat(),'domains':{domain:{name:{'overall':metrics(totals[domain][name]),'size_bins':{b:metrics(c) for b,c in bins.items()}} for name,bins in points.items()} for domain,points in accum.items()},
                'inference_seconds':{'median':float(np.median(times)),'p95':float(np.percentile(times,95))},
                'limits':['Existing validation diagnostic for next research recipe; no independent acceptance claim.','Size refers to square-root GT area after640 aspect-preserving scale, not object real-world dimensions.','CPU timing excludes decode and runs under shared training host load; not pipeline FPS.','Publisher annotations only; no human privacy approval.']}
        atomic_json(output/'validation-private-predictions.json',{'expires_at':expiry.isoformat(),'samples':private})
        atomic_json(output/'aggregate.json',result);atomic_json(workspace_path('docs/reports/POTHOLE-V6-VALIDATION-SIZE-DIAGNOSTIC.json'),result)
        print(json.dumps(result));return result
    finally:model.close()

if __name__=='__main__':run()
