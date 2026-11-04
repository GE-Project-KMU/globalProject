"""Freeze never-used original no-D40 images; no model predictions or training."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,random,sys,tarfile,uuid
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from agent.paths import atomic_json,sha256,workspace_path
from agent.dataset import validate_research_release
from scripts.improve_pothole_data import ARCHIVE_SHA,dhash,group,NAMESPACE
from scripts.prepare_mwpd import historical_identity,isolate_components
OUT=ROOT/'data/training/research-pothole-clean-v6'


def main():
    cv2.setNumThreads(1)
    if OUT.exists():raise ValueError('new_immutable_negative_release_required')
    receipt=json.loads((ROOT/'data/research/rdd2020/source.json').read_text(encoding='utf-8'))
    expiry=datetime.fromisoformat(receipt['expires_at'])
    if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc):raise ValueError('original_research_source_expired')
    archive=ROOT/'data/research/rdd2020/train.tar.gz'
    if sha256(archive)!=ARCHIVE_SHA:raise ValueError('publisher_original_archive_changed')
    history=[]
    for p in sorted((ROOT/'data/training').glob('research-pothole*/release.json')):
        history+=json.loads(p.read_text(encoding='utf-8'))
    oldgroups={r['leakage_group_id'] for r in history}
    for r in history:
        if r.get('original_member','').startswith(('train/Japan/images/','train/Czech/images/')):oldgroups.add(group(r['original_member']))
    choices=defaultdict(list)
    with tarfile.open(archive,'r|gz') as source:
        for member in source:
            if not member.isfile() or not member.name.endswith('.xml') or member.name.split('/')[1] not in ('Japan','Czech'):continue
            raw=source.extractfile(member).read();node=ET.fromstring(raw)
            if any(o.findtext('name')=='D40' for o in node.findall('object')):continue
            country=member.name.split('/')[1];name=node.findtext('filename') or Path(member.name).with_suffix('.jpg').name
            if not name.lower().endswith('.jpg'):name+='.jpg'
            name=f'train/{country}/images/{name}';token=group(name)
            if token not in oldgroups:choices[token].append((name,raw))
    selected={};rng=random.Random(20261106)
    for country in ('Czech','Japan'):
        tokens=sorted(g for g in choices if g.startswith('RDD2020:'+country+':'));rng.shuffle(tokens)
        for token in tokens[:180]:
            name,annotation=rng.choice(sorted(choices[token]));selected[name]=(country,token,annotation)
    availability=dict(Counter(value[0] for value in selected.values()))
    print(json.dumps({'metadata_only_available_unseen_negative_groups':availability,'model_predictions':0}),flush=True)
    if len(selected)<30:raise ValueError('insufficient_unseen_negative_groups')
    evidence=ROOT/'data/research/rdd2020/license-evidence.json'
    # Temporary byte buffers remain in memory. Only accepted TEST rows are written.
    pending=[];buffers={}
    with tarfile.open(archive,'r|gz') as source:
        for member in source:
            if member.name not in selected:continue
            raw=source.extractfile(member).read();image=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
            if image is None:raise ValueError('negative_source_decode_failed')
            country,token,annotation=selected[member.name];identifier=str(uuid.uuid5(NAMESPACE,'fresh-negative-v6:'+member.name))
            import hashlib
            digest=hashlib.sha256(raw).hexdigest();visual=dhash(image)
            pending.append({'sample_id':identifier,'source_id':str(uuid.uuid5(NAMESPACE,'source:'+digest)),'source_hash':digest,'leakage_group_id':token,'split':'test','country':country,'original_member':member.name,
                'path':(OUT/'source'/(identifier+'.jpg')).relative_to(ROOT).as_posix(),'sha256':digest,'perceptual_dhash':f'{visual:016x}','_flip_dhash':dhash(cv2.flip(image,1)),
                'annotation_version':'RDD2020-original-VOC-literal-no-D40-fresh-negative-v1','annotation_sha256':hashlib.sha256(annotation).hexdigest(),
                'privacy_status':'restricted_research','qc_status':'publisher_annotations_machine_checked','release_scope':'offline_research','human_privacy_reviewed':False,'not_for_deployment':True,'project_acceptance_eligible':False,'independent_route_verified':False,'detection_labels_verified':True,
                'allowed_use':'Private noncommercial CC-BY-NC-3.0 offline research; publisher no-D40 annotations; no human privacy approval',
                'expires_at':receipt['expires_at'],'source_expires_at':receipt['expires_at'],'source_url':'https://data.mendeley.com/datasets/5ty2wb6gvg/1','license_evidence_path':evidence.relative_to(ROOT).as_posix(),'license_evidence_sha256':sha256(evidence),'objects':[]})
            buffers[identifier]=raw
    candidates,removed=isolate_components(pending,historical_identity(history))
    final=[]
    for country in ('Czech','Japan'):
        cohort=sorted((r for r in candidates if r['country']==country),key=lambda r:r['sample_id'])[:100]
        # Select eligible countries by source availability only, never scores.
        # The initial 100-image total target proved unavailable before any
        # predictions; preserve >=30 genuinely unused groups rather than reusing.
        if len(cohort)<30:continue
        final+=cohort
    if len(final)<30:raise ValueError('insufficient_novel_negative_cohort')
    (OUT/'source').mkdir(parents=True)
    for row in final:workspace_path(row['path']).write_bytes(buffers[row['sample_id']])
    check=validate_research_release(final)
    if not check['valid']:raise ValueError(str(check['errors'][:3]))
    atomic_json(OUT/'release.json',final)
    atomic_json(ROOT/'docs/reports/POTHOLE-FRESH-NEGATIVE-DATA.json',{'format':'fresh-public-no-D40-evaluation-cohort-v1','source_url':'https://data.mendeley.com/datasets/5ty2wb6gvg/1','release_sha256':sha256(OUT/'release.json'),'samples':len(final),'countries':dict(Counter(r['country'] for r in final)),'available_unseen_source_groups':availability,'acquisition_plan_revision':'Initial >=100 total groups unavailable; before any model prediction allow countries with >=30 safe groups, cap100 per country. No reusing old evaluated groups.','one_candidate_per_unseen_filename20_block':True,'old_hash_and_connected_visual_filter':True,'removed':dict(removed),'all_objects_empty_publisher_no_D40':True,'training_or_selection_use':False,'model_predictions_performed':False,'source_expiry':receipt['expires_at'],'physical_route_independence_verified':False,'limits':'Fresh source-image/block proxies only; no verified independent drive, exhaustive human reannotation, FP/km, or actual-route acceptance. No-D40 images may contain other road defects.'})
    print(json.dumps({'fresh_negative_samples':len(final),'countries':dict(Counter(r['country'] for r in final)),'removed':dict(removed),'test_predictions':0}),flush=True)


if __name__=='__main__':main()
