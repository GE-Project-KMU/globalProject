"""Publisher-hash-verified MWPD acquisition and conservative grouped research split."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import httpx
from agent.paths import workspace_path,atomic_json,sha256
from agent.dataset import validate_research_release
from scripts.improve_pothole_data import dhash

SOURCE='https://data.mendeley.com/datasets/s5hx9n2jc3/2'
API='https://data.mendeley.com/public-api/datasets/s5hx9n2jc3'
RAW='data/research/mwpd-v6'
RELEASE='data/training/research-pothole-mwpd-v6'
NAMESPACE=uuid.UUID('3c444c1d-42d3-4353-a708-71793ac7edd0')


def source_group(name):
    # All Roboflow augmentations with the same pre-.rf. name stay together.
    if not re.fullmatch(r'[A-Za-z0-9_-]+\.rf\.[a-f0-9]{32}\.(jpg|txt)',name):
        raise ValueError('unexpected_publisher_filename')
    return name.split('.rf.')[0]


def select_pairs(metadata):
    by={(r['split'],r['kind']):r['files'] for r in metadata}
    groups={s:{source_group(f['filename']) for f in by[s,'images']} for s in ('train','valid','test')}
    result=[];removed=Counter()
    for split in ('test','valid','train'):
        protected=groups['test'] if split=='valid' else groups['test']|groups['valid'] if split=='train' else set()
        labels={Path(f['filename']).stem:f for f in by[split,'labels']}
        for image in by[split,'images']:
            group=source_group(image['filename'])
            if group in protected:removed['publisher_cross_split_filename_group']+=1;continue
            if Path(image['filename']).stem not in labels:raise ValueError('publisher_image_label_pair_missing')
            result.append({'split':'validation' if split=='valid' else split,'group':group,'image':image,'label':labels[Path(image['filename']).stem]})
    return result,dict(removed)


def local_file(folder,entry):
    uuid.UUID(entry['id'])
    return workspace_path(folder/'files'/(entry['id']+Path(entry['filename']).suffix))


def acquire(output=RAW):
    folder=workspace_path(output)
    if not folder.is_relative_to(ROOT/'data/research') or folder==ROOT/'data/research':raise ValueError('private_research_directory_required')
    folder.mkdir(parents=True,exist_ok=True)
    receipt_path=folder/'source.json'
    with httpx.Client(timeout=45,follow_redirects=True,limits=httpx.Limits(max_connections=16)) as client:
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
            if datetime.fromisoformat(receipt['expires_at'])<=datetime.now(timezone.utc):raise ValueError('source_expired_no_redownload_extension')
            metadata=json.loads((folder/'publisher-files.json').read_text(encoding='utf-8'))
            if sha256(folder/'publisher-files.json')!=receipt['publisher_metadata_sha256']:raise ValueError('publisher_metadata_changed')
        else:
            page=client.get(SOURCE);page.raise_for_status()
            snapshot=json.JSONDecoder().raw_decode(page.text.split('window.INITIAL_STATE = ',1)[1])[0]['dataset']['snapshot']
            if snapshot['version']!=2 or snapshot['licence']['short_name']!='CC BY 4.0':raise ValueError('publisher_license_or_version_changed')
            (folder/'publisher-page.html').write_text(page.text,encoding='utf-8')
            atomic_json(folder/'license-evidence.json',{'research_allowed':True,'license_id':'CC-BY-4.0','license_url':'https://creativecommons.org/licenses/by/4.0/','source_url':SOURCE,'doi':'10.17632/s5hx9n2jc3.2','human_privacy_reviewed':False,'not_for_deployment':True})
            response=client.get(API+'/folders/2');response.raise_for_status();folders=response.json();indexed={f['id']:f for f in folders}
            metadata=[]
            for f in folders:
                if f['name'] not in ('images','labels'):continue
                rows=[];start=0
                while True:
                    r=client.get(API+f"/files?folder_id={f['id']}&version=2&$start={start}&$limit=1000",headers={'Accept':'application/vnd.mendeley-public-dataset.1+json'});r.raise_for_status();batch=r.json();rows+=batch
                    if len(batch)<1000:break
                    start+=1000
                metadata.append({'split':indexed[f['parent_id']]['name'],'kind':f['name'],'files':rows})
            atomic_json(folder/'publisher-files.json',metadata)
            acquired=datetime.now(timezone.utc)
            receipt={'source_url':SOURCE,'downloaded_at':acquired.isoformat(),'expires_at':(acquired+timedelta(days=7)).isoformat(),'publisher_metadata_sha256':sha256(folder/'publisher-files.json'),'license_id':'CC-BY-4.0','publisher_file_sha256_verification_required':True}
            atomic_json(receipt_path,receipt)
        pairs,removed=select_pairs(metadata);entries=[p[k] for p in pairs for k in ('image','label')]
        (folder/'files').mkdir(exist_ok=True)
        def download(entry):
            target=local_file(folder,entry);expected=entry['content_details']['sha256_hash'];size=entry['size']
            if not 0<=size<=8*1024*1024:raise ValueError('publisher_file_size_limit')
            if target.exists():
                if target.stat().st_size!=size or sha256(target)!=expected:raise ValueError('cached_publisher_file_changed')
                return
            url=entry['content_details']['download_url']
            if not url.startswith('https://data.mendeley.com/public-files/datasets/s5hx9n2jc3/files/'):raise ValueError('unexpected_publisher_download_host')
            for attempt in range(3):
                try:
                    r=client.get(url);r.raise_for_status()
                    if len(r.content)!=size or hashlib.sha256(r.content).hexdigest()!=expected:raise ValueError('publisher_file_hash_mismatch')
                    target.write_bytes(r.content);return
                except httpx.HTTPError:
                    if attempt==2:raise
                    time.sleep(attempt+1)
        with ThreadPoolExecutor(16) as pool:
            for i,_ in enumerate(pool.map(download,entries),1):
                if i%400==0:print(json.dumps({'verified_files':i,'expected_files':len(entries)}),flush=True)
        atomic_json(folder/'acquisition-summary.json',{'image_label_pairs':len(pairs),'publisher_group_exclusions':removed,'verified_files':len(entries),'all_publisher_file_hashes_verified':True,'raw_bytes':sum(e['size'] for e in entries),'source_expires_at':receipt['expires_at'],'user_recordings_used':False})


def parse_yolo(text,width,height):
    objects=[]
    for line in text.splitlines():
        if not line.strip():continue
        fields=line.split()
        if len(fields)!=5 or fields[0]!='0':raise ValueError('single_class_yolo_pothole_required')
        cx,cy,w,h=map(float,fields[1:])
        if not all(math.isfinite(v) for v in (cx,cy,w,h)) or not 0<=cx<=1 or not 0<=cy<=1 or not 0<w<=1 or not 0<h<=1:raise ValueError('invalid_normalized_bbox')
        left,top=max(0.,(cx-w/2)*width),max(0.,(cy-h/2)*height)
        right,bottom=min(width,(cx+w/2)*width),min(height,(cy+h/2)*height)
        if right<=left or bottom<=top:raise ValueError('empty_clipped_bbox')
        objects.append({'class':'pothole','bbox_xywh_px':[left,top,right-left,bottom-top],'certainty':'confirmed'})
    return objects


def prepare(source=RAW,output=RELEASE):
    folder=workspace_path(source,exists=True);output=workspace_path(output)
    if not output.is_relative_to(ROOT/'data/training') or output.exists():raise ValueError('new_private_training_release_required')
    receipt=json.loads((folder/'source.json').read_text());metadata=json.loads((folder/'publisher-files.json').read_text())
    if sha256(folder/'publisher-files.json')!=receipt['publisher_metadata_sha256'] or datetime.fromisoformat(receipt['expires_at'])<=datetime.now(timezone.utc):raise ValueError('invalid_or_expired_source')
    pairs,removed=select_pairs(metadata);removed=Counter(removed)
    historical=[]
    for p in sorted((ROOT/'data/training').glob('research-pothole*/release.json')):
        if p.parent==output:continue
        for r in json.loads(p.read_text(encoding='utf-8')):
            if r.get('perceptual_dhash'):historical.append((r['sha256'],int(r['perceptual_dhash'],16),r['split']))
    all_hash={h for h,_,_ in historical};test_hash={h for h,_,s in historical if s=='test'}
    all_visual={v for _,v,_ in historical};test_visual={v for _,v,s in historical if s=='test'}
    rows=[];seen_hash=set();seen_visual=[]
    evidence=folder/'license-evidence.json'
    for p in pairs: # TEST is protected first, then validation, then TRAIN.
        image_path=local_file(folder,p['image']);label_path=local_file(folder,p['label'])
        for path,entry in ((image_path,p['image']),(label_path,p['label'])):
            if sha256(path)!=entry['content_details']['sha256_hash']:raise ValueError('publisher_file_changed')
        image=cv2.imread(str(image_path))
        if image is None:raise ValueError('publisher_image_decode_failed')
        digest=sha256(image_path);visual=dhash(image);flip=dhash(cv2.flip(image,1));split=p['split']
        oldhash,oldvisual=(test_hash,test_visual) if split=='train' else (all_hash,all_visual)
        if digest in oldhash or any(min((visual^v).bit_count(),(flip^v).bit_count())<=3 for v in oldvisual):
            removed['historical_test_overlap' if split=='train' else 'historical_used_image_overlap']+=1;continue
        if digest in seen_hash or any(s!=split and min((visual^v).bit_count(),(flip^v).bit_count())<=3 for v,s in seen_visual):
            removed['cross_split_exact_or_near_duplicate']+=1;continue
        objects=parse_yolo(label_path.read_text(encoding='utf-8'),image.shape[1],image.shape[0])
        seen_hash.add(digest);seen_visual.append((visual,split))
        identifier=str(uuid.uuid5(NAMESPACE,'MWPD2:'+p['image']['id']))
        rows.append({'sample_id':identifier,'source_id':str(uuid.uuid5(NAMESPACE,'source:'+digest)),'source_hash':digest,'leakage_group_id':'MWPD2:filename:'+p['group'],'split':split,'country':'Bangladesh-source-unverified-location','path':image_path.relative_to(ROOT).as_posix(),'sha256':digest,'perceptual_dhash':f'{visual:016x}','annotation_version':'MWPD2-publisher-YOLO-single-pothole','annotation_sha256':sha256(label_path),'source_url':SOURCE,'license_evidence_path':evidence.relative_to(ROOT).as_posix(),'license_evidence_sha256':sha256(evidence),'privacy_status':'restricted_research','qc_status':'publisher_annotations_machine_checked','release_scope':'offline_research','human_privacy_reviewed':False,'not_for_deployment':True,'project_acceptance_eligible':False,'independent_route_verified':False,'detection_labels_verified':True,'allowed_use':'Private offline CC-BY-4.0 research with attribution; no human privacy/deployment approval','expires_at':receipt['expires_at'],'source_expires_at':receipt['expires_at'],'objects':objects})
    checked=validate_research_release(rows)
    if not checked['valid']:raise ValueError(str(checked['errors'][:3]))
    counts=Counter(r['split'] for r in rows)
    if counts['test']<30 or counts['validation']<50 or counts['train']<300:raise ValueError('insufficient_safe_grouped_split')
    output.mkdir(parents=True);atomic_json(output/'release.json',rows)
    atomic_json(output/'split-evidence.json',{'source_url':SOURCE,'release_sha256':sha256(output/'release.json'),'counts':dict(counts),'positive_images':dict(Counter(r['split'] for r in rows if r['objects'])),'removed':dict(removed),'publisher_hashes_verified':True,'historical_release_count':len(list((ROOT/'data/training').glob('research-pothole*/release.json')))-1,'physical_route_independence_verified':False,'limits':'Filename groups and exact/dHash(+horizontal flip) filters reduce leakage; web-source provenance/physical route independence not established. Previously evaluated benchmarks are regression only.'})
    print(json.dumps({'counts':dict(counts),'removed':dict(removed),'release_sha256':sha256(output/'release.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['acquire','prepare']);p.add_argument('--source',default=RAW);p.add_argument('--output');a=p.parse_args()
    if a.action=='acquire':acquire(a.output or RAW)
    else:prepare(a.source,a.output or RELEASE)
