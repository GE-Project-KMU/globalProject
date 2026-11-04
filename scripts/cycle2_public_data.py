"""New public-source-only cycle; no user recordings or historical TEST fitting.

All historical releases are leakage blacklists, not training expansion. New
recording partitions are selected before pixels/model outcomes and immutable.
"""
import argparse,hashlib,json,random,sys,urllib.parse,uuid
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.public_refinement_data import (read,rel,freeze,novel_components,audit_development,a2,SOURCE,annotations,ident,digest,dhash,round_robin,verify_archive,clipped_box,io,zipfile)
from agent.paths import sha256,atomic_json
from agent.dataset import validate_research_release
LANE_HISTORY=[ROOT/p for p in ['data/public/a2d2/manifest.json','data/public/a2d2-improvement-v2/manifest.json','data/public/a2d2-refinement-v3/release.json']]
PRIVACY_HISTORY=[ROOT/f'data/training/{name}/release.json' for name in ['research-redaction','research-redaction-fullframe','research-redaction-v4','research-redaction-v6']]
LANE_OUT=ROOT/'data/public/a2d2-cycle2-v5';PRIVACY_OUT=ROOT/'data/training/research-redaction-cycle2-v7'

def cross_split_clean(rows,hash_field,visual_field,radius):
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(rows):
        for j in range(i):
            b=rows[j]
            if a[hash_field]==b[hash_field] or (int(a[visual_field],16)^int(b[visual_field],16)).bit_count()<=radius:parent[find(i)]=find(j)
    splitsets=defaultdict(set)
    for i,row in enumerate(rows):splitsets[find(i)].add(row['split'])
    return [r for i,r in enumerate(rows) if len(splitsets[find(i)])==1]

def lane():
    history=[r for p in LANE_HISTORY for r in read(p)]
    olddev=read(ROOT/'data/public/a2d2-refinement-v3/development.json')
    audit_development(olddev,history);expiry=min(r['expires_at'] for r in history)
    used={r['recording'] for r in history}
    available=sorted(set(read(ROOT/'data/training/lane-numeric-diagnostics-v3/unused-recording-metadata.json')['unused_recordings'])-used)
    if len(available)!=9:raise ValueError('expected_nine_unused_recordings_audit_required')
    shuffled=available[:];random.Random(20261106).shuffle(shuffled)
    splits={r:('test' if i<2 else 'validation' if i==2 else 'train') for i,r in enumerate(shuffled)}
    planpath=LANE_OUT/'acquisition-plan.json'
    if planpath.exists():plan=read(planpath)
    else:
        pairs=[]
        for recording in sorted(splits):
            split=splits[recording];count={'train':100,'validation':80,'test':60}[split];prefix=f'camera_lidar_semantic/{recording}/'
            labels=sorted((v for v in a2.listing(prefix+'label/cam_front_center/') if v['key'].endswith('.png')),key=lambda v:v['key'])
            cameras={v['key']:v for v in a2.listing(prefix+'camera/cam_front_center/') if v['key'].endswith('.png')}
            if len(labels)<count:raise ValueError('insufficient_publisher_samples')
            for index in np.linspace(0,len(labels)-1,count,dtype=int):
                label=labels[int(index)];camera=label['key'].replace('/label/','/camera/').replace('_label_','_camera_')
                pairs.append({'recording':recording,'split':split,'label':label,'image':cameras[camera]})
        plan={'format':'a2d2-cycle2-acquisition-v1','seed':20261106,'recording_splits':splits,'pairs':pairs,'source':'official A2D2 AWS','uniform_selection_before_pixels':True,'source_histories':[sha256(p) for p in LANE_HISTORY],'expiry':expiry,'source_expiry_extended':False,'total_bytes':sum(p[k]['bytes'] for p in pairs for k in ('image','label')),'physical_route_or_date_independence_verified':False}
        if plan['total_bytes']>6_000_000_000:raise ValueError('acquisition_budget_6GB_exceeded')
        freeze(planpath,plan)
    if plan['source_histories']!=[sha256(p) for p in LANE_HISTORY] or plan['recording_splits']!=splits:raise ValueError('history_or_split_changed')
    colors=np.array([int(k[1:],16) for k in read(ROOT/'data/public/a2d2/class_list.json')],np.uint32)
    def prepare(item):
        saved={}
        for kind in ('image','label'):
            path=LANE_OUT/'original'/item[kind]['key'];path.parent.mkdir(parents=True,exist_ok=True)
            raw=path.read_bytes() if path.exists() else a2.get(a2.HOST+urllib.parse.quote(item[kind]['key'],safe='/'))
            if len(raw)!=item[kind]['bytes'] or hashlib.md5(raw).hexdigest()!=item[kind]['etag']:raise ValueError('publisher_object_mismatch')
            if not path.exists():path.write_bytes(raw)
            saved[kind]=path
        image,label=saved['image'],saved['label']
        with Image.open(image) as im:im.load();w,h=im.size;visual=a2.phash(im)
        with Image.open(label) as im:rgb=np.asarray(im.convert('RGB'),np.uint32)
        if rgb.shape[:2]!=(h,w):raise ValueError('publisher_shape_mismatch')
        packed=rgb[:,:,0]*65536+rgb[:,:,1]*256+rgb[:,:,2];mask=np.full(packed.shape,255,np.uint8);mask[np.isin(packed,colors)]=0
        mask[np.isin(packed,[0xffc125,0x8000ff])]=1;mask[np.isin(packed,[0x60458f,0x352e52])]=255
        sid=str(uuid.uuid5(uuid.NAMESPACE_URL,a2.HOST+item['image']['key']));target=LANE_OUT/'masks'/(sid+'.png');target.parent.mkdir(exist_ok=True)
        if not target.exists():Image.fromarray(mask).save(target)
        return dict(history[0],sample_id=sid,source_id=sid,source_hash=sha256(image),sha256=sha256(image),path=rel(image),mask_path=rel(target),mask_sha256=sha256(target),publisher_mask_path=rel(label),publisher_mask_sha256=sha256(label),split=item['split'],recording=item['recording'],leakage_group_id='A2D2-recording:'+item['recording'],expires_at=expiry,image={'width':w,'height':h},perceptual_phash=f'{visual:016x}',source_url=a2.HOST+item['image']['key'],positive_pixels=int((mask==1).sum()),valid_pixels=int((mask!=255).sum()))
    LANE_OUT.mkdir(parents=True,exist_ok=True);freeze(LANE_OUT/'source.json',{'expires_at':expiry,'source_expiry_extended':False})
    with ThreadPoolExecutor(max_workers=6) as pool:new=list(pool.map(prepare,plan['pairs']))
    # Exclude all connected novel components touching ANY prior sample (all old splits).
    history_clean=novel_components(history,new,'source_hash','perceptual_phash',6)
    novel=cross_split_clean(history_clean,'source_hash','perceptual_phash',6)
    release=olddev+novel;state=validate_research_release(release)
    if not state['valid']:raise ValueError('release_invalid:'+str(state['errors'][:3]))
    dev=[r for r in release if r['split'] in ('train','validation')];fresh=[r for r in novel if r['split']=='test']
    if len(fresh)<60 or len({r['recording'] for r in fresh})!=2:raise ValueError('fresh_support_too_small_no_replacement')
    audit=audit_development(dev,history)
    freeze(LANE_OUT/'development.json',dev);freeze(LANE_OUT/'fresh-test.json',fresh);freeze(LANE_OUT/'release.json',release)
    report={'format':'public-cycle2-data-qc-v1','task':'lane','development_counts':dict(Counter(r['split'] for r in dev)),'new_recording_counts':dict(Counter(splits.values())),'fresh_images':len(fresh),'initial_new_samples':len(new),'connected_history_duplicates_excluded':len(new)-len(history_clean),'connected_new_cross_split_excluded':len(history_clean)-len(novel),'development_sha256':sha256(LANE_OUT/'development.json'),'fresh_sha256':sha256(LANE_OUT/'fresh-test.json'),'release_sha256':sha256(LANE_OUT/'release.json'),'expiry':expiry,'license':'CC-BY-ND-4.0','physical_route_or_date_independence_verified':False,'TEST_fitting':False,'user_recordings_used':False,'source_expiries_extended':False,'development_audit':audit}
    freeze(LANE_OUT/'source-qc.json',report);atomic_json(ROOT/'docs/reports/LANE-CYCLE2-DATA.json',report);print(json.dumps(report),flush=True)

def privacy():
    history=[r for p in PRIVACY_HISTORY for r in read(p)];current=read(PRIVACY_HISTORY[-1]);development=[r for r in current if r['split'] in ('train','validation')]
    audit=audit_development(development,history);expiry=min(r['expires_at'] for r in history)
    events=read(ROOT/'data/training/research-redaction/source-qc.json')['face_event_split']
    records={name:verify_archive(name) for name in ['WIDER_train.zip','WIDER_val.zip','wider_face_split.zip','CCPD2020.zip']}
    oldids={r['source_id'] for r in history};oldhashes={r['source_hash'] for r in history};oldvisual={int(r['source_dhash'],16) for r in history}
    oldplate={r['leakage_group_id'] for r in history if r.get('evaluation_class',r['objects'][0]['class'])=='license_plate'}
    fresh=[];rejected=Counter();PRIVACY_OUT.mkdir(parents=True,exist_ok=True);(PRIVACY_OUT/'fresh-originals').mkdir(exist_ok=True)
    freeze(PRIVACY_OUT/'fresh-plan.json',{'selection':'source-name sorted round-robin heldout publisher groups; no outcome based selection','source_histories':[sha256(p) for p in PRIVACY_HISTORY],
        'face_target':100,'plate_target':100,'expiry':expiry,'model_predictions_examined':False,'independent_person_or_route_verified':False,'face_event_partition_unchanged':True})
    with zipfile.ZipFile(SOURCE/'wider_face_split.zip') as labels,zipfile.ZipFile(SOURCE/'WIDER_train.zip') as trainzip,zipfile.ZipFile(SOURCE/'WIDER_val.zip') as valzip,zipfile.ZipFile(SOURCE/'CCPD2020.zip') as plates:
        facegroups=defaultdict(list)
        for ps in ['train','val']:
            for name,boxes in annotations(labels.read(f'wider_face_split/wider_face_{ps}_bbx_gt.txt').decode()):
                event=name.split('/')[0]
                if events[event]=='test':facegroups[event].append((name,boxes,ps))
        plategroups=defaultdict(list);partitions=defaultdict(set)
        for name in plates.namelist():
            fields=Path(name).stem.split('-')
            if not name.endswith('.jpg') or len(fields)!=7:continue
            group='CCPD-plate-sha256:'+digest(('CCPD2020:'+fields[4]).encode());ps=Path(name).parts[-2];partitions[group].add(ps);plategroups[group].append((name,group,ps))
        for kind,entries in [('face',round_robin(facegroups)),('license_plate',round_robin({g:v for g,v in plategroups.items() if partitions[g]=={'test'} and g not in oldplate}))]:
            count=0;newgroups=set()
            for entry in entries:
                name,other,ps=entry;sid=ident(kind+':'+name)
                if sid in oldids:continue
                group='WIDER-event:'+name.split('/')[0] if kind=='face' else other
                if kind=='license_plate' and group in newgroups:continue
                raw=(trainzip if ps=='train' else valzip).read(f'WIDER_{ps}/images/{name}') if kind=='face' else plates.read(name)
                digestval=digest(raw)
                with Image.open(io.BytesIO(raw)) as im:im.load();w,h=im.size;visual=dhash(im)
                if digestval in oldhashes or any((visual^v).bit_count()<=3 for v in oldvisual):rejected['history_exact_or_near']+=1;continue
                objects=[];invalid=False
                if kind=='face':
                    for box in other:
                        bbox=clipped_box(box[:4],w,h)
                        if not bbox:continue
                        if box[7]:invalid=True;break
                        objects.append({'class':kind,'bbox_xywh_px':bbox,'certainty':'confirmed'})
                else:
                    a,b=Path(name).stem.split('-')[2].split('_');x,y=map(float,a.split('&'));xx,yy=map(float,b.split('&'));bbox=clipped_box([x,y,xx-x,yy-y],w,h)
                    if bbox:objects=[{'class':kind,'bbox_xywh_px':bbox,'certainty':'confirmed'}]
                if invalid or not objects or len(objects)>512:rejected['invalid_publisher_or_gt_count']+=1;continue
                target=PRIVACY_OUT/'fresh-originals'/(sid+'.jpg');target.write_bytes(raw)
                template=next(r for r in current if r['evaluation_class']==kind)
                fresh.append(dict(template,sample_id=ident('cycle2-v7:'+sid),source_id=sid,parent_sample_id=sid,source_hash=digestval,sha256=digestval,path=rel(target),
                    image={'width':w,'height':h},source_dhash=f'{visual:016x}',leakage_group_id=group,split='test',publisher_split=ps,objects=objects,expires_at=expiry,
                    annotation_version='publisher_original_cycle2_v7',frozen_crop_source_sample_id=None,test_status='new_source_unseen_before_cycle2_training_no_person_independence_claim'))
                newgroups.add(group);count+=1
                if count==100:break
            if count!=100:raise ValueError('insufficient_unused_private_research_source:'+kind)
    retained=novel_components(history,fresh,'source_hash','source_dhash',3)
    if len(retained)!=200:raise ValueError('fresh_connected_component_cross_history_unexpected')
    state=validate_research_release(development+retained)
    if not state['valid']:raise ValueError('combined_privacy_invalid:'+str(state['errors'][:3]))
    freeze(PRIVACY_OUT/'development.json',development);freeze(PRIVACY_OUT/'fresh-test.json',retained);freeze(PRIVACY_OUT/'release.json',development+retained)
    summary={'kind':'privacy','development':audit,'fresh_counts':dict(Counter(r['evaluation_class'] for r in retained)),'fresh_GT':dict(Counter(o['class'] for r in retained for o in r['objects'])),
        'fresh_manifest_sha256':sha256(PRIVACY_OUT/'fresh-test.json'),'development_sha256':sha256(PRIVACY_OUT/'development.json'),'release_sha256':sha256(PRIVACY_OUT/'release.json'),
        'expiry':expiry,'licenses':['CC-BY-NC-ND-4.0','MIT'],'research_only':True,'rejected_candidates':dict(rejected),'independent_person_or_route_verified':False,
        'known_class_conditional_only':True,'frozen_before_training':True,'user_recordings_used':False}
    freeze(PRIVACY_OUT/'source-qc.json',summary);atomic_json(ROOT/'docs/reports/PRIVACY-CYCLE2-DATA.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['lane','privacy']);a=p.parse_args();lane() if a.kind=='lane' else privacy()
