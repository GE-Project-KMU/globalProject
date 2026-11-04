"""Public-publisher-only refinement releases. Never reads user driving recordings.

Old TEST images never enter fitting. New evaluation images are frozen before
training; recording/person/route independence limits remain explicitly disclosed.
"""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import argparse,hashlib,io,json,sys,urllib.parse,uuid,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from agent.dataset import validate_research_release
from scripts import prep_a2d2 as a2
from scripts.prepare_redaction_research import SOURCE,annotations,ident,digest,dhash,round_robin,verify_archive
from scripts.prepare_redaction_fullscene import clipped_box
LANE_HISTORY=[ROOT/'data/public/a2d2/manifest.json',ROOT/'data/public/a2d2-improvement-v2/manifest.json']
PRIVACY_HISTORY=[ROOT/f'data/training/{name}/release.json' for name in ['research-redaction','research-redaction-fullframe','research-redaction-v4']]
LANE_OUT=ROOT/'data/public/a2d2-refinement-v3';PRIVACY_OUT=ROOT/'data/training/research-redaction-v6'

def read(p):return json.loads(workspace_path(p,exists=True).read_text(encoding='utf-8'))
def rel(p):return p.relative_to(ROOT).as_posix()
def freeze(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError('immutable_preparation_changed')
    else:atomic_json(path,value)

def novel_components(old,new,hash_field,visual_field,radius):
    """Remove ALL new members connected to ANY old image, including chains."""
    rows=old+new;parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        value=int(row[visual_field],16)
        for j in range(i):
            other=rows[j]
            if row[hash_field]==other[hash_field] or (value^int(other[visual_field],16)).bit_count()<=radius:parent[find(i)]=find(j)
    oldroots={find(i) for i in range(len(old))}
    return [r for i,r in enumerate(new,len(old)) if find(i) not in oldroots]

def audit_development(dev,histories):
    if any(r['split'] not in ('train','validation') for r in dev):raise ValueError('test_in_development')
    past_test=[r for r in histories if r['split']=='test'];ids={r['source_id'] for r in past_test};hashes={r['source_hash'] for r in past_test}
    if any(r['source_id'] in ids or r['source_hash'] in hashes for r in dev):raise ValueError('historical_test_in_development')
    if any('recorded-drive' in r['path'] or r['path'].startswith('데이터/') for r in dev):raise ValueError('user_recording_forbidden')
    state=validate_research_release(dev)
    if not state['valid']:raise ValueError('development_release_invalid:'+str(state['errors'][:3]))
    return {'development_rows':len(dev),'historical_test_rows':len(past_test),'old_test_used_for_fitting':False,'source_expiry_extended':False}

def lane():
    history=[r for p in LANE_HISTORY for r in read(p)];development=read(ROOT/'data/public/a2d2-improvement-v2/development-manifest.json')
    audit=audit_development(development,history);expiry=min(r['expires_at'] for r in history)
    oldgroups={r['recording'] for r in history};available=read(ROOT/'data/training/lane-numeric-diagnostics-v3/unused-recording-metadata.json')['unused_recordings']
    chosen=sorted(set(available)-oldgroups)[:2]
    if len(chosen)!=2:raise ValueError('two_unused_recordings_required')
    plan_path=LANE_OUT/'fresh-plan.json'
    if plan_path.exists():plan=read(plan_path)
    else:
        pairs=[]
        for recording in chosen:
            prefix=f'camera_lidar_semantic/{recording}/'
            labels=sorted((v for v in a2.listing(prefix+'label/cam_front_center/') if v['key'].endswith('.png')),key=lambda v:v['key'])
            cameras={v['key']:v for v in a2.listing(prefix+'camera/cam_front_center/') if v['key'].endswith('.png')}
            if len(labels)<40:raise ValueError('insufficient_label_support')
            for index in np.linspace(0,len(labels)-1,40,dtype=int):
                label=labels[int(index)];camera=label['key'].replace('/label/','/camera/').replace('_label_','_camera_')
                pairs.append({'recording':recording,'label':label,'image':cameras[camera]})
        plan={'source':'official A2D2 AWS','recordings':chosen,'pairs':pairs,'uniform_selection_before_pixels':True,
            'expiry':expiry,'historical_manifest_hashes':[sha256(p) for p in LANE_HISTORY],
            'total_bytes':sum(p[k]['bytes'] for p in pairs for k in ('image','label')),'physical_route_independent':False,'date_independent':False}
        if plan['total_bytes']>600_000_000:raise ValueError('fresh_download_budget_600MB_exceeded')
        freeze(plan_path,plan)
    if plan['historical_manifest_hashes']!=[sha256(p) for p in LANE_HISTORY]:raise ValueError('history_changed')
    classes=read(ROOT/'data/public/a2d2/class_list.json');colors=np.array([int(k[1:],16) for k in classes],np.uint32)
    def acquire(item,kind):
        path=LANE_OUT/'fresh-original'/item[kind]['key'];path.parent.mkdir(parents=True,exist_ok=True)
        raw=path.read_bytes() if path.exists() else a2.get(a2.HOST+urllib.parse.quote(item[kind]['key'],safe='/'))
        if len(raw)!=item[kind]['bytes'] or hashlib.md5(raw).hexdigest()!=item[kind]['etag']:raise ValueError('publisher_object_mismatch')
        if not path.exists():path.write_bytes(raw)
        return path
    def prepare(item):
        image=acquire(item,'image');label=acquire(item,'label')
        with Image.open(image) as im:im.load();size=im.size;visual=a2.phash(im)
        with Image.open(label) as im:rgb=np.asarray(im.convert('RGB'),np.uint32)
        if rgb.shape[:2]!=(size[1],size[0]):raise ValueError('label_shape_mismatch')
        packed=rgb[:,:,0]*65536+rgb[:,:,1]*256+rgb[:,:,2];mask=np.full(packed.shape,255,np.uint8);mask[np.isin(packed,colors)]=0
        mask[(packed==0xffc125)|(packed==0x8000ff)]=1;mask[np.isin(packed,[0x60458f,0x352e52])]=255
        sid=str(uuid.uuid5(uuid.NAMESPACE_URL,a2.HOST+item['image']['key']));target=LANE_OUT/'fresh-masks'/(sid+'.png');target.parent.mkdir(exist_ok=True)
        if not target.exists():Image.fromarray(mask).save(target)
        value=dict(history[0],sample_id=sid,source_id=sid,source_hash=sha256(image),path=rel(image),sha256=sha256(image),mask_path=rel(target),mask_sha256=sha256(target),
            publisher_mask_path=rel(label),publisher_mask_sha256=sha256(label),split='test',recording=item['recording'],leakage_group_id='A2D2-recording:'+item['recording'],
            expires_at=expiry,image={'width':size[0],'height':size[1]},perceptual_phash=f'{visual:016x}',source_url=a2.HOST+item['image']['key'],
            evaluation_role='new_unseen_recording_only_no_physical_route_or_date_independence',positive_pixels=int((mask==1).sum()),valid_pixels=int((mask!=255).sum()))
        return value
    with ThreadPoolExecutor(max_workers=6) as pool:fresh=list(pool.map(prepare,plan['pairs']))
    retained=novel_components(history,fresh,'source_hash','perceptual_phash',6)
    if len(retained)<20:raise ValueError('too_few_fresh_samples_after_leakage_exclusion')
    combined=development+retained;state=validate_research_release(combined)
    if not state['valid']:raise ValueError('combined_lane_invalid')
    freeze(LANE_OUT/'development.json',development);freeze(LANE_OUT/'fresh-test.json',retained);freeze(LANE_OUT/'release.json',combined)
    summary={'kind':'lane','development':audit,'fresh_samples':len(retained),'initial_fresh_samples':80,'fresh_recordings':len(chosen),'connected_duplicate_excluded':len(fresh)-len(retained),
        'fresh_manifest_sha256':sha256(LANE_OUT/'fresh-test.json'),'development_sha256':sha256(LANE_OUT/'development.json'),'release_sha256':sha256(LANE_OUT/'release.json'),
        'expiry':expiry,'license':'CC-BY-ND-4.0','research_only':True,'physical_route_independence_verified':False,'date_independent':False,'frozen_before_training':True,'user_recordings_used':False}
    freeze(LANE_OUT/'source-qc.json',summary);atomic_json(ROOT/'docs/reports/LANE-REFINEMENT-DATA.json',summary);print(json.dumps(summary),flush=True)

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
                fresh.append(dict(template,sample_id=ident('refinement-v6:'+sid),source_id=sid,parent_sample_id=sid,source_hash=digestval,sha256=digestval,path=rel(target),
                    image={'width':w,'height':h},source_dhash=f'{visual:016x}',leakage_group_id=group,split='test',publisher_split=ps,objects=objects,expires_at=expiry,
                    annotation_version='publisher_original_refinement_v6',frozen_crop_source_sample_id=None,test_status='new_source_unseen_before_refinement_training_no_person_independence_claim'))
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
    freeze(PRIVACY_OUT/'source-qc.json',summary);atomic_json(ROOT/'docs/reports/PRIVACY-REFINEMENT-DATA.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['lane','privacy']);a=p.parse_args();lane() if a.kind=='lane' else privacy()
