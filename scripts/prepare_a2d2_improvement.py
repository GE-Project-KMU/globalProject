"""Freeze recording/date-disjoint A2D2 research extension before training.

Writes only a new private dataset. Existing releases and retention state stay
unchanged. Publisher labels do not establish road independence or wear GT.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,uuid
from pathlib import Path
from datetime import datetime,timedelta,timezone
from concurrent.futures import ThreadPoolExecutor
from collections import Counter,defaultdict
import urllib.parse
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import prep_a2d2 as source
from agent.dataset import validate_research_release

BASE=ROOT/'data/public/a2d2-improvement-v2'
OLD=ROOT/'data/public/a2d2/manifest.json'
ADDITIONAL_TRAIN=('20180925_112730','20180925_124435','20180925_135056','20181016_082154')
FRESH_TEST=('20181107_132300','20181107_133445')

def write_once(path,value):
    text=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('immutable_preparation_file_differs')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:stream.write(text)

def rel(path):return path.relative_to(ROOT).as_posix()

def freeze_plan():
    BASE.mkdir(parents=True,exist_ok=True)
    path=BASE/'pretraining-plan.json'
    if path.exists():
        plan=json.loads(path.read_text(encoding='utf-8'))
        if plan['old_manifest_sha256']!=source.sha(OLD):raise ValueError('old_release_changed')
        return plan
    old=json.loads(OLD.read_text(encoding='utf-8'))
    old_recordings={r['recording'] for r in old}
    if old_recordings&set(ADDITIONAL_TRAIN+FRESH_TEST):raise ValueError('recording_previously_used')
    if {x[:8] for x in FRESH_TEST}&{x[:8] for x in old_recordings|set(ADDITIONAL_TRAIN)}:raise ValueError('test_date_not_disjoint')
    def list_pairs(recording):
        labels=sorted([x for x in source.listing(f'camera_lidar_semantic/{recording}/label/cam_front_center/') if x['key'].endswith('.png')],key=lambda x:x['key'])
        cameras={x['key']:x for x in source.listing(f'camera_lidar_semantic/{recording}/camera/cam_front_center/') if x['key'].endswith('.png')}
        split='train' if recording in ADDITIONAL_TRAIN else 'test';count=100 if split=='train' else 50
        if len(labels)<count:raise ValueError('recording_has_insufficient_pairs')
        pairs=[]
        for index in np.linspace(0,len(labels)-1,count,dtype=int):
            label=labels[int(index)];key=label['key'].replace('/label/','/camera/').replace('_label_','_camera_')
            if key not in cameras:raise ValueError('publisher_camera_label_pair_missing')
            pairs.append({'recording':recording,'split':split,'image':cameras[key],'label':label})
        return {'recording':recording,'available_label_count':len(labels),'selected_count':count,'split':split,'pairs':pairs}
    with ThreadPoolExecutor(max_workers=6) as pool:recordings=list(pool.map(list_pairs,ADDITIONAL_TRAIN+FRESH_TEST))
    pairs=[p for recording in recordings for p in recording['pairs']]
    total=sum(p[k]['bytes'] for p in pairs for k in ('image','label'))
    if total>3_000_000_000:raise ValueError('new_acquisition_exceeds_predeclared_3GB')
    plan={'schema_version':1,'created_before_training':True,'created_at':datetime.now(timezone.utc).isoformat(),'old_manifest_sha256':source.sha(OLD),'selection':'Uniformly spaced sorted front-center frame indices; source/date split fixed before pixel or prediction inspection; no outcome-based replacements','additional_training_recordings':list(ADDITIONAL_TRAIN),'fresh_test_recordings':list(FRESH_TEST),'new_pairs':pairs,'recording_summary':[{k:v for k,v in item.items() if k!='pairs'} for item in recordings],'total_new_source_bytes':total,'old_test_role':'regression_only_never_training','duplicate_policy':'Connected exact SHA256 or 64bit grayscale DCT pHash Hamming<=6; remove new train connected to any old validation/test or fresh test; fresh test must not connect to any old source or development data. No label/outcome-based replacement.','physical_roads_independently_verified':False,'project_acceptance_eligible':False}
    write_once(path,plan)
    return plan

def acquire(item,kind):
    branch='quarantined-test' if item['split']=='test' else 'development'
    path=(BASE/branch/'original'/item[kind]['key']).resolve()
    if not path.is_relative_to(BASE):raise ValueError('download_path_outside_dataset')
    raw=path.read_bytes() if path.exists() else source.get(source.HOST+urllib.parse.quote(item[kind]['key'],safe='/'))
    if len(raw)!=item[kind]['bytes'] or hashlib.md5(raw).hexdigest()!=item[kind]['etag']:raise ValueError('publisher_size_or_md5_mismatch')
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)
    return path

def build():
    plan=freeze_plan();old=json.loads(OLD.read_text(encoding='utf-8'))
    if source.sha(OLD)!=plan['old_manifest_sha256']:raise ValueError('old_release_changed')
    acquisition_path=BASE/'acquisition.json'
    if not acquisition_path.exists():
        now=datetime.now(timezone.utc)
        write_once(acquisition_path,{'acquired_at':now.isoformat(),'expires_at':(now+timedelta(days=7)).isoformat(),'plan_sha256':source.sha(BASE/'pretraining-plan.json'),'scope':'Only newly acquired extension files; all existing rows retain their original expiry'})
    acquisition=json.loads(acquisition_path.read_text());expiry=acquisition['expires_at']
    if acquisition['plan_sha256']!=source.sha(BASE/'pretraining-plan.json') or datetime.fromisoformat(expiry)<=datetime.now(timezone.utc):raise ValueError('acquisition_plan_changed_or_expired')
    classes=json.loads((ROOT/'data/public/a2d2/class_list.json').read_text())
    if classes.get('#ffc125')!='Solid line' or classes.get('#8000ff')!='Dashed line':raise ValueError('class_mapping_changed')
    license_path=ROOT/'data/public/a2d2/LICENSE.txt'
    if 'CC BY-ND 4.0' not in license_path.read_text():raise ValueError('license_changed')
    valid_colors=np.array([int(c[1:],16) for c in classes],np.uint32)
    def prepare(item):
        image=acquire(item,'image');label=acquire(item,'label')
        with Image.open(image) as im:im.load();size=im.size;visual=source.phash(im)
        with Image.open(label) as im:rgb=np.asarray(im.convert('RGB'),np.uint32)
        if rgb.shape[:2]!=(size[1],size[0]):raise ValueError('label_image_shape_mismatch')
        packed=rgb[:,:,0]*65536+rgb[:,:,1]*256+rgb[:,:,2]
        mask=np.full(packed.shape,255,np.uint8);mask[np.isin(packed,valid_colors)]=0
        mask[(packed==0xffc125)|(packed==0x8000ff)]=1;mask[np.isin(packed,[0x60458f,0x352e52])]=255
        sid=str(uuid.uuid5(uuid.NAMESPACE_URL,source.HOST+item['image']['key']))
        branch='quarantined-test' if item['split']=='test' else 'development';mask_path=BASE/branch/'paint-masks'/(sid+'.png');mask_path.parent.mkdir(parents=True,exist_ok=True)
        if not mask_path.exists():Image.fromarray(mask).save(mask_path)
        else:
            with Image.open(mask_path) as previous:
                if not np.array_equal(mask,np.asarray(previous)):raise ValueError('existing_derived_mask_changed')
        template=old[0].copy();digest=source.sha(image)
        template.update(sample_id=sid,source_id=sid,source_hash=digest,path=rel(image),sha256=digest,mask_path=rel(mask_path),mask_sha256=source.sha(mask_path),publisher_mask_path=rel(label),publisher_mask_sha256=source.sha(label),split=item['split'],recording=item['recording'],leakage_group_id='A2D2-recording:'+item['recording'],expires_at=expiry,image={'width':size[0],'height':size[1]},positive_pixels=int((mask==1).sum()),valid_pixels=int((mask!=255).sum()),perceptual_phash=f'{visual:016x}',source_url=source.HOST+item['image']['key'],new_acquisition_sha256=source.sha(acquisition_path),evaluation_role='fresh_final_holdout' if item['split']=='test' else 'training')
        return template
    new=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index,row in enumerate(pool.map(prepare,plan['new_pairs'])):
            new.append(row)
            if (index+1)%25==0:print(json.dumps({'acquired_pairs':index+1,'total':len(plan['new_pairs'])}),flush=True)
    allrows=old+new;parent=list(range(len(allrows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(allrows):
        for j in range(i):
            other=allrows[j]
            if row['sha256']==other['sha256'] or (int(row['perceptual_phash'],16)^int(other['perceptual_phash'],16)).bit_count()<=6:parent[find(i)]=find(j)
    groups=defaultdict(set)
    for i,row in enumerate(allrows):groups[find(i)].add(('old_' if i<len(old) else 'new_')+row['split'])
    retained=[];removed=Counter()
    for offset,row in enumerate(new,len(old)):
        roles=groups[find(offset)]
        forbidden=(roles-{'new_test'}) if row['split']=='test' else roles&{'old_validation','old_test','new_test'}
        if forbidden:removed[row['split']]+=1
        else:retained.append(row)
    development=[r for r in old if r['split'] in ('train','validation')]+[r for r in retained if r['split']=='train']
    fresh=[r for r in retained if r['split']=='test']
    combined=development+fresh
    validation=validate_research_release(combined)
    if not validation['valid']:raise ValueError('combined_release_integrity_failed')
    write_once(BASE/'development-manifest.json',development)
    write_once(BASE/'quarantined-test'/'manifest.json',fresh)
    # Full release is for immutable provenance/evaluation, not loader selection.
    write_once(BASE/'manifest.json',combined)
    write_once(BASE/'regression-manifest.json',[r|{'evaluation_role':'previously_seen_regression_only'} for r in old if r['split']=='test'])
    report={'format':'a2d2-improvement-source-qc-v1','frozen_before_training':True,'old_manifest_sha256':source.sha(OLD),'new_plan_sha256':source.sha(BASE/'pretraining-plan.json'),'acquisition_sha256':source.sha(acquisition_path),'new_source_expiry':expiry,'old_deadlines_preserved':True,'manifest_sha256':source.sha(BASE/'manifest.json'),'development_manifest_sha256':source.sha(BASE/'development-manifest.json'),'fresh_test_manifest_sha256':source.sha(BASE/'quarantined-test'/'manifest.json'),'counts':dict(Counter(r['split'] for r in combined)),'new_duplicate_candidates_excluded':dict(removed),'fresh_test_isolated':True,'old_test_reused_for_training':False,'new_test_connected_to_old_data':False,'recording_heldout_verified':True,'physical_roads_independently_verified':False,'project_acceptance_eligible':False,'known_class_mapping_verified':True,'publisher_size_md5_and_png_sha256_verified':True,'human_privacy_reviewed':False,'source_license':'CC BY-ND 4.0','source_url':'https://registry.opendata.aws/aev-a2d2/','source_new_bytes':plan['total_new_source_bytes'],'script_sha256':source.sha(Path(__file__)),'limitations':['Recording/date independence does not establish physically independent roads.','Previous TEST is regression only. No model or threshold selection may use fresh final holdout.','Publisher solid/dashed pixels are not independently annotated wear/outline/gap/valid GT.','Machine checks do not establish human privacy approval or redistribution permission.']}
    write_once(BASE/'source-qc.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    if args.plan_only:
        p=freeze_plan();print(json.dumps({'plan_frozen':True,'recordings':p['recording_summary'],'new_bytes':p['total_new_source_bytes'],'plan_sha256':source.sha(BASE/'pretraining-plan.json')},indent=2))
    else:build()

if __name__=='__main__':main()
