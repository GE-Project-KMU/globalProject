"""Freeze expanded original-frame specialist research data, never deployment approval."""
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import sys
import zipfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_redaction_research import SOURCE,annotations,digest,ident,rel,sha,write,dhash,round_robin,verify_archive
from scripts.prepare_redaction_fullscene import clipped_box
from agent.dataset import validate_research_release

DEST=ROOT/'data/training/research-redaction-v4'

def main():
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'originals').mkdir(exist_ok=True)
    if (DEST/'release.json').exists():raise ValueError('release_already_frozen')
    crop_path=ROOT/'data/training/research-redaction/release.json'
    v3_path=ROOT/'data/training/research-redaction-fullframe/release.json'
    old_crop=json.loads(crop_path.read_text());old=json.loads(v3_path.read_text());all_old=old_crop+old
    qc=json.loads((ROOT/'data/training/research-redaction/source-qc.json').read_text())
    events=qc['face_event_split'];archive_records={name:verify_archive(name) for name in ['WIDER_train.zip','WIDER_val.zip','CCPD2020.zip','wider_face_split.zip']}
    az={s:zipfile.ZipFile(SOURCE/f'WIDER_{s}.zip') for s in ('train','val')};cz=zipfile.ZipFile(SOURCE/'CCPD2020.zip')
    sz=zipfile.ZipFile(SOURCE/'wider_face_split.zip');faces=defaultdict(lambda:defaultdict(list))
    for ps in ('train','val'):
        for name,boxes in annotations(sz.read(f'wider_face_split/wider_face_{ps}_bbx_gt.txt').decode()):
            event=name.split('/')[0];faces[events[event]][event].append((name,boxes,ps))
    plates=defaultdict(list);partitions=defaultdict(set)
    for name in cz.namelist():
        fields=Path(name).stem.split('-')
        if not name.endswith('.jpg') or len(fields)!=7:continue
        group='CCPD-plate-sha256:'+digest(('CCPD2020:'+fields[4]).encode());ps=Path(name).parts[-2]
        plates[group].append((name,group,ps));partitions[group].add(ps)
    old_ids={r['source_id'] for r in all_old};old_hashes={r['source_hash'] for r in all_old}
    old_plate_groups={r['leakage_group_id'] for r in all_old if r['objects'][0]['class']=='license_plate'}
    old_visual=[int(r['source_dhash'],16) for r in all_old]
    rows=[];selected_ids=set();rejected=Counter()
    # Preserve genuine original TRAIN/VAL membership; old test is diagnostic only.
    for row in old:
        if row['split']=='test':continue
        new=dict(row);new['sample_id']=ident('fullframe-v4:'+row['source_id'])
        new['annotation_version']='publisher-original-fullframe-specialists-v4'
        new['test_status']='preserved_previous_train_validation_membership'
        new['coverage_limitation']='One known-class specialist is trained per source. The other class remains unannotated; combined evaluation is conditional, never complete privacy coverage.'
        rows.append(new);selected_ids.add(new['source_id'])
    def candidate(kind,entry,split):
        if kind=='face':
            name,boxes,ps=entry;group='WIDER-event:'+name.split('/')[0];raw=az[ps].read(f'WIDER_{ps}/images/{name}');record=archive_records[f'WIDER_{ps}.zip']
        else:
            name,group,ps=entry;boxes=None;raw=cz.read(name);record=archive_records['CCPD2020.zip']
        sid=ident(kind+':'+name)
        if sid in selected_ids:raise ValueError('source_already_selected')
        if split=='test' and (sid in old_ids or (kind=='license_plate' and group in old_plate_groups)):raise ValueError('test_seen_source_or_identity')
        h=digest(raw)
        with Image.open(io.BytesIO(raw)) as im:im.load();w,height=im.size;visual=dhash(im)
        if split=='test' and (h in old_hashes or any((visual^v).bit_count()<=3 for v in old_visual)):raise ValueError('test_seen_hash_or_near_duplicate')
        # No cross-partition exact/near duplicates. Every old test stays excluded from fitting.
        others=[r for r in all_old if r['split']!=split]+[r for r in rows if r['split']!=split]
        if any(h==r['source_hash'] or (visual^int(r['source_dhash'],16)).bit_count()<=3 for r in others):raise ValueError('cross_partition_hash_or_near_duplicate')
        objects=[]
        if kind=='face':
            for b in boxes:
                bbox=clipped_box(b[:4],w,height)
                if not bbox:continue
                if b[7]:raise ValueError('publisher_invalid_face_image_omitted')
                objects.append({'class':kind,'bbox_xywh_px':bbox,'certainty':'confirmed'})
        else:
            a,b=Path(name).stem.split('-')[2].split('_');x,y=map(float,a.split('&'));xx,yy=map(float,b.split('&'))
            bbox=clipped_box([x,y,xx-x,yy-y],w,height)
            if bbox:objects=[{'class':kind,'bbox_xywh_px':bbox,'certainty':'confirmed'}]
        if not objects or len(objects)>512:raise ValueError('no_gt_or_gt_count_exceeds512_whole_image_omitted')
        sample=ident('fullframe-v4:'+sid);path=DEST/'originals'/(sample+'.jpg');path.write_bytes(raw)
        policy=SOURCE/('wider-license.json' if kind=='face' else 'ccpd-license.json')
        template=next(r for r in old if r['evaluation_class']==kind)
        row={**template,'sample_id':sample,'source_id':sid,'parent_sample_id':sid,'source_hash':h,'sha256':h,'path':rel(path),'image':{'width':w,'height':height},'split':split,'publisher_split':ps,'leakage_group_id':group,'source_dhash':f'{visual:016x}','objects':objects,'expires_at':record['expires_at'],'license_evidence_path':rel(policy),'license_evidence_sha256':sha(policy),'annotation_version':'publisher-original-fullframe-specialists-v4','frozen_crop_source_sample_id':None,'test_status':'new_unseen_source_heldout_before_v4_training' if split=='test' else 'expanded_same_frozen_event_or_plate_identity_partition','coverage_limitation':'One known-class specialist per source; unknown other class is not an annotated negative for that specialist. Combined metrics remain known-class conditional, not global privacy acceptance.'}
        return row
    # Entire clean identity-exclusive train pool contains1993 plates, not2000.
    # This support-only correction is frozen before any candidate training.
    targets={('train','face'):3000,('validation','face'):400,('train','license_plate'):1993,('validation','license_plate'):108,('test','face'):200,('test','license_plate'):200}
    for (split,kind),target in targets.items():
        count=sum(r['split']==split and r['evaluation_class']==kind for r in rows)
        if kind=='face':entries=round_robin(faces[split])
        else:
            ps='val' if split=='validation' else split
            groups={g:v for g,v in plates.items() if partitions[g]=={ps} and (split!='test' or g not in old_plate_groups)}
            entries=round_robin(groups)
        new_groups=set()
        for entry in entries:
            if count>=target:break
            if kind=='license_plate' and split=='test' and entry[1] in new_groups:continue
            try:row=candidate(kind,entry,split)
            except ValueError as error:rejected[f'{split}:{kind}:{error}']+=1;continue
            rows.append(row);selected_ids.add(row['source_id']);new_groups.add(row['leakage_group_id']);count+=1
            if count%100==0:print(split,kind,count,flush=True)
        if count!=target:raise ValueError(f'insufficient_clean_sources:{split}:{kind}:{count}/{target}')
    state=validate_research_release(rows)
    if not state['valid']:raise ValueError(str(state['errors'][:10]))
    release=DEST/'release.json';write(release,rows)
    report={'manifest_path':rel(release),'manifest_sha256':sha(release),'script_sha256':sha(Path(__file__)),'counts':dict(Counter(r['split']+':'+r['evaluation_class'] for r in rows)),'gt_counts':dict(Counter(r['split']+':'+o['class'] for r in rows for o in r['objects'])),'max_gt':max(len(r['objects']) for r in rows),'rejected':dict(rejected),'previous_manifests':[{'sha256':sha(p),'path':rel(p)} for p in [crop_path,v3_path]],'previous_tests_status':'development diagnostic regression only; NEVER used in train/validation/new final test','face_event_split_unchanged':events,'test_policy':'Before model fitting: new final test excludes EVERY previous train/validation/test source UUID, SHA256, dHash<=3 neighbor, and plate identity; all new partitions crosschecked against existing other splits and newly selected other splits. Event groups preserved. Different WIDER events do not prove independent persons. Plate test identities are unique.','independent_person_identity_verified':False,'test_frozen_before_training':True,'model_predictions_used_for_selection':False,'source_licenses':qc['source_licenses'],'redistribution_policy':qc['redistribution_policy'],'machine_validation':state,'human_privacy_reviewed':False,'project_acceptance_eligible':False,'source_expiries_inherited_not_extended':True}
    write(DEST/'source-qc.json',report);write(ROOT/'docs/reports/REDACTION-SPECIALISTS-V4-DATA.json',report)
    print(json.dumps({'manifest_sha256':sha(release),'counts':report['counts'],'valid':state['valid']}),flush=True)

if __name__=='__main__':main()
