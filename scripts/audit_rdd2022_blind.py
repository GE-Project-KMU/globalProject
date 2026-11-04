"""Pixel/identity-only QC of frozen blind data; never loads a model."""
from __future__ import annotations
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from scripts.prepare_rdd2022_blind import filename_block,RELEASE,aware

def near_components(rows):
    parent=list(range(len(rows)));pairs=[]
    def find(index):
        while parent[index]!=index:index=parent[index]
        return index
    for i,row in enumerate(rows):
        a=(int(row['perceptual_dhash'],16),row['_flip_dhash'])
        for j,other in enumerate(rows[:i]):
            b=(int(other['perceptual_dhash'],16),other['_flip_dhash'])
            if min((x^y).bit_count() for x in a for y in b)<=3:
                pairs.append([j,i]);parent[find(i)]=find(j)
    components={}
    for i in range(len(rows)):components.setdefault(find(i),[]).append(i)
    connected=[members for members in components.values() if len(members)>1]
    mixed=sum(len({bool(rows[i]['objects']) for i in members})>1 for members in connected)
    return pairs,connected,mixed

def audit():
    import cv2
    from agent.dataset import validate_research_release
    from scripts.improve_pothole_data import dhash
    cv2.setNumThreads(1)
    folder=workspace_path(RELEASE,exists=True);manifest=folder/'release.json'
    original=json.loads(manifest.read_text(encoding='utf-8'));rows=[]
    checked=validate_research_release(original)
    if not checked['valid']:raise ValueError('research_release_validation_failed')
    if len(original)!=200 or any(row['split']!='test' for row in original):raise ValueError('frozen_blind_200_test_only_required')
    for item in original:
        row=dict(item);path=workspace_path(row['path'],exists=True)
        if not path.is_relative_to(folder/'source') or path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise ValueError('frozen_source_scope_required')
        if sha256(path)!=row['sha256']:raise ValueError('source_image_sha_mismatch')
        image=cv2.imread(str(path))
        if image is None or image.shape[:2]!=(row['image']['height'],row['image']['width']):raise ValueError('source_geometry_mismatch')
        if f'{dhash(image):016x}'!=row['perceptual_dhash']:raise ValueError('source_visual_identity_mismatch')
        block='RDD2022:'+filename_block(row['country'],row['original_member'])
        if row['leakage_group_id']!=block:raise ValueError('filename_group_identity_mismatch')
        row['_flip_dhash']=dhash(cv2.flip(image,1));rows.append(row)
    pairs,components,mixed=near_components(rows)
    exact_duplicates=len(rows)-len({row['sha256'] for row in rows})
    duplicate_groups=len(rows)-len({row['leakage_group_id'] for row in rows})
    counts=Counter((row['country'],bool(row['objects'])) for row in rows)
    balanced=len(counts)==4 and set(counts.values())=={50}
    result={'format':'RDD2022-blind-v7-pixel-QC-v1','release_sha256':sha256(manifest),'qc_source_sha256':sha256(Path(__file__)),
            'samples':len(rows),'countries':dict(Counter(row['country'] for row in rows)),
            'positive_images':sum(bool(row['objects']) for row in rows),'negative_images':sum(not row['objects'] for row in rows),
            'distinct_exact_hashes':len(rows)-exact_duplicates,'distinct_filename_groups':len(rows)-duplicate_groups,
            'near_pair_count':len(pairs),'near_component_count':len(components),'mixed_annotation_component_count':mixed,
            'all_image_bytes_and_visual_identities_verified':True,'model_inferences':0,
            'expires_at':min(aware(row['expires_at']) for row in rows).isoformat(),
            'valid':balanced and exact_duplicates==duplicate_groups==len(pairs)==0}
    atomic_json(folder/'within-cohort-qc-private.json',{**result,'near_pairs_sample_ids':[[rows[a]['sample_id'],rows[b]['sample_id']] for a,b in pairs]})
    return result

if __name__=='__main__':
    result=audit();print(json.dumps(result));sys.exit(0 if result['valid'] else 1)
