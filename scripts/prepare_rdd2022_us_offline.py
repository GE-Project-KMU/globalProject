"""Network-free feasibility of a distinct US-only blind fallback, never training.

The failed original two-country protocol remains unchanged. This first phase
only checks publisher bytes/labels and duplicate support; no release is written.
"""
from __future__ import annotations
import hashlib,io,json,random,sys,uuid,zipfile
from collections import OrderedDict,Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from scripts import prepare_rdd2022_blind as original
from scripts.acquire_rdd2022_blind_api import SELECTION_SHA,PROTOCOL_SHA
from scripts.audit_rdd2022_blind import near_components

OUTPUT='data/training/research-pothole-rdd2022-us-offline-v7'
NAMESPACE=uuid.UUID('0d973d81-b774-4feb-9366-9435c30df267')

class OfflineRange(original.OfficialRange):
    def __init__(self,folder=None,block=4*1024*1024):
        io.RawIOBase.__init__(self)
        self.folder=folder or original.scoped_output(original.DEST)
        self.expiry=original.aware(json.loads((self.folder/'source.json').read_text(encoding='utf-8'))['expires_at'])
        self.disk=self.folder/'range-cache'
        if self.disk.is_symlink() or getattr(self.disk,'is_junction',lambda:False)():raise ValueError('source_links_forbidden')
        self.block=block;self.size=original.SIZE;self.position=0;self.cache=OrderedDict();self.owned_client=False
        self.network_bytes=0;self.range_responses=0;self.verified_cached_blocks=set()
    def fetch(self,start,end):
        if datetime.now(timezone.utc)>=self.expiry:raise ValueError('original_source_expired')
        expected=end-start+1
        if start<0 or end<start or end>=self.size or expected>self.block:raise ValueError('bounded_cached_range_required')
        cache=self.disk/f'{start}-{end}.bin';proof=cache.with_suffix('.sha256')
        if cache.is_symlink() or proof.is_symlink():raise ValueError('source_links_forbidden')
        if not cache.is_file() or not proof.is_file():raise ValueError('required_verified_offline_range_missing')
        if cache.stat().st_size!=expected:raise ValueError('offline_range_size_mismatch')
        data=cache.read_bytes()
        if hashlib.sha256(data).hexdigest()!=proof.read_text(encoding='utf-8'):raise ValueError('offline_range_integrity_failure')
        self.verified_cached_blocks.add((start,end));return data

def component_representatives(rows):
    """Predefined data-QC rule: reject mixed labels; smallest stable UUID wins."""
    pairs,components,mixed=near_components(rows);covered={i for group in components for i in group}
    kept=[row for i,row in enumerate(rows) if i not in covered];removed=Counter()
    for group in components:
        if len({bool(rows[i]['objects']) for i in group})>1:removed['mixed_label_component']+=len(group)
        else:
            chosen=min(group,key=lambda i:rows[i]['sample_id']);kept.append(rows[chosen]);removed['same_label_component_redundancy']+=len(group)-1
    return kept,{'near_pairs':len(pairs),'near_components':len(components),'mixed_components':mixed,'removed':dict(removed)}

def feasibility():
    import cv2,numpy as np
    from scripts.improve_pothole_data import dhash
    from scripts.prepare_mwpd import isolate_components
    cv2.setNumThreads(1)
    folder=original.scoped_output(original.DEST)
    if sha256(Path(original.__file__))!=SELECTION_SHA or sha256(folder/'selection-protocol.json')!=PROTOCOL_SHA:raise ValueError('original_source_or_protocol_changed')
    history,historical,old_groups,history_paths=original.history_snapshot(workspace_path(OUTPUT))
    frozen=json.loads((folder/'history-exclusions.json').read_text(encoding='utf-8'))['releases']
    if history_paths!=frozen:raise ValueError('historical_exclusions_changed')
    candidates=[];buffers={};country='United_States'
    with OfflineRange(folder=folder) as remote:
        with zipfile.ZipFile(remote) as outer:
            country_zip,binding=original.open_country(remote,outer,country)
            with country_zip as source:
                infos=source.infolist();images={PurePosixPath(i.filename).name:i for i in infos if '/train/' in '/'+i.filename and PurePosixPath(i.filename).suffix.lower() in ('.jpg','.jpeg','.png')}
                pools={True:defaultdict(list),False:defaultdict(list)};xml_count=0
                for info in infos:
                    if '/train/' not in '/'+info.filename or not info.filename.lower().endswith('.xml'):continue
                    original.safe_member(info.filename)
                    if not 0<info.file_size<=256*1024:raise ValueError('annotation_member_size_limit')
                    item=original.parse_voc(source.read(info),country,info.filename);xml_count+=1
                    if item['filename'] not in images:raise ValueError('publisher_image_pair_missing')
                    if item['group'] not in old_groups:pools[bool(item['objects'])][item['group']].append(item)
                # Identical publisher-metadata candidate ordering to original v1.
                rng=random.Random(original.PROTOCOL['seed']);chosen=[];reserved=set()
                for positive in (True,False):
                    tokens=sorted(pools[positive]);rng.shuffle(tokens)
                    for token in tokens:
                        if token in reserved:continue
                        chosen.append(rng.choice(sorted(pools[positive][token],key=lambda x:x['filename'])));reserved.add(token)
                        if sum(bool(x['objects'])==positive for x in chosen)>=150:break
                for item in chosen:
                    info=images[item['filename']];original.safe_member(info.filename)
                    if not 0<info.file_size<=24*1024*1024:raise ValueError('image_member_size_limit')
                    raw=source.read(info);image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                    if image is None or image.shape[:2]!=(item['height'],item['width']):raise ValueError('original_image_geometry_mismatch')
                    digest=hashlib.sha256(raw).hexdigest();identifier=str(uuid.uuid5(NAMESPACE,country+'/'+item['filename']))
                    candidates.append({'sample_id':identifier,'source_id':str(uuid.uuid5(NAMESPACE,'source:'+digest)),
                                       'source_hash':digest,'sha256':digest,'split':'test','country':country,'original_member':info.filename,
                                       'leakage_group_id':'RDD2022:'+item['group'],'perceptual_dhash':f'{dhash(image):016x}',
                                       '_flip_dhash':dhash(cv2.flip(image,1)),'objects':item['objects'],
                                       'annotation_sha256':item['annotation_sha256'],'publisher_inner_image_crc32':info.CRC,
                                       'image':{'width':item['width'],'height':item['height']}})
                    buffers[identifier]=raw
        checked_blocks=len(remote.verified_cached_blocks)
    filtered,history_removed=isolate_components(candidates,historical)
    flip={row['sample_id']:row['_flip_dhash'] for row in candidates}
    for row in filtered:row['_flip_dhash']=flip[row['sample_id']]
    clean,near=component_representatives(filtered)
    counts=Counter('positive' if row['objects'] else 'negative' for row in clean)
    report={'format':'RDD2022-US-offline-feasibility-v1','diagnostic_source_sha256':sha256(Path(__file__)),
            'original_selection_source_sha256':SELECTION_SHA,'original_protocol_sha256':PROTOCOL_SHA,
            'status':'feasible' if min(counts.get('positive',0),counts.get('negative',0))>=50 else 'insufficient',
            'country':country,'publisher_XML_CRC_verified':xml_count,'candidate_image_CRC_SHA_verified':len(candidates),
            'all_original_US_candidates_recovered':len(candidates)==240,'verified_cached_blocks_read':checked_blocks,
            'history_removed':dict(history_removed),'near_QC':near,'safe_positive_images':counts.get('positive',0),
            'safe_negative_images':counts.get('negative',0),'distinct_safe_filename_groups':len({r['leakage_group_id'] for r in clean}),
            'safe_exact_hashes':len({r['sha256'] for r in clean}),'network_requests':0,'model_inferences':0,'release_created':False,
            'source_expires_at':json.loads((folder/'source.json').read_text(encoding='utf-8'))['expires_at']}
    atomic_json(folder/'offline-us-feasibility.json',{**report,'candidate_identity_rows':[{k:v for k,v in row.items() if k!='_flip_dhash'} for row in candidates]})
    return report

if __name__=='__main__':
    try:print(json.dumps(feasibility()))
    except BaseException as error:
        print(json.dumps({'status':'offline_feasibility_failed','error_type':type(error).__name__}),file=sys.stderr);sys.exit(1)
