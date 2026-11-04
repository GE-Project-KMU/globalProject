"""Official-publisher range acquisition for a NEW blind external test only.

No training/inference, no old source mutation and no retention extension. A
partial range acquisition never claims the complete 13GB publisher MD5 passed.
"""
from __future__ import annotations
import argparse,hashlib,io,json,os,re,sys,time,uuid,zipfile
from collections import OrderedDict,Counter,defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx
from agent.paths import workspace_path,atomic_json,sha256

API='https://api.figshare.com/v2/articles/21431547'
URL='https://ndownloader.figshare.com/files/38030910'
SOURCE='https://figshare.com/articles/dataset/21431547/1'
SIZE=13_264_172_619
OUTER_MD5='b62bd51d2ffcfaa76c60f234f0cc2bb3'
DEST='data/research/rdd2022-blind-v7'
RELEASE='data/training/research-pothole-rdd2022-blind-v7'
COUNTRIES=('United_States','Norway','China_MotorBike','China_Drone','Japan','India','Czech')
PROTOCOL={'format':'RDD2022-new-blind-selection-v1','seed':20261107,'country_preference':list(COUNTRIES),
          'target_countries':2,'positive_per_country':50,'negative_per_country':50,'publisher_positive':'literal VOC object name D40',
          'publisher_negative':'no literal D40; not human-confirmed clean road','one_frame_per_country_filename20_block':True,
          'all_historical_splits_excluded':True,'exact_and_connected_dhash_hamming_le3_with_horizontal_flip':True,
          'training_or_threshold_selection_use':False,'model_predictions_before_selection':0,
          'route_or_session_independence_verified':False,'physical_route_claim':'Filename blocks are proxies only; no verified route/session metadata.'}

def safe_member(name):
    path=PurePosixPath(name)
    if not name or name.startswith('/') or '\\' in name or ':' in name or '..' in path.parts:raise ValueError('unsafe_publisher_archive_member')
    return path

def scoped_output(value):
    lexical=Path(value)
    if '..' in lexical.parts:raise ValueError('parent_path_forbidden')
    lexical=lexical if lexical.is_absolute() else ROOT/lexical
    for component in [lexical,*lexical.parents]:
        if component==ROOT:break
        if not component.is_relative_to(ROOT):raise ValueError('workspace_scope_required')
        if component.is_symlink() or getattr(component,'is_junction',lambda:False)():raise ValueError('source_links_forbidden')
    path=workspace_path(lexical)
    if not path.is_relative_to(ROOT/'data/research') or path==ROOT/'data/research':raise ValueError('new_research_source_root_required')
    return path

def aware(value):
    date=datetime.fromisoformat(value.replace('Z','+00:00'))
    if date.tzinfo is None:raise ValueError('original_aware_expiry_required')
    return date.astimezone(timezone.utc)

def inherited_expiry(now=None):
    now=now or datetime.now(timezone.utc);dates=[now+timedelta(days=7)];proofs=[]
    for base in ('data/research/rdd2022','data/research/rdd2022-official-v5'):
        folder=workspace_path(base)
        if not folder.exists():continue
        for path in sorted(folder.glob('*.json')):
            value=json.loads(path.read_text(encoding='utf-8'))
            if isinstance(value,dict) and value.get('expires_at'):
                dates.append(aware(value['expires_at']));proofs.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path),'expires_at':value['expires_at']})
    return min(dates).isoformat(),proofs

class OfficialRange(io.RawIOBase):
    def __init__(self,*,client=None,block=4*1024*1024):
        self.client=client or httpx.Client(follow_redirects=True,timeout=60)
        self.owned_client=client is None;self.block=block;self.position=0;self.size=SIZE;self.cache=OrderedDict();self.network_bytes=0;self.range_responses=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        value=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if value<0:raise ValueError('negative_or_invalid_seek')
        self.position=value;return value
    def fetch(self,start,end):
        expected=end-start+1
        for attempt in range(5):
            try:
                url=URL+'?cache='+str(time.time_ns())
                with self.client.stream('GET',url,headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache','Pragma':'no-cache'}) as response:
                    response.raise_for_status()
                    for item in [*response.history,response]:
                        host=item.url.host or ''
                        if item.url.scheme!='https' or not (host.endswith('.figshare.com') or host.endswith('.amazonaws.com')):raise ValueError('unexpected_publisher_redirect_host')
                    if response.status_code!=206 or response.headers.get('content-range')!=f'bytes {start}-{end}/{self.size}':raise ValueError('exact_content_range_required')
                    result=bytearray()
                    for chunk in response.iter_bytes(1024*1024):
                        result.extend(chunk)
                        if len(result)>expected:raise ValueError('oversized_range_response')
                    if len(result)!=expected:raise ValueError('truncated_range_response')
                self.network_bytes+=len(result);self.range_responses+=1
                return bytes(result)
            except (httpx.HTTPError,ValueError):
                if attempt==4:raise
                time.sleep(min(4,attempt+1))
    def read(self,size=-1):
        end=self.size if size<0 else min(self.size,self.position+size);parts=[]
        while self.position<end:
            number=self.position//self.block;start=number*self.block
            if number not in self.cache:
                self.cache[number]=self.fetch(start,min(self.size-1,start+self.block-1))
                while len(self.cache)>4:self.cache.popitem(last=False)
            data=self.cache[number];local=self.position-start;count=min(end-self.position,len(data)-local)
            parts.append(data[local:local+count]);self.position+=count
        return b''.join(parts)
    def close(self):
        if self.owned_client:self.client.close()
        super().close()

class Segment(io.RawIOBase):
    def __init__(self,source,start,size):self.source,self.start,self.size,self.position=source,start,size,0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        value=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if value<0:raise ValueError('negative_or_invalid_segment_seek')
        self.position=value;return value
    def read(self,size=-1):
        count=max(0,self.size-self.position) if size<0 else max(0,min(size,self.size-self.position))
        self.source.seek(self.start+self.position);data=self.source.read(count);self.position+=len(data);return data

def open_country(remote,outer,country):
    if country not in COUNTRIES:raise ValueError('explicit_country_required')
    info=outer.getinfo('RDD2022/'+country+'.zip');safe_member(info.filename)
    if info.compress_type!=zipfile.ZIP_STORED:raise ValueError('country_not_stored_requires_full_verified_download')
    remote.seek(info.header_offset);header=remote.read(30)
    import struct
    if header[:4]!=b'PK\x03\x04':raise ValueError('local_zip_header_required')
    name_len,extra_len=struct.unpack_from('<HH',header,26)
    start=info.header_offset+30+name_len+extra_len
    segment=Segment(remote,start,info.file_size)
    return zipfile.ZipFile(segment),{'outer_member':info.filename,'outer_member_crc32':info.CRC,'bytes':info.file_size,'stored_segment_start':start,'full_outer_md5_verified':False,'whole_country_crc_verified':False}

def source_folder(output=DEST):
    folder=scoped_output(output);folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'source.json'
    if receipt.exists():
        record=json.loads(receipt.read_text());expiry=aware(record['expires_at'])
        if expiry<=datetime.now(timezone.utc):raise ValueError('source_expired_no_redownload_extension')
        if record.get('source_url')!=SOURCE:raise ValueError('existing_source_contract_mismatch')
    else:
        expiry,proofs=inherited_expiry()
        if aware(expiry)<=datetime.now(timezone.utc):raise ValueError('inherited_RDD2022_source_expired')
        atomic_json(receipt,{'source_url':SOURCE,'acquisition_started_at':datetime.now(timezone.utc).isoformat(),'expires_at':expiry,'inherited_earlier_source_evidence':proofs,'capture_time':None,'capture_time_inferred':False,'full_outer_md5_verified':False,'human_privacy_reviewed':False})
    protocol=folder/'selection-protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=PROTOCOL:raise ValueError('frozen_blind_protocol_changed')
    if not protocol.exists():atomic_json(protocol,PROTOCOL)
    response=httpx.get(API,timeout=30);response.raise_for_status();publisher=response.json()
    archive=next(item for item in publisher['files'] if item['id']==38030910)
    if publisher['license']['name']!='CC BY 4.0' or archive['size']!=SIZE or archive['computed_md5']!=OUTER_MD5:raise ValueError('official_publisher_contract_changed')
    path=folder/'publisher.json'
    if not path.exists():atomic_json(path,publisher)
    evidence=folder/'license-evidence.json'
    if not evidence.exists():atomic_json(evidence,{'research_allowed':True,'license_id':'CC-BY-4.0','source_url':SOURCE,'license_url':'https://creativecommons.org/licenses/by/4.0/','attribution':'Arya et al. RDD2022. DOI 10.6084/m9.figshare.21431547.v1','code_license_does_not_replace_data_license':True,'human_privacy_reviewed':False,'not_for_deployment':True})
    return folder

def index(output=DEST):
    folder=source_folder(output)
    with OfficialRange() as remote:
        with zipfile.ZipFile(remote) as outer:
            entries=[{'name':x.filename,'bytes':x.file_size,'compressed_bytes':x.compress_size,'compression':x.compress_type,'crc32':x.CRC,'header_offset':x.header_offset} for x in outer.infolist()]
            atomic_json(folder/'outer-central-directory.json',{'publisher_outer_md5_expected':OUTER_MD5,'full_outer_md5_verified':False,'outer_size':SIZE,'entries':entries})
            print(json.dumps({'official_range_responses':remote.range_responses,'downloaded_bytes':remote.network_bytes,'countries':[{k:v for k,v in entry.items() if k!='header_offset'} for entry in entries if entry['name'].endswith('.zip')]}),flush=True)
    return folder

def filename_block(country,name):
    if country not in COUNTRIES:raise ValueError('known_country_required')
    match=re.fullmatch(re.escape(country)+r'_(\d+)\.(?:jpg|jpeg|png)',PurePosixPath(name).name,re.I)
    if not match:raise ValueError('original_country_filename_required')
    return country+':filename-block20:'+str(int(match[1])//20)

def parse_voc(raw,country,member):
    import xml.etree.ElementTree as ET
    if len(raw)>256*1024 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():raise ValueError('bounded_plain_publisher_XML_required')
    node=ET.fromstring(raw);name=node.findtext('filename') or PurePosixPath(member).with_suffix('.jpg').name
    if not PurePosixPath(name).suffix:name+='.jpg'
    block=filename_block(country,name)
    width,height=int(node.findtext('size/width')),int(node.findtext('size/height'))
    if not 1<=min(width,height)<=max(width,height)<=16384:raise ValueError('publisher_dimensions_invalid')
    objects=[];classes=Counter()
    for obj in node.findall('object'):
        label=obj.findtext('name');classes[label]+=1
        if label!='D40':continue
        import math
        box=obj.find('bndbox')
        x1,y1,x2,y2=[float(box.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
        if not all(math.isfinite(v) for v in (x1,y1,x2,y2)):raise ValueError('finite_publisher_bbox_required')
        left,top=max(0.,x1-1),max(0.,y1-1);right,bottom=min(float(width),x2),min(float(height),y2)
        if right<=left or bottom<=top:raise ValueError('publisher_D40_bbox_invalid')
        objects.append({'class':'pothole','bbox_xywh_px':[left,top,right-left,bottom-top],'certainty':'confirmed'})
    return {'filename':PurePosixPath(name).name,'group':block,'width':width,'height':height,'objects':objects,'publisher_classes':dict(classes),'annotation_sha256':hashlib.sha256(raw).hexdigest()}

def history_snapshot(exclude):
    from scripts.prepare_mwpd import historical_identity
    rows=[];paths=[];groups=set()
    for path in sorted((ROOT/'data/training').glob('research-pothole*/release.json')):
        if path.parent==exclude:continue
        value=json.loads(path.read_text(encoding='utf-8'));rows.extend(value)
        paths.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path),'samples':len(value)})
        for row in value:
            token=row.get('leakage_group_id','')
            match=re.search(r'(China_MotorBike|China_Drone|United_States|Norway|Japan|India|Czech):filename-block20:(\d+)',token)
            if match:groups.add(match[1]+':filename-block20:'+match[2])
            for key in ('original_member','source_member','source_url'):
                text=row.get(key,'')
                if not isinstance(text,str):continue
                for country in COUNTRIES:
                    match=re.search(re.escape(country)+r'_(\d+)\.(?:jpg|jpeg|png)',text,re.I)
                    if match:groups.add(country+':filename-block20:'+str(int(match[1])//20))
    if not rows:raise ValueError('complete_historical_development_and_test_exclusions_required')
    return rows,historical_identity(rows),groups,paths

def prepare(output=DEST,release=RELEASE):
    import random
    import cv2,numpy as np
    from agent.dataset import validate_research_release
    from scripts.improve_pothole_data import dhash
    from scripts.prepare_mwpd import isolate_components
    cv2.setNumThreads(1)
    folder=source_folder(output);release=workspace_path(release)
    if not release.is_relative_to(ROOT/'data/training') or release.exists():raise ValueError('new_blind_release_directory_required')
    receipt=json.loads((folder/'source.json').read_text());expiry=aware(receipt['expires_at'])
    if expiry<=datetime.now(timezone.utc):raise ValueError('original_source_expired')
    history,historical,old_groups,history_paths=history_snapshot(release)
    frozen_history=folder/'history-exclusions.json'
    if frozen_history.exists():
        if json.loads(frozen_history.read_text())['releases']!=history_paths:raise ValueError('historical_releases_changed_after_blind_preparation')
    else:atomic_json(frozen_history,{'releases':history_paths,'all_splits_excluded':True,'never_reenter_training':True})
    namespace=uuid.UUID('0d973d81-b774-4feb-9366-9435c30df267')
    candidates=[];buffers={};availability=[];class_counts={};country_snapshots=[];filtered=[];removed=Counter();supported=[]
    with OfficialRange() as remote:
        with zipfile.ZipFile(remote) as outer:
            for country in COUNTRIES:
                country_zip,binding=open_country(remote,outer,country)
                with country_zip as source:
                    infos=source.infolist();images={PurePosixPath(i.filename).name:i for i in infos if '/train/' in '/'+i.filename and PurePosixPath(i.filename).suffix.lower() in ('.jpg','.jpeg','.png')}
                    # Retain publisher hierarchy and inner CRCs in private data.
                    country_snapshot={'binding':binding,'entries':[{'name':i.filename,'size':i.file_size,'crc32':i.CRC,'compression':i.compress_type} for i in infos]}
                    snapshot=folder/(country+'-central-directory.json');atomic_json(snapshot,country_snapshot);country_snapshots.append({'path':snapshot.relative_to(ROOT).as_posix(),'sha256':sha256(snapshot)})
                    pools={True:defaultdict(list),False:defaultdict(list)};counts=Counter();xml_count=0
                    for info in infos:
                        if '/train/' not in '/'+info.filename or not info.filename.lower().endswith('.xml'):continue
                        safe_member(info.filename)
                        if not 0<info.file_size<=256*1024:raise ValueError('annotation_member_size_limit')
                        raw=source.read(info);item=parse_voc(raw,country,info.filename);xml_count+=1;counts.update(item['publisher_classes'])
                        if item['filename'] not in images:raise ValueError('original_publisher_image_pair_missing')
                        if item['group'] in old_groups:continue
                        pools[bool(item['objects'])][item['group']].append(item)
                    availability.append({'country':country,'positive_unseen_filename_groups':len(pools[True]),'negative_unseen_filename_groups':len(pools[False]),'publisher_XML_read_with_CRC':xml_count})
                    class_counts[country]=dict(counts)
                    print(json.dumps(availability[-1]),flush=True)
                    if min(len(pools[True]),len(pools[False]))<50:continue
                    rng=random.Random(PROTOCOL['seed']);chosen=[];reserved=set()
                    for positive in (True,False):
                        tokens=sorted(pools[positive]);rng.shuffle(tokens)
                        for token in tokens:
                            if token in reserved:continue
                            chosen.append(rng.choice(sorted(pools[positive][token],key=lambda x:x['filename'])));reserved.add(token)
                            if sum(bool(x['objects'])==positive for x in chosen)>=150:break
                    if min(sum(bool(x['objects'])==positive for x in chosen) for positive in (True,False))<50:continue
                    for item in chosen:
                        info=images[item['filename']];safe_member(info.filename)
                        if not 0<info.file_size<=24*1024*1024:raise ValueError('image_member_size_limit')
                        raw=source.read(info);image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                        if image is None or image.shape[:2]!=(item['height'],item['width']):raise ValueError('original_image_geometry_mismatch')
                        digest=hashlib.sha256(raw).hexdigest();identifier=str(uuid.uuid5(namespace,country+'/'+item['filename']))
                        candidates.append({'sample_id':identifier,'source_id':str(uuid.uuid5(namespace,'source:'+digest)),'source_hash':digest,'leakage_group_id':'RDD2022:'+item['group'],'split':'test','country':country,'original_member':info.filename,
                            'path':(release/'source'/(identifier+'.jpg')).relative_to(ROOT).as_posix(),'sha256':digest,'perceptual_dhash':f'{dhash(image):016x}','_flip_dhash':dhash(cv2.flip(image,1)),
                            'annotation_version':'RDD2022-official-VOC-literal-D40-blind-v7','annotation_sha256':item['annotation_sha256'],'publisher_inner_image_crc32':info.CRC,
                            'image':{'width':item['width'],'height':item['height']},'objects':item['objects'],
                            'privacy_status':'restricted_research','qc_status':'publisher_annotations_machine_checked','release_scope':'offline_research','human_privacy_reviewed':False,'not_for_deployment':True,'project_acceptance_eligible':False,'independent_route_verified':False,
                            'detection_labels_verified':True,'allowed_use':'CC BY 4.0 private offline evaluation only; no human privacy/deployment approval','expires_at':expiry.isoformat(),'source_expires_at':expiry.isoformat(),
                            'source_url':SOURCE,'license_evidence_path':(folder/'license-evidence.json').relative_to(ROOT).as_posix(),'license_evidence_sha256':sha256(folder/'license-evidence.json')})
                        buffers[identifier]=raw
                    filtered,removed=isolate_components(candidates,historical)
                    supported=[]
                    for name in COUNTRIES:
                        rows=[r for r in filtered if r['country']==name]
                        if min(sum(bool(r['objects'])==p for r in rows) for p in (True,False))>=50:supported.append(name)
                    print(json.dumps({'countries_with_safe_balanced_support':len(supported),'candidate_images':len(candidates),'downloaded_bytes':remote.network_bytes,'duplicate_exclusions':dict(removed)}),flush=True)
                    if len(supported)>=2:break
            final=[]
            for country in supported[:2]:
                for positive in (True,False):final.extend(sorted((row for row in filtered if row['country']==country and bool(row['objects'])==positive),key=lambda row:row['sample_id'])[:50])
            metadata={'availability':availability,'publisher_class_counts':class_counts,'official_range_responses':remote.range_responses,'downloaded_bytes':remote.network_bytes,'whole_outer_MD5_verified':False,'whole_country_CRC_verified':False,'selected_individual_inner_members_CRC_verified':True,'country_index_snapshots':country_snapshots,'removed':dict(removed) if candidates else {},'protocol_sha256':sha256(folder/'selection-protocol.json'),'history_snapshot_sha256':sha256(frozen_history),'expires_at':expiry.isoformat()}
            atomic_json(folder/'preparation-audit.json',metadata)
    if len(final)!=200:raise ValueError('insufficient_novel_balanced_two_country_blind_cohort')
    if expiry<=datetime.now(timezone.utc):raise ValueError('source_expired_before_blind_freeze')
    if any(sha256(workspace_path(p['path'],exists=True))!=p['sha256'] for p in history_paths):raise ValueError('historical_exclusions_changed_during_preparation')
    (release/'source').mkdir(parents=True)
    for row in final:workspace_path(row['path']).write_bytes(buffers[row['sample_id']])
    check=validate_research_release(final)
    if not check['valid']:raise ValueError(str(check['errors'][:3]))
    atomic_json(release/'release.json',final)
    atomic_json(release/'split-evidence.json',{**metadata,'samples':200,'counts':dict(Counter(row['country'] for row in final)),'positive_images':sum(bool(row['objects']) for row in final),'negative_images':sum(not row['objects'] for row in final),'test_predictions':0,'never_training_or_selection':True,'physical_route_independence_verified':False,'release_sha256':sha256(release/'release.json')})
    print(json.dumps({'status':'new_blind_cohort_frozen','samples':200,'positive_images':100,'negative_images':100,'countries':dict(Counter(row['country'] for row in final)),'source_expires_at':expiry.isoformat(),'release_sha256':sha256(release/'release.json'),'test_predictions':0}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['index','prepare']);parser.add_argument('--output',default=DEST);args=parser.parse_args()
    (index if args.action=='index' else prepare)(args.output)
