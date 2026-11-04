"""Predeclared US publisher-TRAIN research split; execution needs separate approval.

Draft preparation is metadata-only. No downloading, model inference or training.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,random,re,stat,sys,uuid,zipfile
from collections import Counter,defaultdict,deque
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256
from scripts import probe_rdd2022_domain_v11 as prior
from scripts import acquire_rdd2022_domain_v11 as acquire

DEST=ROOT/'data/training/research-pothole-us-domain-v11'
POLICY=ROOT/'docs/reports/POTHOLE-US-DOMAIN-V11-DRAFT-POLICY.json'
HISTORY=ROOT/'data/research/rdd2022-blind-v7/history-exclusions.json'
US100=ROOT/'data/training/research-pothole-rdd2022-us100-v7/release.json'
LICENSE=ROOT/'data/research/rdd2022-blind-v7/license-evidence.json'
US100_SHA='592c83e5967073354031ade9b50eb0a24b7b2f65f33abf477fc3990611221703'
NAMESPACE=uuid.UUID('31f67152-826b-4a90-b755-b11169c165af')
COUNTRY='United_States';SEED=20261111;NEAR=6
CLASSES={'D00','D10','D20','D40'}
DEPENDENCIES=['agent/dataset.py','agent/paths.py','scripts/probe_rdd2022_domain_v11.py','scripts/acquire_rdd2022_domain_v11.py','scripts/audit_rdd2022_domain_v11.py']

def aware(value):
 d=datetime.fromisoformat(value.replace('Z','+00:00'))
 if d.tzinfo is None:raise ValueError('aware_source_deadline_required')
 return d.astimezone(timezone.utc)
def alive(value):
 if aware(value)<=datetime.now(timezone.utc):raise ValueError('original_source_expired')
def safe_path(value,*,exists=False):
 p=workspace_path(value,exists=exists)
 for parent in [p,*p.parents]:
  if parent==ROOT.parent:break
  if parent.is_symlink() or getattr(parent,'is_junction',lambda:False)():raise ValueError('source_or_output_links_refused')
 return p
def read(path):
 p=safe_path(path,exists=True)
 with p.open('rb') as f:raw=f.read(40_000_001)
 if len(raw)>40_000_000:raise ValueError('bounded_metadata_required')
 return json.loads(raw)
def new_json(path,value):
 p=safe_path(path)
 with p.open('x',encoding='utf-8') as f:
  json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush()
  import os
  os.fsync(f.fileno())
def canonical_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def pointer(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(safe_path(path,exists=True))}
def policy_value():
 return {'format':'pothole-US-domain-v11-predeclared-draft-v1','status':'feasibility_blocked_no_split_or_training_authorization',
  'execution_blocked':True,'feasibility':{'official_metadata_audit':pointer(prior.DEST/'annotation-audit-results.json'),
   'publisher_TRAIN_D40_positive_images':116,'prior_US100_positive_images_excluded':50,'new_positive_upper_bound_before_other_QC':66,
   'required_new_positive_minimum':200,'decision':'Insufficient before pixel/near/group QC; criteria unchanged; no stage/materialization/training.',
   'future_execution_requires_new_explicit_plan_not_this_blocked_draft':True},
  'source_sha256':sha256(Path(__file__)),'dependency_sha256':{n:sha256(ROOT/n) for n in DEPENDENCIES},
  'official_source':'https://doi.org/10.6084/m9.figshare.21431547.v1','license_id':'CC-BY-4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
  'publisher_MD5':prior.MD5,'archive_bytes':prior.SIZE,'requires_complete_archive_MD5_SHA_CRC_and_bounds':True,
  'history_snapshot':pointer(HISTORY),'history_release_count':6,'US100_release':pointer(US100),'US100_required_sha256':US100_SHA,
  'license_evidence':pointer(LICENSE),'source_expires_at':prior.EXPIRY,'expiry_extended':False,
  'publisher_pool':'United_States/train paired JPG and VOC XML only; unannotated publisher TEST is never negative',
  'GT':'literal D40 only; all known-class boxes must be finite and within declared dimensions; no clip/truncation; unknown class image rejected',
  'coordinate_convention':'publisher coordinates treated as continuous edges without guessed origin adjustment; stored xmin,ymin,xmax-xmin,ymax-ymin',
  'history_exclusion':'ALL splits of the exact frozen six releases plus US100; exact bytes, normalized publisher member and basename; old US frame-block100; old US TEST frame id +/-20; history-connected visual components removed as a whole',
  'visual_identity':'64-bit DCT pHash: gray32x32 INTER_AREA, cv2.dct upper8x8, median excluding DC, DC bit zero; original and horizontal flip, Hamming<=6',
  'sequence_group':'country + floor(numeric frame id / 100); near/exact duplicate DSU unions sequence groups',
  'split_seed':SEED,'group_assignment':'sorted DSU component identifiers shuffled with Python random.Random(seed); floor70% train, floor15% validation, remainder test',
  'boundary_rule':'drop BOTH sides of all different-split frame-id pairs separated by <=20; no reassignment',
  'selection':'per split/class sort SHA256(sample_id UTF8); TRAIN cap1500 each; validation/test exact50 each; no metric-dependent selection',
  'requirements':{'train_min_positive':100,'train_min_negative':100,'train_cap_each':1500,'validation_each':50,'test_each':50,'min_independent_components_each_split':5},
  'insufficient_support':'record metadata failure, no release; never relax/reassign/retry automatically',
  'materialization':'exclusive membership plan before UUID JPG writes; file-backed validation AFTER actual writes; atomic release bundle publish only on success',
  'physical_route_independence_verified':False,'research_only':True,'human_privacy_reviewed':False,'not_for_deployment':True,
  'actual_image_reads':0,'actual_splits':0,'actual_model_predictions':0,'training_updates':0,'execution_requires_separate_root_authorization':True}
def draft():
 value=policy_value()
 if value['US100_release']['sha256']!=US100_SHA:raise ValueError('unchanged_US100_required')
 new_json(POLICY,value);return {'status':'draft_only_no_data_operations','policy_sha256':sha256(POLICY)}
def policy():
 value=read(POLICY)
 if value!=policy_value():raise ValueError('frozen_policy_source_or_inputs_changed')
 return value
def validate_transfer_receipt(receipt):
 if (receipt.get('status')!='verified_archive_structure_CRC' or receipt.get('usable_source') is not True or receipt.get('all_members_CRC_verified') is not True or receipt.get('md5')!=prior.MD5 or receipt.get('bytes')!=prior.SIZE or receipt.get('source_expires_at')!=prior.EXPIRY or not re.fullmatch('[a-f0-9]{64}',str(receipt.get('sha256','')))):raise ValueError('whole_official_MD5_CRC_bounds_required')
 alive(receipt['source_expires_at'])
def gate():
 value=policy()
 if value.get('execution_blocked') is True:raise ValueError('declared_policy_feasibility_blocked_no_execution')
 receiptpath=prior.DEST/'full-transfer-results.json';receipt=read(receiptpath);validate_transfer_receipt(receipt)
 transfer=read(acquire.PLAN)
 if transfer.get('transport_source_sha256')!=sha256(ROOT/'scripts/acquire_rdd2022_domain_v11.py') or transfer.get('source_expires_at')!=prior.EXPIRY:raise ValueError('verified_transfer_source_plan_required')
 auditpath=prior.DEST/'annotation-audit-results.json';audit=read(auditpath);auditplan=read(prior.DEST/'annotation-audit-plan.json')
 if audit.get('plan_sha256')!=sha256(prior.DEST/'annotation-audit-plan.json') or audit.get('official_whole_MD5_verified') is not True or audit.get('whole_archive_CRC_verified') is not True or audit.get('archive_sha256')!=receipt['sha256'] or audit.get('source_expires_at')!=prior.EXPIRY or auditplan.get('audit_source_sha256')!=sha256(ROOT/'scripts/audit_rdd2022_domain_v11.py') or COUNTRY not in audit.get('countries',{}):raise ValueError('complete_bound_official_annotation_audit_required')
 p=safe_path(acquire.ARCHIVE,exists=True);before=p.stat()
 if before.st_size!=prior.SIZE or sha256(p)!=receipt['sha256']:raise ValueError('official_archive_SHA_size_changed')
 after=p.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('archive_changed_during_hash')
 history=read(HISTORY)
 if len(history.get('releases',[]))!=6 or history.get('all_splits_excluded') is not True or history.get('never_reenter_training') is not True:raise ValueError('frozen_six_ALLsplit_history_required')
 old=[];refs=[]
 for ref in [*history['releases'],value['US100_release']]:
  path=safe_path(ref['path'],exists=True)
  if sha256(path)!=ref['sha256']:raise ValueError('permanent_history_release_changed')
  rows=read(path)
  if not isinstance(rows,list) or not rows:raise ValueError('complete_history_required')
  old.extend(rows);refs.append(pointer(path))
 evidence=read(LICENSE)
 if evidence.get('license_id')!='CC-BY-4.0' or evidence.get('research_allowed') is not True:raise ValueError('publisher_license_evidence_required')
 return value,old,{'archive':pointer(p),'transfer_receipt':pointer(receiptpath),'transfer_plan':pointer(acquire.PLAN),'annotation_audit':pointer(auditpath),'annotation_audit_plan':pointer(prior.DEST/'annotation-audit-plan.json'),'history_releases':refs,'history_snapshot':pointer(HISTORY),'license_evidence':pointer(LICENSE)}
def member_identity(name):
 if not isinstance(name,str) or '\\' in name or ':' in name or '\x00' in name:raise ValueError('plain_publisher_member_required')
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('safe_relative_publisher_member_required')
 match=re.fullmatch(r'United_States_(\d+)\.(?:jpg|jpeg)',p.name,re.I)
 if not match:raise ValueError('literal_US_publisher_filename_required')
 number=int(match[1]);base='united_states_'+str(number)+'.jpg'
 return number,COUNTRY+':'+base,base
def annotation(raw,expected_filename):
 if len(raw)>2*1024**2 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():raise ValueError('bounded_plain_XML_required')
 node=ET.fromstring(raw);width=int(node.findtext('size/width','0'));height=int(node.findtext('size/height','0'))
 if not 0<width<=16384 or not 0<height<=16384:raise ValueError('valid_publisher_geometry_required')
 declared=node.findtext('filename','')
 if not declared or PurePosixPath(declared).name!=declared or member_identity(declared)[1]!=member_identity(expected_filename)[1]:raise ValueError('exact_publisher_image_XML_pair_required')
 objects=[];counts=Counter()
 for obj in node.findall('object'):
  label=(obj.findtext('name') or '').strip()
  if label not in CLASSES:raise ValueError('unknown_publisher_class')
  values=[float(obj.findtext('bndbox/'+key,'')) for key in ('xmin','ymin','xmax','ymax')];x0,y0,x1,y1=values
  if not all(math.isfinite(v) for v in values) or not 0<=x0<x1<=width or not 0<=y0<y1<=height:raise ValueError('every_publisher_bbox_must_be_valid_no_clipping')
  counts[label]+=1
  if label=='D40':objects.append({'class':'pothole','bbox_xywh_px':[x0,y0,x1-x0,y1-y0],'certainty':'confirmed'})
 return {'image':{'width':width,'height':height},'objects':objects,'publisher_classes':dict(counts),'annotation_sha256':hashlib.sha256(raw).hexdigest()}
def paired_members(archive):
 images={};labels={};unpaired=0
 for info in archive.infolist():
  path=PurePosixPath(info.filename)
  if info.is_dir():continue
  if '\\' in info.filename or path.is_absolute() or '..' in path.parts or ':' in info.filename or '\x00' in info.orig_filename or info.flag_bits&1 or stat.S_ISLNK(info.external_attr>>16):raise ValueError('safe_unencrypted_archive_member_required')
  if path.parts[:2]!=(COUNTRY,'train'):continue
  if path.suffix.lower() in ('.jpg','.jpeg') and path.parent==PurePosixPath(COUNTRY+'/train/images'):
   if not 0<info.file_size<=32*1024**2:raise ValueError('image_member_size_bound')
   key=member_identity(path.name)[1];target=images
  elif path.suffix.lower()=='.xml' and path.parent==PurePosixPath(COUNTRY+'/train/annotations/xmls'):
   if not 0<info.file_size<=2*1024**2:raise ValueError('annotation_member_size_bound')
   key=member_identity(path.with_suffix('.jpg').name)[1];target=labels
  else:continue
  if key in target:raise ValueError('duplicate_canonical_publisher_member')
  target[key]=info
 keys=sorted(images.keys()&labels.keys());unpaired=len(images.keys()^labels.keys())
 return [(images[k],labels[k]) for k in keys],unpaired
def phashes(image):
 import cv2,numpy as np
 if image is None or image.ndim!=3 or image.shape[2]!=3:raise ValueError('decoded_BGR_required')
 def one(source):
  gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY);small=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
  low=cv2.dct(small)[:8,:8].reshape(-1);bits=low>np.median(low[1:]);bits[0]=False
  result=0
  for bit in bits:result=(result<<1)|int(bit)
  return result
 return [one(image),one(cv2.flip(image,1))]
class Tree:
 def __init__(self):self.root=None
 def add(self,key,tag):
  if self.root is None:self.root=[key,[tag],{}];return
  node=self.root
  while True:
   distance=(key^node[0]).bit_count()
   if distance==0:node[1].append(tag);return
   if distance not in node[2]:node[2][distance]=[key,[tag],{}];return
   node=node[2][distance]
 def query(self,key):
  result=[];stack=[self.root] if self.root else []
  while stack:
   node=stack.pop();distance=(key^node[0]).bit_count()
   if distance<=NEAR:result.extend(node[1])
   stack.extend(child for edge,child in node[2].items() if distance-NEAR<=edge<=distance+NEAR)
  return result
def history_blacklist(rows):
 result={'sha':set(),'members':set(),'basenames':set(),'test_ids':set(),'sequence_blocks':set()}
 for row in rows:
  digest=row.get('sha256','')
  if not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('historical_image_SHA_required')
  result['sha'].add(digest)
  for field in ['original_member','source_member','source_id','path']:
   text=row.get(field,'')
   if not isinstance(text,str):continue
   result['basenames'].add(PurePosixPath(text.replace('\\','/')).name.lower())
   for name in re.findall(r'United_States_\d+\.(?:jpg|jpeg)',text,re.I):
    number,member,base=member_identity(name);result['members'].add(member);result['basenames'].add(base);result['sequence_blocks'].add(number//100)
    if row.get('split')=='test':result['test_ids'].add(number)
 return result
def blocked_identity(number,member,base,digest,history):
 return digest in history['sha'] or member in history['members'] or base in history['basenames'] or number//100 in history['sequence_blocks'] or any(abs(number-old)<=20 for old in history['test_ids'])
def decode(raw,expected_sha=None):
 import cv2,numpy as np
 if len(raw)>32*1024**2 or expected_sha and hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('bounded_image_SHA_required')
 # Bound declared geometry BEFORE allocating the decoded historical/source frame.
 width,height=image_header_size(raw)
 if not 0<width<=16384 or not 0<height<=16384 or width*height>32_000_000:raise ValueError('decoded_image_pixel_budget_exceeded')
 image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
 if image is None or image.shape[:2]!=(height,width):raise ValueError('bounded_decoded_image_required')
 return image
def image_header_size(raw):
 if raw[:8]==b'\x89PNG\r\n\x1a\n' and len(raw)>=24 and raw[12:16]==b'IHDR':return int.from_bytes(raw[16:20],'big'),int.from_bytes(raw[20:24],'big')
 if raw[:2]!=b'\xff\xd8':raise ValueError('bounded_JPEG_or_PNG_header_required')
 i=2
 while i<len(raw):
  if raw[i]!=255:raise ValueError('invalid_JPEG_header')
  while i<len(raw) and raw[i]==255:i+=1
  if i>=len(raw):break
  marker=raw[i];i+=1
  if marker in (0xd8,0xd9) or 0xd0<=marker<=0xd7:continue
  if i+2>len(raw):break
  length=int.from_bytes(raw[i:i+2],'big')
  if length<2 or i+length>len(raw):break
  if marker in (0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf):
   if length<8:break
   return int.from_bytes(raw[i+5:i+7],'big'),int.from_bytes(raw[i+3:i+5],'big')
  if marker==0xda:break
  i+=length
 raise ValueError('bounded_image_dimensions_missing')
def historical_tree(rows):
 tree=Tree();seen=set()
 for row in rows:
  if row['sha256'] in seen:continue
  alive(row['expires_at']);path=safe_path(row['path'],exists=True)
  if path.stat().st_size>32*1024**2:raise ValueError('historical_image_size_limit')
  image=decode(path.read_bytes(),row['sha256'])
  for value in phashes(image):tree.add(value,True)
  seen.add(row['sha256'])
 return tree
def group_and_split(rows):
 rows=copy.deepcopy(rows);parent=list(range(len(rows)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 sequence={};exact={};tree=Tree()
 for i,row in enumerate(rows):
  key=COUNTRY+':frame-block100:'+str(row['frame_id']//100);row['sequence_group_id']=key
  if key in sequence:union(i,sequence[key])
  sequence[key]=i
  if row['sha256'] in exact:union(i,exact[row['sha256']])
  exact[row['sha256']]=i
  for h in row['_phashes']:
   for old in tree.query(h):union(i,old)
  for h in row['_phashes']:tree.add(h,i)
 components=defaultdict(list)
 for i in range(len(rows)):components[find(i)].append(i)
 blocked={root for root,indices in components.items() if any(rows[i].get('_history_hit') is True for i in indices)}
 keys={root:'RDD2022-US-v11:'+canonical_digest(sorted({rows[i]['sequence_group_id'] for i in indices})) for root,indices in components.items() if root not in blocked}
 ordered=sorted(keys.values());random.Random(SEED).shuffle(ordered);n=len(ordered);a=int(n*.7);b=a+int(n*.15)
 assigned={k:'train' if i<a else 'validation' if i<b else 'test' for i,k in enumerate(ordered)}
 eligible=[]
 for i,row in enumerate(rows):
  if find(i) not in blocked:row['leakage_group_id']=keys[find(i)];row['split']=assigned[keys[find(i)]];eligible.append(row)
 history_drop=len(rows)-len(eligible);rows=eligible
 dropped=set();window=deque()
 for i in sorted(range(len(rows)),key=lambda i:rows[i]['frame_id']):
  while window and rows[i]['frame_id']-rows[window[0]]['frame_id']>20:window.popleft()
  for j in window:
   if rows[i]['split']!=rows[j]['split']:dropped.update([i,j])
  window.append(i)
 return [row for i,row in enumerate(rows) if i not in dropped],{'components_before_boundary_drop':n,'history_connected_images_dropped':history_drop,'boundary_dropped_images':len(dropped)}
def select(rows):
 selected=[];counts={}
 for split in ['train','validation','test']:
  chosen=[]
  for positive in [True,False]:
   pool=sorted((r for r in rows if r['split']==split and bool(r['objects'])==positive),key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())
   minimum=100 if split=='train' else 50
   if len(pool)<minimum:raise ValueError('insufficient_fixed_balanced_support_'+split)
   chosen.extend(pool[:1500 if split=='train' else 50])
  if len({r['leakage_group_id'] for r in chosen})<5:raise ValueError('insufficient_independent_components_'+split)
  counts[split]={'positive':sum(bool(r['objects']) for r in chosen),'negative':sum(not r['objects'] for r in chosen),'components':len({r['leakage_group_id'] for r in chosen})};selected.extend(chosen)
 verify_cross_split(selected);return selected,counts
def verify_cross_split(rows):
 exact={};groups={};tree=Tree()
 for row in rows:
  for key,cache in [(row['sha256'],exact),(row['leakage_group_id'],groups),(row['sequence_group_id'],groups)]:
   if key in cache and cache[key]!=row['split']:raise ValueError('cross_split_exact_or_group_leakage')
   cache[key]=row['split']
  for h in row['_phashes']:
   if any(s!=row['split'] for s in tree.query(h)):raise ValueError('cross_split_visual_leakage')
  for h in row['_phashes']:tree.add(h,row['split'])
def stage(authorized=False):
 if authorized is not True:raise ValueError('separate_pixel_split_authorization_required')
 value,old,bindings=gate()
 if DEST.exists():raise ValueError('new_experiment_only_no_retry')
 safe_path(DEST);DEST.mkdir(parents=True,exist_ok=False);removed=Counter();availability={}
 try:
  import cv2
  cv2.setNumThreads(1);history=history_blacklist(old);old_tree=historical_tree(old);rows=[]
  with zipfile.ZipFile(acquire.ARCHIVE) as outer:
   info=outer.getinfo('RDD2022/United_States.zip')
   with outer.open(info) as handle,zipfile.ZipFile(handle) as nested:
    pairs,unpaired=paired_members(nested);removed['unpaired_not_negative']=unpaired
    for imageinfo,xmlinfo in pairs:
     alive(prior.EXPIRY);number,member,base=member_identity(imageinfo.filename)
     if blocked_identity(number,member,base,'',history):removed['historical_name_or_TEST_neighbor']+=1;continue
     rawxml=nested.read(xmlinfo)
     try:parsed=annotation(rawxml,PurePosixPath(imageinfo.filename).name)
     except (ValueError,ET.ParseError):removed['invalid_annotation_whole_image']+=1;continue
     raw=nested.read(imageinfo);digest=hashlib.sha256(raw).hexdigest()
     if blocked_identity(number,member,base,digest,history):removed['historical_exact_SHA']+=1;continue
     image=decode(raw);h,w=image.shape[:2]
     if parsed['image']!={'width':w,'height':h}:removed['publisher_geometry_mismatch']+=1;continue
     hashes=phashes(image)
     history_hit=any(old_tree.query(h) for h in hashes)
     if history_hit:removed['direct_historical_visual_hit']+=1
     sample=str(uuid.uuid5(NAMESPACE,member+':'+digest));rows.append(dict(parsed,sample_id=sample,source_id=str(uuid.uuid5(NAMESPACE,member)),source_hash=digest,sha256=digest,
      original_member=imageinfo.filename,annotation_member=xmlinfo.filename,publisher_image_crc32=imageinfo.CRC,publisher_annotation_crc32=xmlinfo.CRC,frame_id=number,_phashes=hashes,_history_hit=history_hit))
  grouped,qc=group_and_split(rows)
  availability={split:{'positive':sum(r['split']==split and bool(r['objects']) for r in grouped),'negative':sum(r['split']==split and not r['objects'] for r in grouped),'components':len({r['leakage_group_id'] for r in grouped if r['split']==split})} for split in ['train','validation','test']}
  selected,counts=select(grouped)
  for row in selected:
   row.update(path=(DEST/'source'/(row['sample_id']+'.jpg')).relative_to(ROOT).as_posix(),annotation_version='RDD2022-US-domain-v11-literal-D40-edge-coordinates',
    country=COUNTRY,privacy_status='restricted_research',qc_status='publisher_annotations_machine_checked',release_scope='offline_research',human_privacy_reviewed=False,
    not_for_deployment=True,project_acceptance_eligible=False,independent_route_verified=False,detection_labels_verified=True,
    allowed_use='CC BY 4.0 private offline research; no privacy acceptance or deployment approval',expires_at=prior.EXPIRY,source_expires_at=prior.EXPIRY,
    source_url='https://doi.org/10.6084/m9.figshare.21431547.v1',license_evidence_path=LICENSE.relative_to(ROOT).as_posix(),license_evidence_sha256=sha256(LICENSE))
  # Recheck every original pointer after potentially lengthy image-only QC.
  for ref in [v for v in bindings.values() if isinstance(v,dict)]+bindings['history_releases']:
   if sha256(safe_path(ref['path'],exists=True))!=ref['sha256']:raise ValueError('input_changed_during_membership_QC')
  alive(prior.EXPIRY);plan={'format':'US-domain-v11-prematerialization-membership-v1','policy_sha256':sha256(POLICY),'source_sha256':sha256(Path(__file__)),
   'bindings':bindings,'source_expires_at':prior.EXPIRY,'rows':selected,'rows_sha256':canonical_digest(selected),'counts':counts,'removed':dict(removed),'QC':qc,
   'old_ALLsplit_visual_overlap':0,'cross_split_visual_overlap':0,'TEST_model_predictions':0,'training_updates':0,'physical_route_independence_verified':False}
  new_json(DEST/'membership-plan.json',plan)
  return {'status':'membership_frozen_no_materialization','counts':counts,'membership_sha256':sha256(DEST/'membership-plan.json')}
 except BaseException as error:
  new_json(DEST/'incomplete-membership.json',{'status':'failed','error_type':type(error).__name__,'reason':str(error)[:120],'removed':dict(removed),'available_before_cap':availability,'source_expires_at':prior.EXPIRY,'release_published':False});raise
def materialize_rows(rows,buffers,folder,expiry):
 """Reusable file-backed contract: never validate nonexistent planned images."""
 from agent.dataset import validate_research_release
 alive(expiry);safe_path(folder);source=folder/'source';source.mkdir(exist_ok=False)
 for row in rows:
  uuid.UUID(row['sample_id']);path=safe_path(row['path'])
  if path!=source/(row['sample_id']+'.jpg'):raise ValueError('exact_UUID_source_path_required')
  raw=buffers(row)
  if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('materialized_SHA_mismatch')
  with path.open('xb') as f:f.write(raw)
  image=decode(path.read_bytes(),row['sha256'])
  if row['image']!={'height':image.shape[0],'width':image.shape[1]} or phashes(image)!=row['_phashes']:raise ValueError('materialized_geometry_or_visual_identity_changed')
 alive(expiry);check=validate_research_release(rows)
 if not check['valid']:raise ValueError('file_backed_research_validation_failed')
 return check
def materialize(membership_sha,authorized=False):
 if authorized is not True:raise ValueError('separate_materialization_authorization_required')
 value,old,bindings=gate();planpath=DEST/'membership-plan.json';plan=read(planpath)
 if sha256(planpath)!=membership_sha or plan['source_sha256']!=sha256(Path(__file__)) or plan['policy_sha256']!=sha256(POLICY) or plan['bindings']!=bindings or canonical_digest(plan['rows'])!=plan['rows_sha256']:raise ValueError('frozen_membership_and_source_required')
 if (DEST/'release.json').exists() or (DEST/'source').exists():raise ValueError('exclusive_materialization_no_retry')
 rows=plan['rows'];verify_cross_split(rows)
 try:
  with zipfile.ZipFile(acquire.ARCHIVE) as outer:
   with outer.open('RDD2022/United_States.zip') as handle,zipfile.ZipFile(handle) as nested:
    def payload(row):
     alive(prior.EXPIRY);image=nested.getinfo(row['original_member']);xml=nested.getinfo(row['annotation_member'])
     if image.CRC!=row['publisher_image_crc32'] or xml.CRC!=row['publisher_annotation_crc32']:raise ValueError('publisher_member_CRC_changed')
     parsed=annotation(nested.read(xml),PurePosixPath(image.filename).name)
     if any(parsed[k]!=row[k] for k in parsed):raise ValueError('frozen_annotation_changed')
     return nested.read(image)
    materialize_rows(rows,payload,DEST,prior.EXPIRY)
  clean=[{k:v for k,v in row.items() if not k.startswith('_')} for row in rows]
  bundle=DEST/'validated-release-bundle';temporary=DEST/'validated-release-bundle.incomplete';temporary.mkdir(exist_ok=False)
  for name,data in [('release.json',clean),('development.json',[r for r in clean if r['split']!='test']),('fresh100.json',[r for r in clean if r['split']=='test'])]:new_json(temporary/name,data)
  alive(prior.EXPIRY);temporary.rename(bundle)
  new_json(DEST/'materialization-completed.json',{'status':'validated','membership_sha256':membership_sha,'counts':plan['counts'],'source_expires_at':prior.EXPIRY,'release_bundle':'validated-release-bundle','training_updates':0,'model_predictions':0})
  return {'status':'research_release_validated','counts':plan['counts']}
 except BaseException as error:
  new_json(DEST/'incomplete-materialization.json',{'status':'failed','error_type':type(error).__name__,'source_expires_at':prior.EXPIRY,'success_claimed':False,'original_sources_deleted':False});raise
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['draft','stage','materialize']);p.add_argument('--authorized',action='store_true');p.add_argument('--membership-sha256');a=p.parse_args()
 try:result=draft() if a.action=='draft' else stage(a.authorized) if a.action=='stage' else materialize(a.membership_sha256,a.authorized);print(json.dumps(result));return 0
 except Exception as error:print(json.dumps({'status':'US_domain_preparation_stopped','error_type':type(error).__name__}));return 1
if __name__=='__main__':raise SystemExit(main())
