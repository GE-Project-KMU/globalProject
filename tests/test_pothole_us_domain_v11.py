import copy,hashlib,io,json,random,uuid,zipfile
from datetime import datetime,timedelta,timezone
from pathlib import Path
import cv2,numpy as np,pytest
from agent.dataset import validate_research_release
from scripts import prepare_pothole_us_domain_v11 as m

def xml(label='D40',coords=(0,1,30,31),filename='United_States_000001.jpg'):
 box=''.join(f'<{k}>{v}</{k}>' for k,v in zip(['xmin','ymin','xmax','ymax'],coords))
 return f'<annotation><filename>{filename}</filename><size><width>32</width><height>32</height></size><object><name>{label}</name><bndbox>{box}</bndbox></object></annotation>'.encode()

def test_only_literal_D40_foreground_all_other_valid_known_classes_are_not_potholes():
 assert m.annotation(xml(),'United_States_1.jpg')['objects'][0]['bbox_xywh_px']==[0.,1.,30.,30.]
 assert m.annotation(xml('D00'),'United_States_1.jpg')['objects']==[]
 assert m.annotation(xml('D20'),'United_States_1.jpg')['objects']==[]

@pytest.mark.parametrize('label,coords',[('D99',(0,0,3,3)),('D40',(-1,0,3,3)),('D00',(0,0,33,3)),('D40',(3,0,3,3)),('D40',(0,0,float('nan'),3)),('D10',(0,0,3,float('inf')))])
def test_no_unknown_or_invalid_box_is_clipped_or_silently_ignored(label,coords):
 with pytest.raises(ValueError):m.annotation(xml(label,coords),'United_States_1.jpg')

def test_XML_identity_entities_and_missing_labels_are_fail_closed():
 with pytest.raises(ValueError):m.annotation(xml(filename='United_States_2.jpg'),'United_States_1.jpg')
 with pytest.raises(ValueError):m.annotation(b'<!DOCTYPE x><annotation/>','United_States_1.jpg')
 stream=io.BytesIO()
 with zipfile.ZipFile(stream,'w') as z:
  z.writestr('United_States/train/images/United_States_1.jpg',b'JPEG')
  z.writestr('United_States/test/images/United_States_2.jpg',b'UNANNOTATED TEST')
 with zipfile.ZipFile(stream) as z:assert m.paired_members(z)==([],1)

def valid_receipt():return {'status':'verified_archive_structure_CRC','usable_source':True,'all_members_CRC_verified':True,'md5':m.prior.MD5,'bytes':m.prior.SIZE,'sha256':'a'*64,'source_expires_at':m.prior.EXPIRY}
@pytest.mark.parametrize('field,value',[('status','MD5_only'),('usable_source',False),('all_members_CRC_verified',False),('md5','0'*32),('bytes',1),('sha256','bad'),('source_expires_at','2026-11-11T17:07:24.286159+00:00')])
def test_whole_verified_archive_gate_rejects_each_missing_or_changed_evidence(field,value):
 row=valid_receipt();row[field]=value
 with pytest.raises(ValueError):m.validate_transfer_receipt(row)

def test_full_receipt_and_exact_deadline_contract():m.validate_transfer_receipt(valid_receipt())

def test_blocked_feasibility_policy_cannot_be_overridden_by_authorized_flag(monkeypatch):
 monkeypatch.setattr(m,'policy',lambda:{'execution_blocked':True})
 def forbidden(*args):raise AssertionError('blocked policy must not consume source')
 monkeypatch.setattr(m,'read',forbidden)
 with pytest.raises(ValueError,match='feasibility_blocked'):m.stage(True)
 with pytest.raises(ValueError,match='feasibility_blocked'):m.materialize('unavailable',True)

@pytest.mark.parametrize('action',['stage','materialize'])
def test_no_permission_never_reaches_metadata_archive_or_pixels(monkeypatch,action):
 def forbidden():raise AssertionError('data operation reached before permission')
 monkeypatch.setattr(m,'gate',forbidden)
 with pytest.raises(ValueError,match='authorization'):
  m.stage() if action=='stage' else m.materialize('none')

def test_US_normalization_all_split_blacklist_and_old_TEST_twenty_frame_guard():
 rows=[{'sha256':'a'*64,'split':'train','source_member':'RDD2022/United_States/train/images/United_States_00100.jpg'},
       {'sha256':'b'*64,'split':'test','original_member':'United_States_00200.jpeg'}]
 b=m.history_blacklist(rows)
 assert m.member_identity('United_States/train/images/United_States_000100.jpg')[1]==m.member_identity('United_States_100.jpeg')[1]
 for number,digest in [(150,'q'*64),(220,'q'*64),(999,'a'*64)]:
  _,member,base=m.member_identity(f'United_States_{number}.jpg');assert m.blocked_identity(number,member,base,digest,b)
 _,member,base=m.member_identity('United_States_1000.jpg');assert not m.blocked_identity(1000,member,base,'q'*64,b)

def test_pHash_is_genuine_DCT_and_horizontal_flip_invariant_pair():
 image=np.random.default_rng(7).integers(0,256,(71,93,3),np.uint8)
 values=m.phashes(image)
 assert values==list(reversed(m.phashes(cv2.flip(image,1))))
 assert all(0<=v<2**63 for v in values)
 tree=m.Tree();tree.add(0,'near');tree.add((1<<12)-1,'far')
 assert tree.query((1<<6)-1)==['near','far']
 assert tree.query((1<<7)-1)==['far']

def row(i,frame=None,split='train',positive=True,group=None):
 rng=random.Random(i);h=rng.getrandbits(63)
 return {'sample_id':str(uuid.UUID(int=i+1)),'sha256':hashlib.sha256(str(i).encode()).hexdigest(),'frame_id':i*100 if frame is None else frame,
  '_phashes':[h,h],'objects':[{'class':'pothole'}] if positive else [],'split':split,'leakage_group_id':group or f'g{i//20}','sequence_group_id':group or f'g{i//20}'}

def test_near_and_sequence_DSU_before_group_split_history_hits_propagate():
 rows=[row(i) for i in range(30)];rows[0]['_phashes']=[0,0];rows[1]['_phashes']=[3,3];rows[1]['_history_hit']=True
 rows[2]['frame_id']=199;rows[2]['_phashes']=[0x333333333333333,0x333333333333333]
 chosen,qc=m.group_and_split(rows)
 assert {rows[i]['sample_id'] for i in [0,1,2]}.isdisjoint({r['sample_id'] for r in chosen})
 assert qc['history_connected_images_dropped']>=3
 assert m.group_and_split(rows)==(chosen,qc)
 m.verify_cross_split(chosen)

def test_guardband_drops_both_sides_without_moving_assignments():
 rows=[row(i,frame=100*i+99) for i in range(120)]
 before,_=m.group_and_split(rows)
 # Find two adjacent components with different predetermined assignments.
 by={r['sample_id']:r for r in before};pair=next((i,i+1) for i in range(119) if by[rows[i]['sample_id']]['split']!=by[rows[i+1]['sample_id']]['split'])
 left,right=pair;rows[left]['frame_id']=100*left+99;rows[right]['frame_id']=100*(left+1)
 after,qc=m.group_and_split(rows);ids={r['sample_id'] for r in after}
 assert rows[left]['sample_id'] not in ids and rows[right]['sample_id'] not in ids
 assert qc['boundary_dropped_images']>=2
 assert all(by[r['sample_id']]['split']==r['split'] for r in after)

def supported_rows():
 return [row(i+10000*s,split=split,positive=i%2==0,group=f'{split}-{i%7}') for s,split in enumerate(['train','validation','test']) for i in range(240 if split=='train' else 130)]

def test_fixed_caps_counts_hash_order_minimum_groups_and_no_input_mutation():
 rows=supported_rows();before=copy.deepcopy(rows);chosen,counts=m.select(rows)
 assert rows==before and counts['train']['positive']==counts['train']['negative']==120
 assert counts['validation']['positive']==counts['test']['negative']==50
 for split in ['validation','test']:
  for positive in [True,False]:
   expected=sorted((r for r in rows if r['split']==split and bool(r['objects'])==positive),key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())[:50]
   assert [r['sample_id'] for r in chosen if r['split']==split and bool(r['objects'])==positive]==[r['sample_id'] for r in expected]
 with pytest.raises(ValueError,match='support'):m.select([r for r in rows if not(r['split']=='test' and not r['objects'])])
 for r in rows:
  if r['split']=='test':r['leakage_group_id']='single'
 with pytest.raises(ValueError,match='components'):m.select(rows)

@pytest.mark.parametrize('kind',['group','exact','near'])
def test_final_cross_split_guard_rejects_every_kind(kind):
 a,b=row(1),row(2,split='test')
 if kind=='group':b['sequence_group_id']=a['sequence_group_id']
 elif kind=='exact':b['sha256']=a['sha256']
 else:b['_phashes']=a['_phashes']
 with pytest.raises(ValueError,match='leakage'):m.verify_cross_split([a,b])

def actual_fixture(tmp_path):
 folder=tmp_path/'new';folder.mkdir();identifier=str(uuid.uuid4());image=np.random.default_rng(13).integers(0,256,(32,32,3),np.uint8);ok,encoded=cv2.imencode('.jpg',image);assert ok
 raw=encoded.tobytes();decoded=cv2.imdecode(encoded,cv2.IMREAD_COLOR);digest=hashlib.sha256(raw).hexdigest();expiry=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
 license=tmp_path/'license.json';license.write_text(json.dumps({'research_allowed':True,'license_id':'synthetic-only','source_url':'https://example.invalid/synthetic','license_url':'https://example.invalid/license'}),encoding='utf-8')
 r={'sample_id':identifier,'source_id':identifier,'source_hash':digest,'sha256':digest,'leakage_group_id':'synthetic','annotation_version':'synthetic','split':'train',
  'path':str(folder/'source'/(identifier+'.jpg')),'image':{'width':32,'height':32},'_phashes':m.phashes(decoded),'objects':[],'expires_at':expiry,
  'privacy_status':'restricted_research','qc_status':'publisher_annotations_machine_checked','release_scope':'offline_research','human_privacy_reviewed':False,'not_for_deployment':True,
  'allowed_use':'Synthetic fixture only','license_evidence_path':str(license),'license_evidence_sha256':hashlib.sha256(license.read_bytes()).hexdigest()}
 return folder,r,raw,expiry

def test_real_file_backed_research_validation_occurs_after_UUID_bytes_are_written(tmp_path):
 folder,r,raw,expiry=actual_fixture(tmp_path)
 assert not validate_research_release([r])['valid']
 assert m.materialize_rows([r],lambda _:raw,folder,expiry)['valid']
 assert validate_research_release([r])['valid'] and Path(r['path']).read_bytes()==raw
 with pytest.raises(FileExistsError):m.materialize_rows([r],lambda _:raw,folder,expiry)

def test_real_validation_failure_preserves_bytes_and_publishes_no_success(tmp_path):
 folder,r,raw,expiry=actual_fixture(tmp_path);r['license_evidence_sha256']='0'*64
 with pytest.raises(ValueError,match='file_backed'):m.materialize_rows([r],lambda _:raw,folder,expiry)
 assert Path(r['path']).exists() and not (folder/'release.json').exists()

def test_UUID_path_escape_and_expired_source_cannot_write_images(tmp_path):
 folder,r,raw,expiry=actual_fixture(tmp_path);r['path']=str(tmp_path/'outside.jpg')
 with pytest.raises(ValueError,match='UUID_source'):m.materialize_rows([r],lambda _:raw,folder,expiry)
 assert not (tmp_path/'outside.jpg').exists()
 with pytest.raises(ValueError,match='expired'):m.alive('2020-01-01T00:00:00+00:00')
 with pytest.raises(ValueError,match='aware'):m.alive('2030-01-01T00:00:00')

def test_header_bounds_reject_oversized_pixels_before_decoder(monkeypatch):
 raw=b'\x89PNG\r\n\x1a\n'+b'\0\0\0\rIHDR'+(16384).to_bytes(4,'big')*2
 def forbidden(*args):raise AssertionError('must reject before allocating decoded image')
 monkeypatch.setattr(cv2,'imdecode',forbidden)
 with pytest.raises(ValueError,match='pixel_budget'):m.decode(raw)
 with pytest.raises(ValueError,match='header'):m.image_header_size(b'not an image')

@pytest.mark.parametrize('valid',[True,False])
def test_actual_synthetic_archive_materialization_publishes_bundle_only_after_validation(tmp_path,monkeypatch,valid):
 folder,r,raw,expiry=actual_fixture(tmp_path)
 policy=tmp_path/'policy.json';policy.write_text('{}',encoding='utf-8')
 member='United_States/train/images/United_States_1.jpg';label='United_States/train/annotations/xmls/United_States_1.xml'
 annotation=xml('D00');r.update(m.annotation(annotation,'United_States_1.jpg'));r['sequence_group_id']='synthetic-block'
 nested_bytes=io.BytesIO()
 with zipfile.ZipFile(nested_bytes,'w') as z:z.writestr(member,raw);z.writestr(label,annotation)
 with zipfile.ZipFile(nested_bytes) as z:r.update(original_member=member,annotation_member=label,publisher_image_crc32=z.getinfo(member).CRC,publisher_annotation_crc32=z.getinfo(label).CRC)
 outer=tmp_path/'synthetic-outer.zip'
 with zipfile.ZipFile(outer,'w') as z:z.writestr('RDD2022/United_States.zip',nested_bytes.getvalue())
 if not valid:r['license_evidence_sha256']='0'*64
 plan={'source_sha256':m.sha256(Path(m.__file__)),'policy_sha256':m.sha256(policy),'bindings':{},'rows':[r],'rows_sha256':m.canonical_digest([r]),'counts':{'synthetic_fixture_rows':1}}
 m.new_json(folder/'membership-plan.json',plan)
 monkeypatch.setattr(m,'DEST',folder);monkeypatch.setattr(m,'POLICY',policy);monkeypatch.setattr(m.acquire,'ARCHIVE',outer)
 # Synthetic admission adapter only; real archive gating is independently tested.
 monkeypatch.setattr(m,'gate',lambda:({},[],{}))
 if valid:
  result=m.materialize(m.sha256(folder/'membership-plan.json'),True)
  bundle=folder/'validated-release-bundle';assert result['status']=='research_release_validated'
  assert {p.name for p in bundle.iterdir()}=={'release.json','development.json','fresh100.json'}
  released=json.loads((bundle/'release.json').read_text(encoding='utf-8'));assert validate_research_release(released)['valid']
  assert not (folder/'validated-release-bundle.incomplete').exists()
 else:
  with pytest.raises(ValueError,match='file_backed'):m.materialize(m.sha256(folder/'membership-plan.json'),True)
  assert (folder/'incomplete-materialization.json').exists() and Path(r['path']).exists()
  assert not (folder/'validated-release-bundle').exists() and not (folder/'materialization-completed.json').exists()
