"""Generated nonpersonal TTL, explicit approval copies and bound cleanup tests."""
from datetime import datetime,timezone,timedelta
import json,uuid
import cv2,numpy as np
import pytest
from agent.paths import ROOT,atomic_json,sha256
from agent import material_policy as policy,pipeline,workflow,governance
from agent.outbox import Outbox
from agent.retention import purge_local_material

def uid():return str(uuid.uuid4())
@pytest.fixture
def folder(tmp_path,monkeypatch):
    assert tmp_path.resolve().is_relative_to(ROOT)
    monkeypatch.setattr(governance,'REGISTRY',tmp_path/'registry');monkeypatch.setattr(governance,'REVOCATIONS',tmp_path/'revocations.json')
    return tmp_path

def package(folder,*,cap=None,captured=None):
    captured=captured or datetime.now(timezone.utc).isoformat()
    profile={'profile_id':uid(),'calibrated_by':'SYNTHETIC-TEST','calibration':{'image_size':[64,48],'height_m':.4,'pitch_deg':30.,'camera_matrix':[[100.,0,32],[0,100.,24],[0,0,1]],'distortion':[0.]*5,'homography':np.eye(3).tolist(),'roi_polygon_px':[[0,0],[63,0],[63,47],[0,47]],'visible_distance_m':[0.,2.4],'calibrated_at':datetime.now(timezone.utc).isoformat()},'measurement':{'bev_size':[64,48],'meters_per_pixel':.05,'lane_tracks':[{'lane_track_id':uid(),'color':'white','line_type':'solid','bev_x_range':[20,30],'outline_verified':True}]}}
    profile_path=folder/'profile.json';atomic_json(profile_path,profile)
    class Contract:
        def __init__(self,task):self.manifest={'task':task,'sha256':'a'*64,'production_approved':False}
        def predict(self,image):
            if self.manifest['task']=='lane':return np.zeros(image.shape[:2],np.uint8)
            if self.manifest['task']=='redaction':return []
            return [{'class':'pothole','bbox_xywh_px':[5.,5.,8.,8.],'confidence':.9}]
    run={'run_id':uid(),'started_at':captured,'conditions':{'daylight':True,'dry_asphalt':True}}
    if cap:run['source_expires_at']=cap
    samples=[{'frame_id':0,'frame':np.full((48,64,3),120,np.uint8),'captured_at':captured,'fixes':[]}]
    bundle={'version':'SYNTHETIC-TTL-TEST','config_hash':'b'*64,'weights_hashes':{task:'a'*64 for task in ('pothole','lane','redaction')},**{task:task for task in ('pothole','lane','redaction')}}
    result=workflow.analyze_samples(samples,folder/'analysis',run=run,profile_path=profile_path,device_id=uid(),bundle=bundle,model_factory=Contract)
    path=folder/'analysis/review.json';item=result['items'][0]
    decisions=folder/'decisions.json';atomic_json(decisions,[{'observation_id':item['observation']['observation_id'],'approve':True,'image_sha256':item['observation']['image']['sha256']}])
    return result,path,decisions

@pytest.mark.parametrize('context',[{}, {'source_expires_at':'2099-01-01T00:00:00'}, {'source_expires_at':'2000-01-01T00:00:00Z'}, {'source_captured_at':'2099-01-01T00:00:00Z'}])
def test_bad_or_missing_source_context_rejected_before_models_or_pixels(folder,monkeypatch,context):
    monkeypatch.setattr(pipeline,'create_models',lambda *a,**k:pytest.fail('model loaded before source validation'))
    monkeypatch.setattr(pipeline,'replay',lambda *a,**k:pytest.fail('pixels read before source validation'))
    with pytest.raises(ValueError):pipeline.analyze_video('not-opened',folder/'out',pothole_manifest='none',lane_manifest='none',redaction_manifest='none',**context)
    assert not (folder/'out').exists()

def test_capture_future_and_expired_source_are_rejected(folder):
    with pytest.raises(ValueError,match='captured_at_in_future'):package(folder,captured=(datetime.now(timezone.utc)+timedelta(seconds=10)).isoformat())
    with pytest.raises(ValueError,match='source_material_expired'):package(folder,cap='2000-01-01T00:00:00Z')

def test_unreviewed_seven_days_approved_separate_copy_ninety_and_cleanup(folder):
    result,path,decisions=package(folder)
    captured=policy.instant(result['items'][0]['observation']['captured_at'])
    assert policy.instant(result['local_material']['expires_at'])==captured+timedelta(days=7)
    original=ROOT/result['items'][0]['image_path']
    workflow.approve_review(path,decisions,reviewed_by='SYNTHETIC-ONLY',queue_path=folder/'queue.sqlite',key_path=folder/'key')
    queue=Outbox(folder/'queue.sqlite',folder/'key')
    with queue.connect() as db:item=queue.decode(db.execute('SELECT * FROM queue').fetchone())
    approved=policy.safe_file(item['image_path'],PathParent(item['image_path']))
    assert approved!=original and approved.read_bytes()==original.read_bytes()
    marker=json.loads((approved.parent/'review.json').read_text())
    assert policy.instant(marker['local_material']['expires_at'])==captured+timedelta(days=90)
    purge_local_material(work=folder/'no-work',results=folder/'analysis',training=folder/'no-training',now=captured+timedelta(days=8),dry_run=False)
    assert not original.exists() and not path.exists() and approved.exists()

def PathParent(value):
    from pathlib import Path
    return Path(value).parent

def test_explicit_cap_is_inherited_by_approved_copy_and_encrypted_queue(folder):
    cap=(datetime.now(timezone.utc)+timedelta(days=2)).isoformat()
    result,path,decisions=package(folder,cap=cap)
    assert result['local_material']['expires_at']==cap
    workflow.approve_review(path,decisions,reviewed_by='SYNTHETIC-ONLY',queue_path=folder/'queue.sqlite',key_path=folder/'key')
    queue=Outbox(folder/'queue.sqlite',folder/'key')
    with queue.connect() as db:item=queue.decode(db.execute('SELECT * FROM queue').fetchone())
    assert item['source_expires_at']==cap
    assert json.loads((PathParent(item['image_path'])/'review.json').read_text())['local_material']['expires_at']==cap
    assert queue.purge_expired(now=policy.instant(cap)+timedelta(seconds=1))['expired']

def test_expired_unreviewed_package_cannot_be_approved_late(folder,monkeypatch):
    result,path,decisions=package(folder)
    expires=policy.instant(result['local_material']['expires_at'])
    monkeypatch.setattr(policy,'current',lambda:expires)
    with pytest.raises(ValueError,match='expired'):workflow.approve_review(path,decisions,reviewed_by='SYNTHETIC',queue_path=folder/'queue.sqlite',key_path=folder/'key')
    assert not (folder/'analysis/approved').exists()

@pytest.mark.parametrize('failure',['outside','changed','legacy'])
def test_cleanup_refuses_outside_or_changed_or_legacy_metadata(folder,failure):
    result,path,_=package(folder)
    image=ROOT/result['items'][0]['image_path'];unrelated=folder/'outside.redacted.png';unrelated.write_bytes(b'KEEP')
    if failure=='outside':
        result['items'][0]['image_path']=str(unrelated);result['items'][0]['observation']['image']['sha256']=sha256(unrelated)
    elif failure=='changed':image.write_bytes(b'CHANGED SYNTHETIC')
    else:result.pop('local_material')
    atomic_json(path,result)
    outcomes=purge_local_material(work=folder/'no-work',results=folder/'analysis',training=folder/'no-training',now=datetime.now(timezone.utc)+timedelta(days=8),dry_run=False)
    assert any(row['kind']=='derived_refused' for row in outcomes)
    assert image.exists() and path.exists() and unrelated.read_bytes()==b'KEEP'

def test_diagnostic_cleanup_recognizes_analysis_and_preserves_input_and_unrelated_files(folder,monkeypatch):
    class Model:
        def __init__(self,task):self.task=task
        def predict(self,frame):return np.zeros(frame.shape[:2],np.uint8) if self.task=='lane' else []
    monkeypatch.setattr(pipeline,'create_models',lambda *a,**k:{task:Model(task) for task in ('pothole','lane','redaction')})
    monkeypatch.setattr(pipeline,'replay',lambda *a,**k:iter([{'frame_id':0,'pts_seconds':0.,'frame':np.full((48,64,3),120,np.uint8)}]))
    source=folder/'source.png';assert cv2.imwrite(str(source),np.zeros((48,64,3),np.uint8))
    cap=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    out=folder/'diagnostic'
    result=pipeline.analyze_video(source,out,pothole_manifest='none',lane_manifest='none',redaction_manifest='none',source_expires_at=cap)
    assert result['local_material']['expires_at']==cap and result['local_material']['capture_time_inferred'] is False
    assert result['observations'][0]['expires_at']==cap
    (out/'unrelated.txt').write_text('KEEP')
    purge_local_material(work=folder/'no-work',results=out,training=folder/'no-training',now=policy.instant(cap)+timedelta(seconds=1),dry_run=False)
    assert source.exists() and (out/'unrelated.txt').read_text()=='KEEP'
    assert not (out/'analysis.json').exists() and not list(out.glob('*.redacted.png'))

def test_direct_new_review_cannot_revive_eight_day_old_unreviewed_image(folder):
    result,_,_=package(folder)
    item=result['items'][0];obs=json.loads(json.dumps(item['observation']));obs['image']['reviewed_for_sharing']=True
    obs['captured_at']=(datetime.now(timezone.utc)-timedelta(days=8)).isoformat()
    with pytest.raises(ValueError,match='expired'):Outbox(folder/'queue.sqlite',folder/'key').enqueue(obs,ROOT/item['image_path'],reviewed_by='SYNTHETIC',confirmed_human_review=True)

def test_source_cap_expired_queued_item_never_contacts_server(folder,monkeypatch):
    import httpx
    import agent.outbox as module
    cap=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    _,path,decisions=package(folder,cap=cap)
    workflow.approve_review(path,decisions,reviewed_by='SYNTHETIC',queue_path=folder/'queue.sqlite',key_path=folder/'key')
    class Future(datetime):
        @classmethod
        def now(cls,tz=None):return policy.instant(cap)+timedelta(seconds=1)
    monkeypatch.setattr(module,'datetime',Future)
    transport=httpx.MockTransport(lambda request:pytest.fail('expired source sent HTTP'))
    queue=Outbox(folder/'queue.sqlite',folder/'key')
    assert queue.drain('http://127.0.0.1:8000/api/v1',device_token='synthetic',transport=transport)[0]['state']=='expired'

def test_junction_inside_results_is_not_walked_or_deleted(folder,monkeypatch):
    from pathlib import Path
    results=folder/'results';link=results/'junction';link.mkdir(parents=True)
    (link/'keep.txt').write_text('KEEP')
    original=Path.is_junction
    monkeypatch.setattr(Path,'is_junction',lambda self:self==link or original(self))
    outcomes=purge_local_material(work=folder/'no-work',results=results,training=folder/'no-training',dry_run=False)
    assert any(row.get('reason')=='material_links_forbidden' for row in outcomes)
    assert (link/'keep.txt').read_text()=='KEEP'

def test_review_cannot_extend_seven_day_limit_by_editing_deadline_metadata(folder,monkeypatch):
    result,path,decisions=package(folder)
    captured=policy.instant(result['items'][0]['observation']['captured_at'])
    result['local_material']['expires_at']=(captured+timedelta(days=90)).isoformat()
    for item in result['items']:item['material_expires_at']=result['local_material']['expires_at']
    atomic_json(path,result)
    monkeypatch.setattr(policy,'current',lambda:captured+timedelta(days=8))
    with pytest.raises(ValueError,match='expired'):workflow.approve_review(path,decisions,reviewed_by='SYNTHETIC',queue_path=folder/'queue.sqlite',key_path=folder/'key')
    assert not (folder/'analysis/approved').exists()

def test_missing_item_cap_cannot_override_original_run_and_package_cap(folder):
    cap=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()
    result,path,decisions=package(folder,cap=cap)
    for item in result['items']:item.pop('source_expires_at')
    atomic_json(path,result)
    workflow.approve_review(path,decisions,reviewed_by='SYNTHETIC',queue_path=folder/'queue.sqlite',key_path=folder/'key')
    queue=Outbox(folder/'queue.sqlite',folder/'key')
    with queue.connect() as db:item=queue.decode(db.execute('SELECT * FROM queue').fetchone())
    assert item['source_expires_at']==cap

def test_cli_missing_source_deadline_fails_closed_without_creating_output(folder,capsys):
    from agent.__main__ import main
    result=main(['analyze',str(folder/'not-opened.mp4'),'--output',str(folder/'output'),
                 '--pothole-manifest','missing','--lane-manifest','missing','--redaction-manifest','missing'])
    assert result==1 and json.loads(capsys.readouterr().out)['reason']=='explicit_source_expiry_required_before_reading_pixels'
    assert not (folder/'output').exists()
