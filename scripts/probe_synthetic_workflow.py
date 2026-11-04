"""Hardware-free contract probe, NOT trained-AI or human privacy evidence.

Only programmatically generated geometry and deterministic ONNX graphs are used.
The fixture adapter is explicit and never changes production readiness guards.
All server resources belong to a newly provisioned, isolated test organization.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone,timedelta
import json
import os
from pathlib import Path
import sys
import time
import uuid
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import httpx
import numpy as np
from agent.paths import atomic_json,sha256,workspace_path
from agent.models import OnnxModel,load_network,readiness
from agent.capture import record
from agent.outbox import Outbox
from agent.workflow import analyze_samples,run_samples,approve_review,sync_run
from scripts.http_fixture_identity import isolated_identity

GRAPHS={
'pothole':'CAgSF3N5bnRoZXRpYy1jb250cmFjdC1vbmx5Op4BCkISBm91dHB1dCIIQ29uc3RhbnQqLgoFdmFsdWUqIggBCAEIBhABShgAAABCAAAAQgAAgEEAAIBBAACAPzMzcz+gAQQSGHN5bnRoZXRpYy1jb250cmFjdC1ncmFwaFogCgZpbWFnZXMSFgoUCAESEAoCCAEKAggDCgIIQAoCCEBiHAoGb3V0cHV0EhIKEAgBEgwKAggBCgIIAQoCCAZCBAoAEBE=',
'redaction':'CAgSF3N5bnRoZXRpYy1jb250cmFjdC1vbmx5OqIBCkYSBm91dHB1dCIIQ29uc3RhbnQqMgoFdmFsdWUqJggBCAEIBxABShwAAEBBAABAQQAAgEEAAIBBAACAP6RwfT8AAAAAoAEEEhhzeW50aGV0aWMtY29udHJhY3QtZ3JhcGhaIAoGaW1hZ2VzEhYKFAgBEhAKAggBCgIIAwoCCEAKAghAYhwKBm91dHB1dBISChAIARIMCgIIAQoCCAEKAggHQgQKABAR',
'lane':'CAgSF3N5bnRoZXRpYy1jb250cmFjdC1vbmx5Os8BCjMKBmltYWdlcwoBdwoBYhIGb3V0cHV0IgRDb252KhUKDGtlcm5lbF9zaGFwZUABQAGgAQcSGHN5bnRoZXRpYy1jb250cmFjdC1ncmFwaConCAIIAwgBCAEQAUIBd0oYAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAKhEIAhABQgFiSggAAAAAAAAAAFogCgZpbWFnZXMSFgoUCAESEAoCCAEKAggDCgIIQAoCCEBiIAoGb3V0cHV0EhYKFAgBEhAKAggBCgIIAgoCCEAKAghAQgQKABAR'}

def uid():return str(uuid.uuid4())
def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(workspace_path(path,exists=True).read_text(encoding='utf-8'))
def settings():
    values={}
    for line in (ROOT/'.env').read_text(encoding='utf-8-sig').splitlines():
        if line and not line.startswith('#') and '=' in line:
            key,value=line.split('=',1);values[key]=value.strip().strip('"').strip("'")
    return values

class SyntheticOnnxModel(OnnxModel):
    """Fixture-only constructor; inherited production preprocessing/predict/NMS.

    Deliberately rejected by production readiness (target_trained=False). This
    explicit adapter must never be used to approve a real model or user video.
    """
    def __init__(self,path):
        self.manifest=read(path)
        if self.manifest.get('fixture_kind')!='deterministic_contract_graph' or self.manifest.get('target_trained') is not False or self.manifest.get('production_approved') is not False:
            raise ValueError('explicit_synthetic_fixture_required')
        task=self.manifest['task']
        if task not in GRAPHS or workspace_path(self.manifest['weights'],exists=True).read_bytes()!=base64.b64decode(GRAPHS[task]):
            raise ValueError('frozen_synthetic_graph_required')
        if sha256(self.manifest['weights'])!=self.manifest['sha256']:raise ValueError('synthetic_graph_digest_mismatch')
        self.net=load_network(self.manifest['weights'])
        self.backend,self.backend_version=self.net.backend,self.net.backend_version

def fixtures(folder):
    folder=workspace_path(folder);folder.mkdir(parents=True,exist_ok=False)
    video=folder/'synthetic-geometry.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(128,48))
    if not writer.isOpened():raise RuntimeError('synthetic_encoder_unavailable')
    for index in range(6):
        image=np.full((48,128,3),105,np.uint8)
        image[::2,::2]=(65,100,135)
        image[:,22:28]=220
        image[4:20,4:20]=np.indices((16,16)).sum(axis=0)[...,None]%2*180+25
        cv2.putText(image,'TEST',(34,44),cv2.FONT_HERSHEY_SIMPLEX,.3,(230,230,230),1)
        writer.write(image)
    writer.release()
    bundle={'version':'SYNTHETIC-GRAPH-CONTRACT-NOT-TRAINED-'+uuid.uuid4().hex[:8],'weights_hashes':{},'config_hash':'e'*64}
    for task,wire in GRAPHS.items():
        weights=folder/(task+'.onnx');weights.write_bytes(base64.b64decode(wire))
        manifest={'task':task,'format':'onnx','fixture_kind':'deterministic_contract_graph','target_trained':False,'production_approved':False,
                  'classes':{'pothole':['pothole'],'lane':['background','paint'],'redaction':['face','license_plate']}[task],
                  'weights':weights.relative_to(ROOT).as_posix(),'sha256':sha256(weights),'input_size':[64,64],
                  'output_layout':'yolox_decoded_xywh','confidence':.35,'nms_iou':.5,'paint_threshold':.5}
        path=folder/(task+'.json');atomic_json(path,manifest);bundle[task]=path.relative_to(ROOT).as_posix();bundle['weights_hashes'][task]=manifest['sha256']
    return video,bundle

def check(response,status=200):
    if response.status_code!=status:raise AssertionError(f'HTTP_{response.status_code}_expected_{status}')
    if not response.headers.get('content-type','').startswith('application/json'):raise AssertionError('json_response_required')
    return response.json()['data']

def wait_job(client,headers,identifier):
    for _ in range(100):
        value=check(client.get('/api/v1/jobs/'+identifier,headers=headers))
        if value['status'] in ('succeeded','failed'):return value
        time.sleep(.1)
    raise AssertionError('job_timeout')

class TransientObservationFault(httpx.BaseTransport):
    """One local simulated dependency failure; all other calls use real HTTP."""
    def __init__(self,methods=None,statuses=None,*,inject=True):
        self.actual=httpx.HTTPTransport();self.injected=0
        self.methods=methods if methods is not None else Counter()
        self.statuses=statuses if statuses is not None else Counter();self.inject=inject
    def handle_request(self,request):
        self.methods[request.method]+=1
        if self.inject and request.url.path=='/api/v1/observations' and request.method=='POST' and not self.injected:
            self.injected+=1
            return httpx.Response(503,json={'error':{'code':'SYNTHETIC_TRANSIENT_FAULT'}},request=request)
        response=self.actual.handle_request(request)
        self.statuses[str(response.status_code)]+=1
        if response.status_code>=500:raise AssertionError('unexpected_live_http_5xx')
        return response
    def close(self):self.actual.close()

def prepare(base='http://127.0.0.1:8000',*,io_profile='default'):
    from agent.evidence_writer import writer_profile
    writer_profile(io_profile)
    if base!='http://127.0.0.1:8000':raise ValueError('owned_local_gateway_only')
    folder=ROOT/'data/synthetic-workflow-probe'/uid()
    video,bundle=fixtures(folder)
    identity=isolated_identity(settings(),prefix='workflow-test')
    methods=Counter();statuses=Counter()
    def request_record(response):
        methods[response.request.method]+=1;statuses[str(response.status_code)]+=1
        if response.status_code>=500:raise AssertionError('unexpected_live_http_5xx')
    with httpx.Client(base_url=base,timeout=30,trust_env=False,event_hooks={'response':[request_record]}) as client:
        assert client.get('/health').status_code==200
        login=check(client.post('/api/v1/auth/login',json=identity));auth={'Authorization':'Bearer '+login['access_token']}
        device=check(client.post('/api/v1/devices',headers=auth,json={'name':'SYNTHETIC workflow graph fixture'}),201)
        device_id=device['device_id'];token=device['device_token']
        calibration={'image_size':[64,48],'height_m':.4,'pitch_deg':30.,'camera_matrix':[[100.,0,32],[0,100.,24],[0,0,1]],
                     'distortion':[0.,0.,0.,0.,0.],'homography':np.eye(3).tolist(),'roi_polygon_px':[[0,0],[63,0],[63,47],[0,47]],'visible_distance_m':[0.,2.4],'calibrated_at':now()}
        body={'name':'SYNTHETIC geometry NOT field calibration','version':1,'camera_serial':'SYNTHETIC-NO-CAMERA','lens_label':'synthetic contract fixture','calibration':calibration,'config_hash':'a'*64}
        profile=check(client.post('/api/v1/profiles',headers=auth,json=body),201)
        local_profile={'profile_id':profile['profile_id'],'calibrated_by':'synthetic-contract-fixture-NOT-FIELD', 'calibration':calibration,
                       'measurement':{'bev_size':[64,48],'meters_per_pixel':.05,'lane_tracks':[{'lane_track_id':uid(),'color':'white','line_type':'solid','bev_x_range':[20,30],'outline_verified':True}]}}
        profile_path=folder/'profile.json';atomic_json(profile_path,local_profile)
        bundle_body={key:bundle[key] for key in ('version','weights_hashes','config_hash')}
        bundle_body.update(preprocess={'pothole':{'input_size':[64,64],'resize':'letterbox'},'lane':{'long_edge':64},'redaction':{'input_size':[64,64]}},
                           thresholds={'pothole_confidence':.35,'pothole_nms_iou':.5,'single_frame_confidence':.8,'lane_monitor':60,'lane_good':80,'min_valid_ratio':.7,'min_segment_support_ratio':.8},code_commit='synthetic-contract-fixture-not-trained',environment_hash='f'*64)
        registered=check(client.post('/api/v1/model-bundles',headers=auth,json=bundle_body),201);bundle['model_bundle_id']=registered['model_bundle_id']
        key=folder/'local.key'
        run=record(video,folder/'capture',seconds=.6,chunk_seconds=1,minimum_free_bytes=0,key_path=key)
        from agent.equipment import CHECKS
        run.update(original_capture_verified=True,device_id=device_id,profile_id=profile['profile_id'],precheck={k:True for k in CHECKS},conditions={'daylight':True,'dry_asphalt':True},source_expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
        # These are generated fixture timestamps/conditions, never field attestations.
        run['synthetic_contract_fixture']=True
        run_path=folder/'run.json';atomic_json(run_path,run)
        from agent import pipeline
        with patch.object(pipeline,'OnnxModel',SyntheticOnnxModel):
            diagnostic=pipeline.analyze_video(video,folder/'diagnostic',pothole_manifest=bundle['pothole'],lane_manifest=bundle['lane'],redaction_manifest=bundle['redaction'],stereo=True,
                source_expires_at=run['source_expires_at'],source_captured_at=run['started_at'],io_profile=io_profile)
        package=analyze_samples(run_samples(run_path,key_path=key),folder/'analysis',run=run,profile_path=profile_path,device_id=device_id,bundle=bundle,model_factory=SyntheticOnnxModel,io_profile=io_profile)
        for item in package['items']:item['observation']['demo']=True
        package['synthetic_contract_fixture']=True
        package_path=folder/'analysis/review.json';atomic_json(package_path,package)
        assert package['runtime']['engine']=='onnxruntime-cpu' and len(package['frames'])==6 and package['runtime']['io_profile']==io_profile
        pothole=next(i for i in package['items'] if i['observation']['type']=='pothole')
        assert pothole['observation']['pothole']['track_observations']==6
        assert all(not i['observation']['image']['reviewed_for_sharing'] for i in package['items'])
        assert all(not m for m in package['model_acceptance'].values())
        assert all(not readiness(bundle[task])['ready'] for task in GRAPHS)
        # Verify that masking changed the checkerboard area on the actual decoded image.
        from agent.capture import replay
        original=next(replay(video,sample_fps=10))['frame'][:,:64]
        masked=cv2.imread(str(workspace_path(pothole['image_path'])))
        assert np.mean(np.abs(masked[4:20,4:20].astype(float)-original[4:20,4:20]))>10
        sync_run(package_path,base+'/api/v1',token,transport=TransientObservationFault(methods,statuses,inject=False))
        denied=client.post('/api/v1/observations',headers={'Authorization':'Bearer '+token},json=pothole['observation'])
        assert denied.status_code in (409,422)
        decisions=folder/'simulated-test-review.json'
        atomic_json(decisions,[{'observation_id':i['observation']['observation_id'],'approve':True,'image_sha256':i['observation']['image']['sha256']} for i in package['items']])
        # This simulates the reviewer contract only for generated nonpersonal pixels;
        # it is not evidence of real human privacy review or approval of road data.
        approve_review(package_path,decisions,reviewed_by='AUTOMATED-SYNTHETIC-CONTRACT-NOT-HUMAN',queue_path=folder/'outbox.sqlite',key_path=key)
        queue=Outbox(folder/'outbox.sqlite',key)
        queue.drain(base+'/api/v1',device_token=token)
        assert all(row['state']=='blocked' and row['error']=='REVIEWER_LOGIN_REQUIRED' for row in queue.list())
        for row in queue.list():queue.retry_blocked(row['id'])
        fault=TransientObservationFault(methods,statuses)
        queue.drain(base+'/api/v1',device_token=token,reviewer_token=login['access_token'],transport=fault)
        assert fault.injected==1 and any(row['state']=='pending' and row['attempts']==1 for row in queue.list())
        assert all('image_path' not in row for row in queue.list())
        with queue.connect() as database:
            encrypted=database.execute('SELECT * FROM queue').fetchall()
        assert all(b'SYNTHETIC-GRAPH-CONTRACT' not in bytes(row['payload']) and b'captured_at' not in bytes(row['payload']) for row in encrypted)
        assert all(queue.decode(row)['observation']['demo'] is True for row in encrypted)
        time.sleep(5.1)
        queue=Outbox(folder/'outbox.sqlite',key)
        queue.drain(base+'/api/v1',device_token=token,reviewer_token=login['access_token'],transport=TransientObservationFault(methods,statuses,inject=False))
        assert all(row['state']=='acked' for row in queue.list())
        details=[]
        for item in package['items']:
            obs=item['observation'];detail=check(client.get('/api/v1/observations/'+obs['observation_id'],headers=auth));details.append(detail)
            assert detail['demo'] is True
            assert datetime.fromisoformat(detail['expires_at'].replace('Z','+00:00'))<=datetime.fromisoformat(run['source_expires_at'])
            image=client.get(detail['image_content_path'],headers=auth);assert image.status_code==200
            assert len(image.content)>0
        detail=details[0]
        patch_body={'expected_case_id':detail['case_id'],'expected_version':detail['case_version'],'status':'confirmed','priority':'normal','reason':'SYNTHETIC contract review; NOT field defect confirmation'}
        check(client.patch('/api/v1/observations/'+detail['observation_id']+'/review',headers=auth,json=patch_body))
        assert client.patch('/api/v1/observations/'+detail['observation_id']+'/review',headers=auth,json=patch_body).status_code==409
        report=client.get('/api/v1/reports',params={'format':'csv','run_id':run['run_id'],'data_scope':'synthetic'},headers=auth)
        job=check(report,report.status_code);job_id=job.get('job_id',job.get('id'))
        assert wait_job(client,auth,job_id)['status']=='succeeded'
        csv=client.get('/api/v1/reports/'+job_id+'/download',headers=auth);assert csv.status_code==200
        assert all(i['observation']['observation_id'] in csv.text for i in package['items'])
        result={'fixture_kind':'deterministic_contract_graph_not_trained_AI','io_profile':io_profile,'hardware_connected':False,'user_recordings_read':False,'public_source_images_or_weights_uploaded':False,
                'generated_nonpersonal_video_frames':6,'diagnostic_analyzed_frames':diagnostic['analyzed_frames'],'workflow_analyzed_frames':len(package['frames']),
                'tracked_frame_observations':pothole['observation']['pothole']['track_observations'],'observations_uploaded':len(details),'retained_redacted_photos':package['retained_photos'],
                'mask_changed_generated_checkerboard':True,'runtime':package['runtime']['engine'],'production_fixture_readiness_rejected':True,'unreviewed_ingestion_rejected':True,
                'missing_reviewer_blocked':True,'injected_local_503_count':fault.injected,'queue_reopened_and_recovered':True,'encrypted_queue_payload':True,'stale_review_conflict':True,
                'scoped_report_contains_workflow_observations':True,'http_methods':dict(methods),'http_statuses':dict(statuses),'unexpected_live_5xx':0,
                'original_source_cap_inherited_by_local_queue_and_server_observation':True,
                'real_AI_accuracy_measured':False,'real_human_privacy_approval_claimed':False,'field_calibration_or_GNSS_verified':False,'private_deletion_pending':True}
        atomic_json(folder/'aggregate.json',result)
        atomic_json(folder/'private-receipt.json',{'identity':identity,'run_id':run['run_id'],'observation_ids':[i['observation']['observation_id'] for i in package['items']],
                   'image_ids':[i['observation']['image']['file_id'] for i in package['items']],'device_token':token,'report_job_id':job_id,'folder':folder.relative_to(ROOT).as_posix(),'aggregate':result})
    return folder,result

def cleanup(receipt_path):
    receipt=read(receipt_path);identity=receipt['identity']
    if not identity['org_code'].startswith('workflow-test-'):raise ValueError('owned_synthetic_tenant_required')
    folder=workspace_path(receipt['folder'],exists=True)
    if folder.parent!=ROOT/'data/synthetic-workflow-probe' or str(uuid.UUID(folder.name))!=folder.name:raise ValueError('owned_synthetic_directory_required')
    previous=read(folder/'aggregate.json')
    if previous.get('private_deletion_pending') is False:return previous
    package=read(folder/'analysis/review.json')
    observations=[item['observation'] for item in package['items']]
    if {obs['observation_id'] for obs in observations}!=set(receipt['observation_ids']) or not package.get('synthetic_contract_fixture'):
        raise ValueError('owned_synthetic_package_required')
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=30,trust_env=False) as client:
        login=check(client.post('/api/v1/auth/login',json=identity));auth={'Authorization':'Bearer '+login['access_token']}
        statuses=[client.get('/api/v1/observations/'+identifier,headers=auth).status_code for identifier in receipt['observation_ids']]
        if all(status==410 for status in statuses):
            # Resume after a later assertion failed: never submit a second
            # deletion for tombstoned IDs, and verify the completed first job.
            requests=check(client.get('/api/v1/privacy-requests',headers=auth))['items']
            matching=[row for row in requests if row['request_type']=='deletion' and {ref['id'] for ref in row['target_refs']}==set(receipt['observation_ids']) and row['status']=='completed']
            assert matching and wait_job(client,auth,matching[-1]['job_id'])['status']=='succeeded'
        else:
            assert all(status==200 for status in statuses)
            result=check(client.post('/api/v1/privacy-requests',headers=auth,json={'request_type':'deletion','target_refs':[{'kind':'observation','id':identifier} for identifier in receipt['observation_ids']], 'reason':'Delete only this generated synthetic workflow fixture after browser verification'}),202)
            assert wait_job(client,auth,result['job_id'])['status']=='succeeded'
        assert all(client.get('/api/v1/observations/'+identifier,headers=auth).status_code==410 for identifier in receipt['observation_ids'])
        assert all(client.get('/api/v1/images/'+identifier+'/content',headers=auth).status_code in (404,410) for identifier in receipt['image_ids'])
        report=client.get('/api/v1/reports/'+receipt['report_job_id']+'/download',headers=auth)
        assert report.status_code in (403,404,409,410)
        for observation in observations:
            # approve_review copies the local unapproved package before setting
            # the image flag. Replay the exact approved payload sent by Outbox.
            uploaded=observation|{'image':observation['image']|{'reviewed_for_sharing':True}}
            replay=client.post('/api/v1/observations',headers={'Authorization':'Bearer '+receipt['device_token']},json=uploaded)
            if replay.status_code!=410:raise AssertionError(f'tombstone_replay_status_{replay.status_code}')
        # Exercise the real local revocation implementation with an exclusively
        # fixture-owned registry. Never scan or mutate the actual training tree.
        from agent import governance
        local_registry=folder/'governance/registry';local_registry.parent.mkdir(exist_ok=True)
        with patch.object(governance,'REGISTRY',local_registry),patch.object(governance,'REVOCATIONS',folder/'governance/revocations.json'):
            request={'observation_ids':receipt['observation_ids'],'reason':'Delete generated fixture local review/outbox copies only'}
            governance.revoke_known(request=request,dry_run=False,training_root=folder/'no-training',results_root=folder/'analysis')
            queue=Outbox(folder/'outbox.sqlite',folder/'local.key')
            queue.revoke_observations([governance.identity_hash('observation_id',identifier) for identifier in receipt['observation_ids']],dry_run=False)
            assert queue.list()==[]
            assert read(folder/'analysis/review.json')['items']==[]
            assert not list((folder/'analysis').glob('*.redacted.png'))
            try:queue.enqueue(observations[0],folder/'analysis/no-longer-existing.png',reviewed_by='SYNTHETIC',confirmed_human_review=True)
            except ValueError as error:assert str(error)=='source_or_observation_revoked'
            else:raise AssertionError('revoked_local_requeue_was_allowed')
        receipt['aggregate'].update(private_deletion_pending=False,privacy_delete_job_succeeded=True,deleted_observations_unavailable=True,deleted_images_unavailable=True,prior_report_invalidated=True)
        receipt['aggregate'].update(server_tombstone_blocks_reupload=True,local_review_photos_removed=True,local_outbox_removed=True,local_revocation_blocks_requeue=True)
        atomic_json(workspace_path(receipt['folder'])/'aggregate.json',receipt['aggregate'])
    return receipt['aggregate']

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','cleanup']);parser.add_argument('--receipt');parser.add_argument('--io-profile',choices=['default','overlap'],default='default');args=parser.parse_args()
    if args.phase=='prepare':
        folder,result=prepare(io_profile=args.io_profile);print(json.dumps({'private_receipt':(folder/'private-receipt.json').relative_to(ROOT).as_posix(),'aggregate':result}))
    else:
        if not args.receipt:parser.error('--receipt is required for cleanup')
        print(json.dumps(cleanup(args.receipt)))

if __name__=='__main__':main()
