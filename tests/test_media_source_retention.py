"""Real HTTP: a source contract deadline survives approval, upload and reports."""
import copy,json,time
from datetime import datetime,timezone
from uuid import uuid4
import pytest
from test_http import client,test_identity,auth,setup,data,utc

pytestmark=pytest.mark.http


def timestamp(value):return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()


def capped_image(client,auth,setup,deadline):
    body=setup['approval_body']|{'file_id':str(uuid4()),'reviewed_at':utc(),'source_expires_at':deadline}
    approval=data(client.post('/api/v1/image-approvals',headers=auth,json=body),201)
    metadata={'file_id':body['file_id'],'approval_id':approval['approval_id'],'sha256':body['sha256'],'captured_at':body['captured_at']}
    image=data(client.post('/api/v1/images',headers=setup['dh'],data={'metadata':json.dumps(metadata)},files={'file':('synthetic.jpg',setup['blob'],'image/jpeg')}),201)
    return body,approval,metadata,image


def capped_observation(client,setup,file_id):
    observation=copy.deepcopy(setup['observation'])
    observation.update(observation_id=str(uuid4()),event_id=str(uuid4()))
    observation['image']['file_id']=file_id
    return observation,data(client.post('/api/v1/observations',headers=setup['dh'],json=observation),201)


def test_source_deadline_caps_image_observation_and_report(client,auth,setup):
    deadline=utc(300)
    body,approval,metadata,image=capped_image(client,auth,setup,deadline)
    assert abs(timestamp(approval['expires_at'])-timestamp(deadline))<.01
    assert abs(timestamp(image['expires_at'])-timestamp(deadline))<.01
    observation,created=capped_observation(client,setup,body['file_id'])
    assert abs(timestamp(created['expires_at'])-timestamp(deadline))<.01
    report=data(client.get('/api/v1/reports',headers=auth,params={'format':'csv','run_id':setup['run']['run_id']}),202)
    assert timestamp(report['expires_at'])<=timestamp(deadline)+.01
    assert client.get('/api/v1/images/'+body['file_id']+'/content',headers=auth).status_code==200
    for changed in (utc(400),utc(200),None):
        response=client.post('/api/v1/image-approvals',headers=auth,json=body|{'source_expires_at':changed})
        assert response.status_code==409
    # New approval with the SAME immutable deadline may safely deduplicate.
    repeated=data(client.post('/api/v1/image-approvals',headers=auth,json=body),201)
    retried=data(client.post('/api/v1/images',headers=setup['dh'],data={'metadata':json.dumps(metadata|{'approval_id':repeated['approval_id']})},files={'file':('synthetic.jpg',setup['blob'],'image/jpeg')}))
    assert retried['deduplicated'] and timestamp(retried['expires_at'])<=timestamp(deadline)+.01


@pytest.mark.parametrize('deadline,expected',[('2026-11-01T00:00:00',422),('not-a-time',422),('2000-01-01T00:00:00Z',422)])
def test_invalid_source_deadline_is_rejected_before_upload(client,auth,setup,deadline,expected):
    body=setup['approval_body']|{'file_id':str(uuid4()),'reviewed_at':utc(),'source_expires_at':deadline}
    assert client.post('/api/v1/image-approvals',headers=auth,json=body).status_code==expected


def test_expired_source_authorization_and_legacy_default(client,auth,setup):
    body=setup['approval_body']|{'file_id':str(uuid4()),'captured_at':utc(-60),'reviewed_at':utc(),'source_expires_at':utc(-1)}
    assert client.post('/api/v1/image-approvals',headers=auth,json=body).status_code==410
    assert abs(timestamp(setup['image']['expires_at'])-(timestamp(setup['observation']['captured_at'])+90*86400))<.01


def test_expiry_denies_original_and_cached_derived_access(client,auth,setup):
    deadline=utc(15)
    body,_,_,_=capped_image(client,auth,setup,deadline)
    observation,_=capped_observation(client,setup,body['file_id'])
    report=data(client.get('/api/v1/reports',headers=auth,params={'format':'csv','run_id':setup['run']['run_id'],'snapshot_at':utc()}),202)
    job_id=report['job_id']
    end=time.monotonic()+10
    while time.monotonic()<end:
        job=data(client.get('/api/v1/jobs/'+job_id,headers=auth))
        if job['status']=='succeeded':break
        assert job['status']!='failed'
        time.sleep(.2)
    assert job['status']=='succeeded'
    wait=timestamp(deadline)-time.time()+.15
    if wait>0:time.sleep(wait)
    assert client.get('/api/v1/images/'+body['file_id']+'/content',headers=auth).status_code in (404,410)
    assert client.get('/api/v1/observations/'+observation['observation_id'],headers=auth).status_code in (404,410)
    assert client.get('/api/v1/reports/'+job_id+'/download',headers=auth).status_code in (404,409,410)
