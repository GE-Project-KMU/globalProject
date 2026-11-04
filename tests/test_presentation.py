"""Presentation views must not reclassify or mutate ordinary user observations."""
from copy import deepcopy
from services.inspection.presentation import data_origin,included

def test_legacy_rule_requires_exact_tenant_and_all_provenance_identifiers():
    body={'run_id':'run','device_id':'device','profile_id':'profile','model_version':'http-abc','demo':False}
    before=deepcopy(body);rule={**body,'org_id':'org','reason_code':'legacy_http_fixture_crosschecked'}
    assert data_origin(body,'org',[rule])=='http_test'
    assert data_origin(body,'other',[rule])=='field'
    for key in ('run_id','device_id','profile_id','model_version'):
        changed={**body,key:'other'}
        assert data_origin(changed,'org',[rule])=='field'
    assert body==before

def test_http_like_production_name_alone_never_hides_real_observation():
    body={'demo':False,'model_version':'http-1234567890'}
    assert data_origin(body,'org',[])=='field'
    assert included('field','presentation')

def test_explicit_synthetic_http_fixture_and_synthetic_demo_separate():
    assert data_origin({'demo':True,'model_version':'http-1234567890'},'org',[])=='http_test'
    assert data_origin({'demo':True,'model_version':'synthetic-demo-123'},'org',[])=='synthetic_demo'
    assert not included('http_test','presentation')
    assert included('synthetic_demo','presentation')
    assert included('http_test','all') and included('http_test','test')

def test_one_policy_snapshot_can_classify_a_whole_request_without_reopening_file(monkeypatch):
    import services.inspection.presentation as module
    monkeypatch.setattr(module,'policy',lambda:(_ for _ in ()).throw(AssertionError('policy reopened')))
    rows=[{'org_id':'org','run_id':'run','device_id':'device','profile_id':'profile','model_version':'http-old'}]
    for index in range(3):
        assert data_origin({'run_id':'run','device_id':'device','profile_id':'profile','model_version':'http-old'},'org',rows)=='http_test'
    assert data_origin({'demo':True,'model_version':None},'org',[])=='synthetic_demo'
