import json
from datetime import datetime,timedelta,timezone
import httpx,pytest
from scripts import acquire_rdd2022_blind_signed as tool

def folder(tmp_path):
    (tmp_path/'source.json').write_text(json.dumps({'expires_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()}))
    return tmp_path

def test_exact_signed_url_reused_without_appended_query_and_disk_cache_checked(tmp_path,monkeypatch):
    monkeypatch.setattr(tool.selection,'SIZE',10);requests=[];data=b'0123456789'
    date=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    signed=f'https://s3-eu-west-1.amazonaws.com/publisher/object?X-Amz-Date={date}&X-Amz-Expires=600&X-Amz-Signature=secret'
    def handler(request):
        requests.append(request)
        if request.url.host=='ndownloader.figshare.com':return httpx.Response(302,headers={'Location':signed})
        start,end=map(int,request.headers['Range'][6:].split('-'))
        return httpx.Response(206,headers={'Content-Range':f'bytes {start}-{end}/10'},content=data[start:end+1])
    with httpx.Client(transport=httpx.MockTransport(handler),follow_redirects=True) as client:
        with tool.SignedRange(client=client,block=4,folder=folder(tmp_path)) as remote:
            assert remote.fetch(0,3)==b'0123';assert remote.fetch(4,7)==b'4567'
            assert sum(r.url.host=='ndownloader.figshare.com' for r in requests)==1
            assert all(str(r.url)==signed for r in requests if r.url.host.endswith('amazonaws.com'))
            assert remote.fetch(0,3)==b'0123' and remote.disk_hits==1
            (remote.disk/'0-3.bin').write_bytes(b'xxxx')
            with pytest.raises(ValueError,match='cache_integrity'):remote.fetch(0,3)

def test_expired_source_rejected_before_any_network_or_cache(tmp_path):
    path=folder(tmp_path)
    (path/'source.json').write_text(json.dumps({'expires_at':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()}))
    with httpx.Client(transport=httpx.MockTransport(lambda _:pytest.fail('no network'))) as client:
        with tool.SignedRange(client=client,folder=path) as remote:
            with pytest.raises(ValueError,match='source_expired'):remote.fetch(0,3)

@pytest.mark.parametrize('url',['http://s3-eu-west-1.amazonaws.com/object','https://attacker.example/object','https://figshare.com.attacker.example/object'])
def test_official_https_hosts_only(url):
    with pytest.raises(ValueError,match='redirect_host'):tool.checked_host(httpx.URL(url))

def test_signed_response_wrong_range_never_cached(tmp_path,monkeypatch):
    monkeypatch.setattr(tool.time,'sleep',lambda _:None)
    with httpx.Client(transport=httpx.MockTransport(lambda _:httpx.Response(206,headers={'Content-Range':'bytes 4-7/10'},content=b'xxxx'))) as client:
        with tool.SignedRange(client=client,folder=folder(tmp_path)) as remote:
            remote.signed_url=httpx.URL('https://s3-eu-west-1.amazonaws.com/object')
            remote.signed_until=datetime.now(timezone.utc)+timedelta(minutes=1)
            monkeypatch.setattr(remote,'resolve',lambda:setattr(remote,'signed_url',httpx.URL('https://s3-eu-west-1.amazonaws.com/object')))
            with pytest.raises(ValueError,match='exact_content_range'):remote.fetch(0,3)
            assert not list(remote.disk.glob('*.bin'))

def test_cli_failure_never_prints_signed_exception_or_traceback(monkeypatch,capsys):
    def fail():raise httpx.HTTPStatusError('secret_signed_url',request=httpx.Request('GET','https://s3-eu-west-1.amazonaws.com/private?X-Amz-Signature=credential'),response=httpx.Response(403))
    monkeypatch.setattr(tool,'run',fail)
    assert tool.main()==1
    captured=capsys.readouterr();record=json.loads(captured.err)
    assert record['error_type']=='HTTPStatusError'
    assert all(word not in captured.err for word in ['credential','secret_signed_url','Traceback','s3-eu'])
