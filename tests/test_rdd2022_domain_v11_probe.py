"""Synthetic HTTP transport guards; no live downloads or source image access."""
import json
import httpx
import pytest
from scripts import probe_rdd2022_domain_v11 as p


def run(handler,index=1,url=p.OUTER,clock=None):
    with httpx.Client(transport=httpx.MockTransport(handler),follow_redirects=False) as client:
        return p.probe(url,index,client,**({'clock':clock} if clock else {}))


def test_fresh_signed_redirect_is_preserved_but_never_returned_or_logged():
    signed='https://s3-eu-west-1.amazonaws.com/publisher/file.zip?X-Amz-Date=20261104T000000Z&X-Amz-Expires=10&X-Amz-Signature=SECRET_TEST_VALUE'
    seen=[]
    def handler(request):
        seen.append(str(request.url));assert request.headers['range']=='bytes=0-63'
        if request.url.host=='ndownloader.figshare.com':return httpx.Response(302,headers={'location':signed})
        assert str(request.url)==signed
        return httpx.Response(206,headers={'content-range':f'bytes 0-63/{p.SIZE}','etag':'"multipart-13"'},content=b'PK\x03\x04'+bytes(60))
    result=run(handler)
    assert result['status']=='bounded_ZIP_range_available' and result['bytes_received']==64
    assert result['whole_archive_checksum_verified'] is False and result['ZIP_CRC_verified'] is False
    assert 'SECRET_TEST_VALUE' not in json.dumps(result) and signed not in json.dumps(result)
    assert len(seen)==2


@pytest.mark.parametrize('status',[403,404,429,500])
def test_unavailable_status_does_not_read_or_serialize_signed_error_body(status):
    result=run(lambda request:httpx.Response(status,content=b'SECRET_TEST_ERROR_BODY'))
    assert result['status']=='source_unavailable' and result['bytes_received']==0
    assert 'SECRET' not in json.dumps(result)


@pytest.mark.parametrize('header,body',[
    ('bytes 1-64/13264172619',b'PK\x03\x04'+bytes(60)),
    ('bytes 0-63/999',b'PK\x03\x04'+bytes(60)),
    ('bytes 0-63/13264172619',b'PK\x03\x04'+bytes(61)),
    ('bytes 0-63/13264172619',b'PK\x03\x04'+bytes(59)),
    ('bytes 0-63/13264172619',b'NOZIP'+bytes(59))])
def test_range_size_and_zip_prefix_must_be_exact(header,body):
    with pytest.raises(ValueError):run(lambda request:httpx.Response(206,headers={'content-range':header},content=body))


@pytest.mark.parametrize('target',['http://s3.amazonaws.com/file.zip','https://evil.example/file.zip','https://user:password@s3.amazonaws.com/file.zip'])
def test_redirect_must_remain_https_official_host_without_inline_credentials(target):
    with pytest.raises(ValueError,match='unexpected_official_redirect_host'):
        run(lambda request:httpx.Response(302,headers={'location':target}))


def test_redirects_are_bounded_and_absolute_attempt_deadline_checked():
    calls=[]
    def redirects(request):
        calls.append(1);return httpx.Response(302,headers={'location':p.OUTER})
    with pytest.raises(ValueError,match='bounded_redirect_hops'):run(redirects)
    assert len(calls)==5
    moments=iter([0.,61.])
    with pytest.raises(ValueError,match='absolute_attempt_deadline'):
        run(lambda request:pytest.fail('transport after deadline'),clock=lambda:next(moments))


def test_200_entire_archive_response_is_not_consumed_or_accepted():
    result=run(lambda request:httpx.Response(200,content=b'PK\x03\x04'+bytes(60)))
    assert result['bytes_received']==0 and result['whole_archive_checksum_verified'] is False
