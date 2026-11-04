"""Synthetic small archive transfer/structure guards; no source photos or GPU."""
import hashlib,io,json,time,zipfile
import httpx,pytest
from scripts import acquire_rdd2022_domain_v11 as a


@pytest.fixture
def files(tmp_path,monkeypatch):
    root=tmp_path/'bulk';root.mkdir()
    monkeypatch.setattr(a,'DEST',root);monkeypatch.setattr(a,'PARTIAL',root/'official.zip.download');monkeypatch.setattr(a,'ARCHIVE',root/'official.zip')
    monkeypatch.setattr(a.shutil,'disk_usage',lambda path:type('Disk',(),{'free':100*1024**3})())
    return root


def zip_bytes(entries):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,data in entries:
            info=zipfile.ZipInfo('synthetic');info.filename=name
            archive.writestr(info,data)
    return stream.getvalue()


def test_synthetic_full_transfer_streams_exact_size_and_checksums(files,monkeypatch):
    raw=zip_bytes([('metadata.txt',b'not an image')]);monkeypatch.setattr(a.prior,'SIZE',len(raw))
    signed='https://s3-eu-west-1.amazonaws.com/official.zip?X-Amz-Signature=DO_NOT_LOG'
    def handler(request):
        if request.url.host=='ndownloader.figshare.com':return httpx.Response(302,headers={'location':signed})
        assert str(request.url)==signed
        return httpx.Response(200,headers={'content-length':str(len(raw))},stream=httpx.ByteStream(raw))
    with httpx.Client(transport=httpx.MockTransport(handler),follow_redirects=False) as client:
        md5,sha,count=a.transfer_attempt(client,time.monotonic()+30,1)
    assert a.PARTIAL.read_bytes()==raw and count==len(raw)
    assert md5==hashlib.md5(raw).hexdigest() and sha==hashlib.sha256(raw).hexdigest()


def test_exact_resume206_hashes_original_and_remaining_bytes(files,monkeypatch):
    raw=b'PK\x03\x04'+b'SYNTHETIC'*30;offset=50;a.PARTIAL.write_bytes(raw[:offset]);monkeypatch.setattr(a.prior,'SIZE',len(raw))
    def handler(request):
        assert request.headers['range']==f'bytes={offset}-'
        return httpx.Response(206,headers={'content-range':f'bytes {offset}-{len(raw)-1}/{len(raw)}','content-length':str(len(raw)-offset)},stream=httpx.ByteStream(raw[offset:]))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:md5,sha,count=a.transfer_attempt(client,time.monotonic()+30,2)
    assert a.PARTIAL.read_bytes()==raw and count==len(raw)-offset and md5==hashlib.md5(raw).hexdigest()


@pytest.mark.parametrize('status,headers',[
    (200,{'content-length':'50'}),(206,{'content-range':'bytes 0-49/100','content-length':'50'}),
    (206,{'content-range':'bytes 50-99/101','content-length':'50'}),
    (206,{'content-range':'bytes 50-99/100','content-length':'49'}),
    (206,{'content-range':'bytes 50-99/100','content-length':'50','content-encoding':'gzip'})])
def test_invalid_resume_never_appends_or_overwrites_partial(files,monkeypatch,status,headers):
    original=b'X'*50;a.PARTIAL.write_bytes(original);monkeypatch.setattr(a.prior,'SIZE',100)
    with httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(status,headers=headers,stream=httpx.ByteStream(b'Y'*50)))) as client:
        with pytest.raises(ValueError):a.transfer_attempt(client,time.monotonic()+30,2)
    assert a.PARTIAL.read_bytes()==original


def test_access_denied_does_not_append_or_log_error_body(files):
    with httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(403,content=b'SECRET'))) as client:
        with pytest.raises(ValueError,match='official_access_denied'):a.transfer_attempt(client,time.monotonic()+30,1)
    assert not a.PARTIAL.exists()


def test_nested_country_crc_and_file_counts_are_metadata_only(files):
    nested=zip_bytes([('United_States/train/images/example.jpg',b'SYNTHETIC_NOT_JPEG'),
        ('United_States/train/annotations/xmls/example.xml',b'<annotation/>'),('United_States/test/images/test.jpg',b'SYNTHETIC')])
    archive=files/'fixture.zip';archive.write_bytes(zip_bytes([('RDD2022/United_States.zip',nested)]))
    result=a.inspect_zip(archive,time.monotonic()+30)
    assert result['all_members_CRC_verified'] is True
    assert result['countries']['United_States']=={'JPG':2,'XML':1,'train_JPG':1,'train_XML':1,'test_JPG':1,'test_XML':0,'entries':3}


@pytest.mark.parametrize('name',['../escape.jpg','/absolute.jpg','C:/escape.xml','folder\\escape.jpg'])
def test_ZipSlip_paths_rejected_without_extracting(files,name):
    archive=files/'bad.zip';archive.write_bytes(zip_bytes([(name,b'SYNTHETIC')]))
    with pytest.raises(ValueError,match='ZIP_path'):a.inspect_zip(archive,time.monotonic()+30)
    assert not (files/'escape.jpg').exists()


def test_entry_expansion_and_per_xml_size_bounds(files,monkeypatch):
    archive=files/'budget.zip';archive.write_bytes(zip_bytes([('a.xml',b'SYNTHETIC'),('b.xml',b'SYNTHETIC')]))
    monkeypatch.setattr(a,'MAX_ENTRIES',1)
    with pytest.raises(ValueError,match='entry_count'):a.inspect_zip(archive,time.monotonic()+30)
    monkeypatch.setattr(a,'MAX_ENTRIES',200000);monkeypatch.setattr(a,'MAX_EXPANDED',5)
    with pytest.raises(ValueError,match='uncompressed_bound'):a.inspect_zip(archive,time.monotonic()+30)


def test_CRC_corruption_fails_without_success(files):
    raw=bytearray(zip_bytes([('image.jpg',b'SYNTHETIC_UNIQUE_CONTENT')]))
    raw[raw.index(b'SYNTHETIC_UNIQUE_CONTENT')]+=1
    archive=files/'CRC-bad.zip';archive.write_bytes(raw)
    with pytest.raises(zipfile.BadZipFile):a.inspect_zip(archive,time.monotonic()+30)


def test_wall_deadline_checked_before_transfer_or_CRC(files):
    with httpx.Client(transport=httpx.MockTransport(lambda req:pytest.fail('transport after deadline'))) as client:
        with pytest.raises(ValueError,match='wall_budget'):a.transfer_attempt(client,time.monotonic()-1,1)


def run_plan(root,monkeypatch):
    for name in ('acquisition-plan.json','availability-results.json','publisher-metadata.json'):
        (root/name).write_text('{}')
    plan=root/'transfer-plan.json';monkeypatch.setattr(a,'PLAN',plan)
    plan.write_text(json.dumps({'transport_source_sha256':a.sha256(a.Path(a.__file__)),
        'parent_probe_plan_sha256':a.sha256(root/'acquisition-plan.json'),
        'parent_availability_sha256':a.sha256(root/'availability-results.json'),
        'publisher_metadata_sha256':a.sha256(root/'publisher-metadata.json'),'source_expires_at':a.prior.EXPIRY}))


def test_publisher_MD5_failure_preserves_partial_and_never_publishes(files,monkeypatch):
    run_plan(files,monkeypatch);raw=zip_bytes([('metadata.txt',b'SYNTHETIC')])
    monkeypatch.setattr(a.prior,'SIZE',len(raw));monkeypatch.setattr(a.prior,'MD5','0'*32)
    def transfer(*args):
        a.PARTIAL.write_bytes(raw);return hashlib.md5(raw).hexdigest(),hashlib.sha256(raw).hexdigest(),len(raw)
    monkeypatch.setattr(a,'transfer_attempt',transfer)
    result=a.run()
    assert result['usable_source'] is False and a.PARTIAL.read_bytes()==raw and not a.ARCHIVE.exists()
    assert not (files/'whole-MD5-verified.json').exists()
    assert json.loads((files/'full-transfer-results.json').read_text())['model_inferences']==0


def test_complete_publisher_MD5_then_CRC_is_required_for_usable_receipt(files,monkeypatch):
    run_plan(files,monkeypatch);raw=zip_bytes([('RDD2022/United_States.zip',zip_bytes([('United_States/train/a.xml',b'<synthetic/>')]))])
    monkeypatch.setattr(a.prior,'SIZE',len(raw));monkeypatch.setattr(a.prior,'MD5',hashlib.md5(raw).hexdigest())
    def transfer(*args):
        a.PARTIAL.write_bytes(raw);return hashlib.md5(raw).hexdigest(),hashlib.sha256(raw).hexdigest(),len(raw)
    monkeypatch.setattr(a,'transfer_attempt',transfer)
    result=a.run();receipt=json.loads((files/'full-transfer-results.json').read_text())
    assert result['usable_source'] is True and receipt['all_members_CRC_verified'] is True
    assert a.ARCHIVE.read_bytes()==raw and not a.PARTIAL.exists() and receipt['image_decodes']==0
