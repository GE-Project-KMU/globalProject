import io
import httpx
import pytest
from scripts import prepare_rdd2022_blind as p

def test_country_filename_blocks_are_metadata_proxies_not_routes():
    assert p.filename_block('United_States','United_States_000039.jpg')=='United_States:filename-block20:1'
    assert p.filename_block('Norway','Norway_000040.jpg')=='Norway:filename-block20:2'
    with pytest.raises(ValueError):p.filename_block('Norway','other_001.jpg')

@pytest.mark.parametrize('name',['../outside.jpg','/outside.jpg','x\\outside.jpg','C:outside.jpg'])
def test_archive_path_escape_refused(name):
    with pytest.raises(ValueError):p.safe_member(name)

def test_original_literal_D40_only_and_one_index_conversion():
    raw=b'<annotation><filename>Norway_000040.jpg</filename><size><width>100</width><height>50</height></size><object><name>D40</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>100</xmax><ymax>50</ymax></bndbox></object><object><name>D00</name></object></annotation>'
    result=p.parse_voc(raw,'Norway','Norway/train/annotations/xmls/Norway_000040.xml')
    assert result['objects'][0]['bbox_xywh_px']==[0.,1.,100.,49.]
    assert result['publisher_classes']=={'D40':1,'D00':1}
    assert p.parse_voc(raw.replace(b'D40',b'D20'),'Norway','x.xml')['objects']==[]
    with pytest.raises(ValueError):p.parse_voc(b'<!DOCTYPE x><annotation/>','Norway','x.xml')

def test_segment_seeking_never_reads_outside_selected_country():
    segment=p.Segment(io.BytesIO(b'0123456789'),2,5)
    assert segment.read()==b'23456'
    segment.seek(-2,2);assert segment.read(9)==b'56'
    segment.seek(10);assert segment.read()==b''
    with pytest.raises(ValueError):segment.seek(-1)

def test_network_blocks_have_fresh_URL_and_exact_range_checks(monkeypatch):
    monkeypatch.setattr(p,'SIZE',10)
    requests=[];data=b'0123456789'
    def handler(request):
        requests.append(request)
        start,end=map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
        return httpx.Response(206,headers={'Content-Range':f'bytes {start}-{end}/10'},content=data[start:end+1])
    with httpx.Client(transport=httpx.MockTransport(handler),follow_redirects=True) as client:
        with p.OfficialRange(client=client,block=4) as remote:
            assert remote.read(6)==b'012345';remote.seek(1);assert remote.read(2)==b'12'
            assert len(requests)==2 and len({str(r.url) for r in requests})==2
            assert all(r.headers['Cache-Control']=='no-cache' for r in requests)
            assert remote.network_bytes==8

@pytest.mark.parametrize('status,content_range',[(200,'bytes 0-3/10'),(206,'bytes 4-7/10'),(206,'bytes 0-3/20')])
def test_wrong_or_stale_range_cannot_be_silently_accepted(monkeypatch,status,content_range):
    monkeypatch.setattr(p,'SIZE',10);monkeypatch.setattr(p.time,'sleep',lambda seconds:None)
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(status,headers={'Content-Range':content_range},content=b'0123'))) as client:
        with p.OfficialRange(client=client,block=4) as remote:
            with pytest.raises(ValueError,match='exact_content_range'):remote.read(1)
