"""Synthetic XML/history metadata only; no real source archive is opened."""
import io,json,zipfile
import pytest
from scripts import audit_rdd2022_domain_v11 as a


def xml(label='D40',box=('1','2','9','10'),width='20',height='30'):
    coords=''.join(f'<{key}>{value}</{key}>' for key,value in zip(('xmin','ymin','xmax','ymax'),box))
    return f'<annotation><size><width>{width}</width><height>{height}</height></size><object><name>{label}</name><bndbox>{coords}</bndbox></object></annotation>'.encode()


def test_literal_D40_and_coordinate_validity_without_converting_publisher_boxes():
    good=a.annotation(xml());assert good['D40_boxes']==1 and good['invalid_boxes']==0
    assert a.annotation(xml('D00'))['D40_boxes']==0
    for box in (('-1','0','3','3'),('1','2','1','5'),('0','0','21','30'),('NaN','0','3','4'),('0','0','inf','4')):
        result=a.annotation(xml(box=box));assert result['invalid_D40_boxes']==1 and result['D40_boxes']==1
    assert a.annotation(xml(width='0'))['invalid_dimensions'] is True


@pytest.mark.parametrize('raw',[b'<!DOCTYPE annotation><annotation/>',b'<!ENTITY x "secret"><annotation/>',b'x'*(2*1024**2+1)])
def test_entity_and_size_guards(raw):
    with pytest.raises(ValueError):a.annotation(raw)


def test_only_XML_payloads_opened_and_unannotated_TEST_not_called_noD40():
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as archive:
        for name,value in [('Country/train/images/one.jpg',b'NOT_A_JPEG'),('Country/train/annotations/xmls/one.xml',xml()),
            ('Country/train/images/two.jpg',b'NOT_A_JPEG'),('Country/train/annotations/xmls/two.xml',xml('D00')),
            ('Country/test/images/three.jpg',b'NOT_A_JPEG'),('Country/train/annotations/xmls/orphan.xml',xml())]:archive.writestr(name,value)
    data.seek(0)
    with zipfile.ZipFile(data) as archive:
        class XMLOnly:
            def infolist(self):return archive.infolist()
            def open(self,item):
                assert item.filename.endswith('.xml'),'image payload opened'
                return archive.open(item)
        result=a.country_counts(XMLOnly())
    train=result['publisher_splits']['train'];test=result['publisher_splits']['test']
    assert train['image_annotation_matches']==2 and train['D40_positive_matched_images']==1 and train['noD40_matched_images']==1
    assert train['annotation_without_image']==1 and test['image_without_annotation']==1
    assert test.get('noD40_annotation_images',0)==0


def test_country_is_only_explicit_or_unambiguous_publisher_member_label():
    assert a.country_label({'country':'United_States'})==('United_States','country_field')
    assert a.country_label({'source_member':'RDD2022/India/train/images/India_0001.jpg'})==('India','publisher_member_label')
    assert a.country_label({'original_member':'United_States_0001.jpg'})==('United_States','publisher_member_label')
    assert a.country_label({'path':'data/training/some-US-path/random.jpg'})==('unknown','missing_or_ambiguous_metadata')
    assert a.country_label({'source_member':'India_0001.jpg','original_member':'Japan_0002.jpg'})[0]=='unknown'


def test_historical_dedup_SHA_is_aggregate_only_with_original_splits_preserved():
    rows=[{'country':'United_States','split':'train','sha256':'a'*64,'objects':[{'class':'pothole'}]}]*2
    rows+=[{'country':'United_States','split':'test','sha256':'b'*64,'objects':[]}]
    report=a.historical_counts(rows);country=report['countries']['United_States']
    assert country['train']['rows']==2 and country['train']['unique_sha256_images']==1 and country['test']['rows']==1
    assert 'a'*64 not in json.dumps(report) and report['country_is_publisher_metadata_not_verified_physical_location'] is True


def test_duplicate_annotations_do_not_double_count_positive_photos_and_conflicts_are_visible():
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as archive:
        archive.writestr('train/images/one.jpg',b'NOT_JPEG')
        archive.writestr('train/ann1/one.xml',xml())
        archive.writestr('train/ann2/one.xml',xml('D00'))
    data.seek(0)
    with zipfile.ZipFile(data) as archive:result=a.country_counts(archive)
    counts=result['publisher_splits']['train']
    assert counts['D40_positive_matched_images']==1 and counts['noD40_matched_images']==0
    assert counts['contradictory_D40_annotation_keys']==1 and counts['duplicate_XML_keys']==1


def test_failed_archive_receipt_blocks_before_source_or_history_is_opened(tmp_path,monkeypatch):
    root=tmp_path/'audit';root.mkdir();(root/'full-transfer-results.json').write_text(json.dumps({'usable_source':False}))
    monkeypatch.setattr(a,'DEST',root);monkeypatch.setattr(a,'PLAN',root/'audit-plan.json')
    monkeypatch.setattr(a,'HISTORY',root/'MUST_NOT_READ_HISTORY.json')
    monkeypatch.setattr(a,'sha256',lambda *args:pytest.fail('archive/history hash before successful receipt'))
    with pytest.raises(ValueError,match='whole_official_MD5_CRC'):a.prepare()
    assert not a.PLAN.exists()
