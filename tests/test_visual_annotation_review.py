"""Synthetic local files only: never decode a source frame or call a model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

import pytest
from scripts import render_visual_annotation_review as r

NOW=datetime(2026,11,8,tzinfo=timezone.utc)
END='2026-11-14T14:59:00Z'


@pytest.fixture
def files(tmp_path,monkeypatch):
    base=tmp_path/'session-v1'
    base.mkdir()
    monkeypatch.setattr(r,'OUTPUT_ROOT',base)
    permission=base/'permission.txt'
    permission.write_text('Synthetic approval only.',encoding='utf-8')
    image=base/'frame.png'
    image.write_bytes(b'synthetic-not-real-png')
    group=str(uuid.uuid4())
    bundle={'format':'visual-ai-annotation-bundle-v1','purpose':'human_review_only',
            'expires_at':END,'source_expires_at':END,'permission_path':str(permission),
            'permission_sha256':hashlib.sha256(permission.read_bytes()).hexdigest(),
            'leakage_group_id':group,'training_export_enabled':False,'evaluation_ground_truth_eligible':False,
            'samples':[{'sample_id':str(uuid.uuid4()),'dataset_schema_version':'1.0',
              'image':{'width':100,'height':60,'coord_space':'left_original','path':str(image),
                       'sha256':hashlib.sha256(image.read_bytes()).hexdigest()},
              'annotation_status':'held','annotator_id':'AIassistant','reviewer_id':None,
              'human_reviewed':False,'detection_eligible':False,'labels_complete':False,'negative_eligible':False,
              'leakage_group_id':group,'expires_at':END,'source_expires_at':END,
              'uncertain_regions':[{'annotation_id':str(uuid.uuid4()),'target':'pothole','bbox_xywh_px':[10,20,30,15],
                                    'reason':'<script>unsafe</script> & still uncertain'}],
              'lane_polygons':[{'kind':'lane_outline','points':[[0,0],[99,0],[99,59]],'reason':'synthetic'}]}]}
    source=base/'labels.json'
    source.write_text(json.dumps(bundle),encoding='utf-8')
    return base,source,bundle,image


def test_private_viewer_coordinates_escape_and_no_image_read(files,monkeypatch):
    base,source,bundle,image=files
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert path != image, 'Renderer must never read image bytes'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    result=r.generate(source,base/'review.html',now=NOW)
    assert result['image_bytes_read'] is False and result['samples']==1
    html=(base/'review.html').read_text(encoding='utf-8')
    match=re.search(r'<script id="bundle-data" type="application/json">(.*?)</script>',html,re.S)
    data=json.loads(match.group(1))
    assert data['samples'][0]['image']=={'relative_path':'frame.png','width':100,'height':60}
    assert data['samples'][0]['uncertain_regions'][0]['bbox_xywh_px']==[10,20,30,15]
    assert data['samples'][0]['lane_polygons'][0]['points']==[[0,0],[99,0],[99,59]]
    assert '<script>unsafe' not in html and '\\u003cscript' in html
    assert str(base) not in html and bundle['samples'][0]['sample_id'] not in html
    assert re.search(r'<img[^>]*\ssrc=',html) is None
    assert 'data-src=""' in html and 'if(locked||!guard())return' in html
    assert 'viewBox' in html and 'ArrowLeft' in html and 'ArrowRight' in html
    assert 'connect-src \'none\'' in html
    assert re.search(r'<(?:script|link)[^>]+(?:src|href)=',html) is None
    assert '대표 키프레임' in html and '전체 프레임' in html and '학습 내보내기' in html


def test_existing_output_preserved(files):
    base,source,_,_=files
    output=base/'review.html'
    output.write_text('preserve',encoding='utf-8')
    with pytest.raises(FileExistsError):r.generate(source,output,now=NOW)
    assert output.read_text(encoding='utf-8')=='preserve'


def test_expired_or_permission_tampered_bundle_never_writes(files):
    base,source,bundle,_=files
    with pytest.raises(ValueError,match='invalid_annotation_bundle'):
        r.generate(source,base/'expired.html',now=datetime(2026,11,15,tzinfo=timezone.utc))
    assert not (base/'expired.html').exists()
    bundle['permission_sha256']='a'*64
    source.write_text(json.dumps(bundle),encoding='utf-8')
    with pytest.raises(ValueError):r.generate(source,base/'bad.html',now=NOW)
    assert not (base/'bad.html').exists()


def test_scope_for_output_and_sidecars(files,tmp_path):
    base,source,bundle,_=files
    with pytest.raises(ValueError,match='output_outside_session'):
        r.generate(source,tmp_path/'outside.html',now=NOW)
    external=tmp_path/'outside.png'
    external.write_bytes(b'synthetic')
    bundle['samples'][0]['image']['path']=str(external)
    source.write_text(json.dumps(bundle),encoding='utf-8')
    with pytest.raises(ValueError,match='frame_outside_session'):
        r.generate(source,base/'bad.html',now=NOW)
    assert not (base/'bad.html').exists()


def test_training_eligibility_rejected_and_relative_nested_sidecar(files):
    base,source,bundle,image=files
    bundle['training_export_enabled']=True
    source.write_text(json.dumps(bundle),encoding='utf-8')
    with pytest.raises(ValueError):r.generate(source,base/'bad.html',now=NOW)
    bundle['training_export_enabled']=False
    nested=base/'nested'
    nested.mkdir()
    source.write_text(json.dumps(bundle),encoding='utf-8')
    r.generate(source,nested/'review.html',now=NOW)
    html=(nested/'review.html').read_text(encoding='utf-8')
    assert '"relative_path": "../frame.png"' in html


def test_label_bundle_outside_session_rejected_without_read(files,tmp_path,monkeypatch):
    base,_,_,_=files
    source=tmp_path/'outside-labels.json'
    source.write_text('{}',encoding='utf-8')
    original=Path.open
    def no_read(path,*args,**kwargs):
        assert path != source
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',no_read)
    with pytest.raises(ValueError,match='bundle_outside_session'):
        r.generate(source,base/'review.html',now=NOW)


def test_new_session_review_note_and_legacy_default(files,tmp_path,monkeypatch):
    base,source,bundle,image=files
    bundle['samples'][0]['note']='legacy note'
    source.write_text(json.dumps(bundle),encoding='utf-8')
    r.generate(source,base/'legacy.html',now=NOW)
    old=(base/'legacy.html').read_text(encoding='utf-8')
    assert 'legacy note' in old
    new=tmp_path/'session-v2-dense'
    new.mkdir()
    picture=new/'frame.png'
    picture.write_bytes(b'synthetic-only')
    bundle['samples'][0]['image']['path']=str(picture)
    bundle['samples'][0]['review_note']='AI reviewed road surface; uncertain shadow remains'
    new_source=new/'labels.json'
    new_source.write_text(json.dumps(bundle),encoding='utf-8')
    original=Path.open
    def guarded(path,*a,**kw):
        assert path not in (image,picture)
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    result=r.generate(new_source,new/'review.html',now=NOW,session_root=new)
    assert result['image_bytes_read'] is False
    html=(new/'review.html').read_text(encoding='utf-8')
    assert 'AI 검토 라벨 · 학습 후보 검수' in html
    assert '사람 검수용' not in html
    assert '사람 검수: 미완료' in html and '독립 평가 정답: 불가' in html
    assert 'AI reviewed road surface; uncertain shadow remains' in html
    assert 'legacy note' not in html
    assert '"relative_path": "frame.png"' in html
    # Old invocation remains pinned to v1; opting into v2 does not widen it.
    with pytest.raises(ValueError,match='bundle_outside_session'):
        r.generate(new_source,new/'not-created.html',now=NOW)
    assert not (new/'not-created.html').exists()


def test_sessions_cannot_mix_bundle_output_or_images(files,tmp_path):
    base,source,bundle,_=files
    new=tmp_path/'session-v2-dense'
    new.mkdir()
    with pytest.raises(ValueError,match='bundle_outside_session'):
        r.generate(source,new/'review.html',now=NOW,session_root=new)
    with pytest.raises(ValueError,match='output_outside_session'):
        r.generate(source,new/'review.html',now=NOW,session_root=base)
    picture=new/'frame.png'
    picture.write_bytes(b'synthetic-only')
    bundle['samples'][0]['image']['path']=str(picture)
    source.write_text(json.dumps(bundle),encoding='utf-8')
    with pytest.raises(ValueError,match='frame_outside_session'):
        r.generate(source,base/'not-created.html',now=NOW,session_root=base)
    assert not (new/'review.html').exists()
    assert not (base/'not-created.html').exists()


@pytest.mark.parametrize('relative',['.', 'arbitrary', 'session-', 'session-v2/nested', '../session-foreign', 'session-v2.html'])
def test_only_direct_session_directories(files,tmp_path,relative):
    base,source,_,_=files
    requested=tmp_path/relative
    with pytest.raises(ValueError,match='invalid_session_root'):
        r.generate(source,base/'rejected.html',now=NOW,session_root=requested)
    assert not (base/'rejected.html').exists()


def test_foreign_session_path_rejected_before_resolve(files,monkeypatch):
    base,source,_,_=files
    foreign=r.ROOT.parent/'outside-session-fixture'/'session-v2'
    original=Path.resolve
    def trap(path,*a,**kw):
        assert 'outside-session-fixture' not in str(path)
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'resolve',trap)
    with pytest.raises(ValueError,match='invalid_session_root'):
        r.generate(source,base/'rejected.html',now=NOW,session_root=foreign)
