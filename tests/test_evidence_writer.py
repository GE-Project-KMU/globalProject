"""Exact PNG, bounded concurrency, failure and source-cap contracts; synthetic only."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
from pathlib import Path
import threading,uuid
import cv2
import numpy as np
import pytest
from agent import evidence_writer as ew
from agent.paths import ROOT,sha256


@pytest.fixture
def folder():
    path=ROOT/'.cache/evidence-writer-tests'/uuid.uuid4().hex;path.mkdir(parents=True);return path


def deadline():return (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()


@pytest.mark.parametrize('shape',[(1,1,3),(17,29,3),(720,1280,3)])
def test_default_and_overlap_png_bytes_and_pixel_snapshot_are_exact(folder,shape):
    original=np.random.default_rng(71).integers(0,256,shape,np.uint8)
    path=folder/'reference.png';assert cv2.imwrite(str(path),original)
    for profile in ('default','overlap'):
        image=original.copy();target=folder/(profile+'.png')
        with ew.EvidenceWriter(profile) as writer:
            future=writer.submit(target,image,deadline());image[:]=0
        assert future.result()==sha256(path)==sha256(target)
        assert np.array_equal(cv2.imread(str(target)),original)
        assert writer.closed and not list(folder.glob('*.pending-*.png'))


def test_one_snapshot_in_flight_and_second_submit_has_backpressure(folder,monkeypatch):
    entered=threading.Event();release=threading.Event();second_started=threading.Event();real=ew.cv2.imwrite
    def blocked(*args,**kwargs):entered.set();assert release.wait(3);return real(*args,**kwargs)
    monkeypatch.setattr(ew.cv2,'imwrite',blocked)
    with ew.EvidenceWriter('overlap') as writer,ThreadPoolExecutor(max_workers=1) as callers:
        one=writer.submit(folder/'one.png',np.zeros((5,5,3),np.uint8),deadline());assert entered.wait(2)
        def second():second_started.set();return writer.submit(folder/'two.png',np.ones((5,5,3),np.uint8),deadline())
        queued=callers.submit(second);assert second_started.wait(2)
        assert not one.done() and not queued.done() and writer.pending is one
        release.set();two=queued.result(timeout=3);assert two.result(timeout=3)


def test_write_failure_flushes_and_never_leaves_pending_or_published_files(folder,monkeypatch):
    def failed(path,image):Path(path).write_bytes(b'incomplete');return False
    monkeypatch.setattr(ew.cv2,'imwrite',failed)
    with pytest.raises(RuntimeError,match='image_write_failed'):
        with ew.EvidenceWriter('overlap') as writer:writer.submit(folder/'one.png',np.zeros((5,5,3),np.uint8),deadline())
    assert writer.closed and writer.pool._shutdown and list(folder.iterdir())==[]


def test_primary_processing_error_flushes_and_removes_only_its_unchanged_files(folder):
    existing=folder/'unrelated.txt';existing.write_text('preserve')
    with pytest.raises(RuntimeError,match='processing_failure'):
        with ew.EvidenceWriter('overlap') as writer:
            writer.submit(folder/'one.png',np.zeros((5,5,3),np.uint8),deadline())
            raise RuntimeError('processing_failure')
    assert list(folder.iterdir())==[existing]


def test_cap_expiry_during_encoding_refuses_publish_and_cleans_temporary(folder,monkeypatch):
    calls=[];real=ew.source_deadline
    def cap(value,**kwargs):
        calls.append(value)
        if len(calls)>=3:raise ValueError('source_material_expired')
        return real(value,**kwargs)
    monkeypatch.setattr(ew,'source_deadline',cap)
    with pytest.raises(ValueError,match='source_material_expired'):
        with ew.EvidenceWriter('overlap') as writer:writer.submit(folder/'one.png',np.zeros((5,5,3),np.uint8),deadline())
    assert list(folder.iterdir())==[]


def test_invalid_profile_input_expired_cap_and_existing_target_rejected(folder):
    with pytest.raises(ValueError):ew.EvidenceWriter('unbounded')
    with ew.EvidenceWriter('overlap') as writer:
        with pytest.raises(ValueError):writer.submit(folder/'invalid.png',np.zeros((3,3),np.uint8),deadline())
        with pytest.raises(ValueError,match='source_material_expired'):writer.submit(folder/'expired.png',np.zeros((3,3,3),np.uint8),'2000-01-01T00:00:00Z')
    target=folder/'existing.png';target.write_bytes(b'original')
    with pytest.raises(ValueError,match='new_evidence_path'):
        with ew.EvidenceWriter('overlap') as writer:writer.submit(target,np.zeros((3,3,3),np.uint8),deadline())
    assert target.read_bytes()==b'original'


def test_pipeline_overlap_preserves_every_png_quality_track_and_flushes_before_json(folder,monkeypatch):
    from agent import pipeline
    frame=np.random.default_rng(78).integers(0,256,(71,95,3),np.uint8)
    class Model:
        def __init__(self,task):self.manifest={'task':task}
        def predict(self,image):
            if self.manifest['task']=='lane':return np.ones(image.shape[:2],np.uint8)
            return [] if self.manifest['task']=='redaction' else [{'class':'pothole','bbox_xywh_px':[5.,5.,9.,9.],'confidence':.8}]
    monkeypatch.setattr(pipeline,'create_models',lambda *a,**k:{task:Model(task) for task in ('pothole','lane','redaction')})
    monkeypatch.setattr(pipeline,'replay',lambda *a,**k:iter([{'frame':frame.copy(),'frame_id':i,'pts_seconds':i/10} for i in range(4)]))
    real_json=pipeline.atomic_json
    def checked_json(path,value):
        for item in value['observations']:
            assert sha256(ROOT/item['image']['path'])==item['image']['sha256']
        real_json(path,value)
    monkeypatch.setattr(pipeline,'atomic_json',checked_json)
    results=[]
    for profile in ('default','overlap'):
        result=pipeline.analyze_video(folder/'memory',folder/profile,pothole_manifest='p',lane_manifest='l',redaction_manifest='r',io_profile=profile,source_expires_at=deadline())
        results.append(result)
    for a,b in zip(results[0]['observations'],results[1]['observations']):
        assert a['frame_id']==b['frame_id'] and a['quality']==b['quality'] and a['lane']==b['lane']
        assert a['image']['sha256']==b['image']['sha256']
        assert [{k:v for k,v in row.items() if k!='event_id'} for row in a['potholes']]==[{k:v for k,v in row.items() if k!='event_id'} for row in b['potholes']]


def test_cli_overlap_option_reaches_workflow(monkeypatch):
    from agent import __main__ as cli
    calls=[];monkeypatch.setattr(cli,'execute_config',lambda *a,**k:calls.append(k) or {'status':'completed'})
    assert cli.main(['workflow','synthetic.json','--io-profile','overlap'])==0
    assert calls[0]['io_profile']=='overlap'


def test_quality_job_runs_with_main_inference_and_errors_are_not_lost(folder):
    entered=threading.Event();inference=threading.Event()
    def quality(frame):entered.set();assert inference.wait(2);return {'usable':False,'reasons':['synthetic']}
    with ew.EvidenceWriter('overlap') as writer:
        future=writer.quality(np.zeros((3,3,3),np.uint8),quality)
        assert entered.wait(2) and not future.done();inference.set()
        assert future.result(timeout=2)=={'usable':False,'reasons':['synthetic']}
    with pytest.raises(ValueError,match='quality_failure'):
        with ew.EvidenceWriter('overlap') as writer:
            writer.quality(None,lambda image:(_ for _ in ()).throw(ValueError('quality_failure')))


def test_modified_owned_output_is_not_deleted_on_primary_failure(folder):
    target=folder/'one.png'
    with pytest.raises(RuntimeError,match='processing_failure'):
        with ew.EvidenceWriter('overlap') as writer:
            writer.submit(target,np.zeros((3,3,3),np.uint8),deadline()).result()
            target.write_bytes(b'changed-by-another-actor')
            raise RuntimeError('processing_failure')
    assert target.read_bytes()==b'changed-by-another-actor'


def test_atomic_publish_race_preserves_concurrently_created_destination(folder,monkeypatch):
    real=ew.os.link;target=folder/'race.png'
    def race(source,destination):
        Path(destination).write_bytes(b'other-actor-original')
        return real(source,destination)
    monkeypatch.setattr(ew.os,'link',race)
    with pytest.raises(ValueError,match='new_evidence_path'):
        with ew.EvidenceWriter('overlap') as writer:
            writer.submit(target,np.zeros((3,3,3),np.uint8),deadline())
    assert target.read_bytes()==b'other-actor-original'
    assert not list(folder.glob('*.pending-*.png'))


@pytest.mark.parametrize('processing_failure',[False,True])
def test_cleanup_guard_failure_is_a_note_and_preserves_primary(folder,monkeypatch,processing_failure):
    primary=ValueError('primary_processing' if processing_failure else 'primary_write')
    writer=ew.EvidenceWriter('overlap')
    def guarded_cleanup():raise PermissionError('cleanup_guard')
    monkeypatch.setattr(writer,'_discard_owned',guarded_cleanup)
    if not processing_failure:
        def failed_write(*args):raise primary
        monkeypatch.setattr(writer,'_write',failed_write)
    with pytest.raises(ValueError) as error:
        with writer:
            if processing_failure:raise primary
            writer.submit(folder/'one.png',np.zeros((3,3,3),np.uint8),deadline())
    assert error.value is primary and any('cleanup failed' in note.lower() for note in primary.__notes__)
    assert writer.closed and writer.pool._shutdown
