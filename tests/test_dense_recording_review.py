"""Dense-review contracts using only generated arrays and tiny local fake sources."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import prepare_dense_recording_review as m


def test_fixed_clock_grid_does_not_accumulate_source_jitter():
    first=1_000_000_000
    tick=first
    selected=[]
    for i,offset in enumerate([0,249,251,499,503,750,1002]):
        take,tick=m.choose_frame(first+offset*1_000_000,first,tick,250_000_000)
        if take:selected.append(i)
    assert selected==[0,2,4,5,6]
    assert tick==first+1_250_000_000
    # A large gap skips missed grid ticks rather than scheduling from arrival.
    assert m.choose_frame(first+1_380_000_000,first,tick,250_000_000)==(True,first+1_500_000_000)
    with pytest.raises(ValueError,match='nonmonotonic_source_clock'):
        m.choose_frame(first-1,first,tick,250_000_000)


def test_road_crop_preserves_source_coordinate_geometry():
    image=np.zeros((101,127,3),dtype=np.uint8)
    image[:60]=[5,10,15]
    image[60:]=[70,80,90]
    before=image.copy()
    panel,crop=m.road_panel(image,2,17,1.25)
    assert crop==[0,60,127,41]
    assert panel.shape==(round(41*640/127)+24,640,3)
    assert panel.dtype==np.uint8
    assert np.all(panel[24:]==[70,80,90])
    assert np.array_equal(image,before)
    # Source crop is exact; panel coords below the 24px title map to crop space.
    assert crop[1]+crop[3]==image.shape[0] and crop[0]+crop[2]==image.shape[1]


class Guard:
    deadline=datetime(2026,11,14,14,59,tzinfo=timezone.utc)
    sha256='a'*64
    def __init__(self,*args):self.calls=0;self.final_checks=0
    def __call__(self):self.calls+=1
    def unchanged(self):self.final_checks+=1


@pytest.fixture
def fake(tmp_path,monkeypatch):
    private=tmp_path/'private'
    private.mkdir()
    source=tmp_path/'tiny-source.bin'
    source.write_bytes(b'synthetic bytes, not MCAP/video')
    reader_source=tmp_path/'agent'/'recorded_mcap.py'
    reader_source.parent.mkdir()
    reader_source.write_text('# fake reader source fixture',encoding='utf-8')
    row={'source_path':str(source),'expected_bytes':source.stat().st_size,
         'expected_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frames':14}
    report=tmp_path/'full-decode.json'
    reader_sha=hashlib.sha256(reader_source.read_bytes()).hexdigest()
    report.write_text(json.dumps({'reader_sha256':reader_sha,'results':[row]}),encoding='utf-8')
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'PRIVATE',private)
    monkeypatch.setattr(m,'SESSION',private/'session-v2-dense')
    monkeypatch.setattr(m,'REPORT',report)
    monkeypatch.setattr(m,'PermissionGuard',Guard)
    monkeypatch.setattr(m,'reviewed_inventory',lambda value:value['results'])
    def digest(path,guard):
        guard()
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    monkeypatch.setattr(m,'digest',digest)
    emitted=[]
    def write(path,array,guard):
        guard()
        if any(item[0]==path for item in emitted):raise FileExistsError()
        emitted.append((path,array.copy()))
        return {'path':path.relative_to(tmp_path).as_posix(),'sha256':hashlib.sha256(array.tobytes()).hexdigest(),
                'width':array.shape[1],'height':array.shape[0]}
    monkeypatch.setattr(m,'write_png',write)
    instances=[]
    class Reader:
        def __init__(self,path,processing_guard):
            self.guard=processing_guard;self.eof=False;self.integrity={'fully_consumed':False}
            self.video_messages=self.decoded_frames=0;self.closed=False;instances.append(self)
        def __enter__(self):return self
        def __exit__(self,*a):self.closed=True
        def __iter__(self):
            for i in range(row['frames']):
                self.video_messages+=1;self.decoded_frames+=1
                yield {'video_message_index':i,'mcap_log_time_ns':1_000_000_000+i*250_000_000,
                       'left':np.full((10,16,3),i+1,np.uint8)}
            self.eof=True;self.integrity['fully_consumed']=True
    monkeypatch.setattr(m,'RecordedMcapReader',Reader)
    return {'row':row,'report':report,'source':source,'emitted':emitted,'instances':instances,'Reader':Reader,'root':tmp_path}


def request(fake,*,output='candidates',clips=None):
    m.SESSION.mkdir(exist_ok=True)
    path=m.SESSION/'request.json'
    path.write_text(json.dumps({'output_directory':output,'clips':clips if clips is not None else [{'clip':1,'frames':[0,13]}]}),encoding='utf-8')
    return path


def test_survey_end_to_end_counts_coordinates_and_blank_padding(fake):
    answer=m.survey(interval_ms=250)
    assert answer['status']=='completed'
    assert answer['total_decoded_frames']==14 and answer['total_survey_frames']==14 and answer['total_sheets']==2
    assert answer['model_predictions_used'] is False and answer['all_frames_visually_reviewed'] is False
    clip=answer['clips'][0]
    assert clip['EOF_verified'] and clip['source_hash_before_after_verified']
    assert [sheet['real_tiles'] for sheet in clip['sheets']]==[12,2]
    all_tiles=[tile for sheet in clip['sheets'] for tile in sheet['tiles']]
    assert [tile['frame_index'] for tile in all_tiles]==list(range(14))
    assert all(tile['crop_xywh_px']==[0,6,16,4] for tile in all_tiles)
    assert all_tiles[-1]['sheet_number']==2 and all_tiles[-1]['tile_index']==1
    second=fake['emitted'][1][1]
    panel_h=round(4*640/16)+24
    assert second.shape==(4*panel_h,3*640,3)
    assert np.count_nonzero(second[:panel_h,2*640:])==0
    assert np.count_nonzero(second[panel_h:])==0
    assert np.all(second[24:panel_h,:640]==13)
    assert np.all(second[24:panel_h,640:1280]==14)
    assert all(reader.closed for reader in fake['instances'])
    receipt=json.loads((m.SESSION/'survey-index.json').read_text())
    assert receipt['status']=='completed' and receipt['total_survey_frames']==14
    with pytest.raises(FileExistsError):m.survey()


def test_survey_fixed_grid_also_preserves_final_frame(fake,monkeypatch):
    fake['row']['frames']=7
    report=json.loads(fake['report'].read_text())
    report['results'][0]['frames']=7
    fake['report'].write_text(json.dumps(report),encoding='utf-8')
    original=fake['Reader']
    class Irregular(original):
        def __iter__(self):
            for frame in super().__iter__():
                frame['mcap_log_time_ns']=1_000_000_000+frame['video_message_index']*100_000_000
                yield frame
    monkeypatch.setattr(m,'RecordedMcapReader',Irregular)
    answer=m.survey(interval_ms=250)
    tiles=answer['clips'][0]['sheets'][0]['tiles']
    assert [tile['frame_index'] for tile in tiles]==[0,3,5,6]
    assert [round(tile['elapsed_seconds'],2) for tile in tiles]==[0,.3,.5,.6]


@pytest.mark.parametrize('interval',[True,0,49,1001,250.0])
def test_invalid_interval_is_rejected_before_session_creation(fake,interval):
    with pytest.raises(ValueError,match='invalid_sampling_interval'):m.survey(interval_ms=interval)
    assert not m.SESSION.exists() and not fake['instances']


def test_source_hash_and_reader_source_hash_fail_before_decoder(fake):
    fake['source'].write_bytes(b'changed-source')
    with pytest.raises(ValueError,match='source_checksum_mismatch'):m.survey()
    assert not fake['instances']
    assert json.loads((m.SESSION/'survey-index.json').read_text())['status']=='failed'


def test_inventory_reader_hash_change_blocks_session(fake):
    (fake['root']/'agent'/'recorded_mcap.py').write_text('# changed',encoding='utf-8')
    with pytest.raises(ValueError,match='validated_reader_changed'):m.survey()
    assert not m.SESSION.exists() and not fake['instances']


@pytest.mark.parametrize('fault',['eof','fully_consumed','counter','identity','clock'])
def test_survey_eof_identity_clock_and_counter_guards(fake,monkeypatch,fault):
    original=fake['Reader']
    class Bad(original):
        def __iter__(self):
            for frame in super().__iter__():
                if frame['video_message_index']==2:
                    if fault=='identity':frame['video_message_index']=1
                    if fault=='clock':frame['mcap_log_time_ns']=1_000_000_000
                yield frame
            if fault=='eof':self.eof=False
            if fault=='fully_consumed':self.integrity['fully_consumed']=False
            if fault=='counter':self.decoded_frames-=1
    monkeypatch.setattr(m,'RecordedMcapReader',Bad)
    with pytest.raises(ValueError):m.survey()
    assert fake['instances'][0].closed
    assert json.loads((m.SESSION/'survey-index.json').read_text())['status']=='failed'


@pytest.mark.parametrize('plans', [[],[{'clip':True,'frames':[0]}],[{'clip':2,'frames':[0]}],
                                    [{'clip':1,'frames':[0]},{'clip':1,'frames':[1]}],
                                    [{'clip':1,'frames':[]}],[{'clip':1,'frames':[0,0]}],
                                    [{'clip':1,'frames':[-1]}],[{'clip':1,'frames':[14]}],
                                    [{'clip':1,'frames':[True]}],[{'clip':1,'frames':[1.0]}]])
def test_invalid_extract_requests_create_no_output_or_reader(fake,plans):
    path=request(fake,clips=plans)
    with pytest.raises(ValueError):m.extract_requests(path)
    assert not (m.SESSION/'candidates').exists() and not fake['instances']


@pytest.mark.parametrize('directory',['..','../escape','nested/escape','.'])
def test_extract_output_scope(fake,directory):
    path=request(fake,output=directory)
    with pytest.raises(ValueError,match='invalid_output_directory'):m.extract_requests(path)
    assert not fake['instances']


def test_extract_exact_original_arrays_and_no_overwrite(fake):
    path=request(fake)
    result=m.extract_requests(path)
    assert result['model_predictions_used'] is False
    assert [sample['frame_index'] for sample in result['samples']]==[0,13]
    assert all(array.shape==(10,16,3) for _,array in fake['emitted'])
    assert np.all(fake['emitted'][0][1]==1) and np.all(fake['emitted'][1][1]==14)
    previous=(m.SESSION/'candidates'/'index.json').read_bytes()
    with pytest.raises(FileExistsError):m.extract_requests(path)
    assert (m.SESSION/'candidates'/'index.json').read_bytes()==previous


def test_extract_rejects_request_outside_session(fake):
    outside=fake['root']/'outside-request.json'
    outside.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='request_outside_session'):m.extract_requests(outside)
    assert not fake['instances']


def test_extract_eof_failure_preserves_partial_output_but_no_success_receipt(fake,monkeypatch):
    original=fake['Reader']
    class NoEOF(original):
        def __iter__(self):
            yield from super().__iter__()
            self.eof=False
    monkeypatch.setattr(m,'RecordedMcapReader',NoEOF)
    with pytest.raises(ValueError,match='candidate_decode_incomplete'):m.extract_requests(request(fake))
    assert len(fake['emitted'])==2
    assert not (m.SESSION/'candidates'/'index.json').exists()
    assert fake['instances'][0].closed


@pytest.mark.parametrize('change',['source','inventory'])
def test_after_decode_source_and_inventory_changes_block_success(fake,monkeypatch,change):
    original=fake['Reader']
    class Changed(original):
        def __iter__(self):
            yield from super().__iter__()
            if change=='source':fake['source'].write_bytes(b'modified after decoding')
            else:fake['report'].write_text('{}',encoding='utf-8')
    monkeypatch.setattr(m,'RecordedMcapReader',Changed)
    expected='source_checksum_mismatch' if change=='source' else 'source_inventory_changed'
    with pytest.raises(ValueError,match=expected):m.survey()
    receipt=json.loads((m.SESSION/'survey-index.json').read_text())
    assert receipt['status']=='failed' and receipt['partial_outputs_preserved'] is True


def test_extract_after_decode_hash_check_preserves_failed_output(fake,monkeypatch):
    original=fake['Reader']
    class Changed(original):
        def __iter__(self):
            yield from super().__iter__()
            fake['source'].write_bytes(b'changed')
    monkeypatch.setattr(m,'RecordedMcapReader',Changed)
    with pytest.raises(ValueError,match='source_checksum_mismatch'):m.extract_requests(request(fake))
    assert len(fake['emitted'])==2
    assert not (m.SESSION/'candidates'/'index.json').exists()


@pytest.mark.parametrize('fault',['video_counter','decoded_counter','clock','identity'])
def test_extract_enforces_matching_counters_and_monotonic_clock(fake,monkeypatch,fault):
    original=fake['Reader']
    class Bad(original):
        def __iter__(self):
            for frame in super().__iter__():
                if frame['video_message_index']==2:
                    if fault=='clock':frame['mcap_log_time_ns']=1_000_000_000
                    if fault=='identity':frame['video_message_index']=1
                yield frame
            if fault=='video_counter':self.video_messages-=1
            if fault=='decoded_counter':self.decoded_frames-=1
    monkeypatch.setattr(m,'RecordedMcapReader',Bad)
    expected='frame_identity_or_monotonicity_failed' if fault in ('clock','identity') else 'candidate_decode_incomplete'
    with pytest.raises(ValueError,match=expected):m.extract_requests(request(fake))
    assert fake['instances'][0].closed
    assert not (m.SESSION/'candidates'/'index.json').exists()
