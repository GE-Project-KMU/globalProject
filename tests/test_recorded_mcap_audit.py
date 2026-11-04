import json
from pathlib import Path
import struct
import uuid

import pytest
from mcap.writer import Writer,CompressionType
from scripts import audit_recorded_mcap as audit


@pytest.fixture
def folder():
    path=audit.ROOT/'.cache/recorded-mcap-audit-tests'/uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def recording(path,gnss=False):
    with path.open('wb') as stream:
        writer=Writer(stream,compression=CompressionType.NONE,enable_crcs=True,enable_data_crcs=True)
        writer.start(profile='synthetic',library='synthetic-metadata-only')
        header=writer.register_channel(topic='svo_header',message_encoding='json',schema_id=0)
        video=writer.register_channel(topic='frames',message_encoding='zed_sdk_encoded',schema_id=0)
        sensor=writer.register_channel(topic='gps' if gnss else 'svo_sensors',message_encoding='custom_binary',schema_id=0)
        writer.add_message(header,1000000000,json.dumps({'ZED_SDK_version':'5.5.0','version':'2.0.5','private_camera_serial':'SYNTHETIC'}).encode(),1000000001)
        for index,stamp in enumerate([1000000000,3000000000,2000000000]):
            writer.add_message(video,stamp,b'not-real-video',stamp+7,index)
        writer.add_message(sensor,2000000000,b'never-decode-sensor',2000000001)
        writer.finish()


def test_stream_counts_crc_and_clock_do_not_invent_utc_or_expiry(folder):
    path=folder/'synthetic.svo2';recording(path)
    result=audit.audit_file(path)
    assert result['statistics']['message_count']==5
    clock=result['video_recorded_time']
    assert clock['message_count']==3 and clock['log_time_span_ns']==2000000000
    assert clock['file_order_log_time_reversals']==1
    assert not clock['actual_utc_capture_time_verified'] and not clock['sdk_image_timestamp_verified']
    assert clock['original_expiry'] is None and not clock['retention_registered']
    assert result['zed_sdk_versions']==['5.5.0'] and result['svo_header_versions']==['2.0.5']
    assert not result['video_pixels_decoded'] and not result['custom_sensor_binary_decoded']


def test_truncated_and_wrong_magic_refused(folder):
    path=folder/'complete.svo2';recording(path)
    truncated=folder/'truncated.svo2';truncated.write_bytes(path.read_bytes()[:-40])
    with pytest.raises(Exception):audit.audit_file(truncated)
    bad=folder/'wrong.svo2';bad.write_bytes(b'invalid file')
    with pytest.raises(ValueError,match='mcap_magic_required'):audit.audit_file(bad)


def test_chunk_crc_corruption_is_not_accepted(folder):
    path=folder/'complete.svo2';recording(path)
    raw=bytearray(path.read_bytes());offset=8
    while offset<len(raw)-8:
        opcode=raw[offset];length=struct.unpack_from('<Q',raw,offset+1)[0]
        if opcode==6: # MCAP Chunk: start/end/uncompressed size then CRC32.
            raw[offset+9+24]^=1
            break
        offset+=9+length
    else:raise AssertionError('synthetic_chunk_missing')
    corrupt=folder/'corrupt.svo2';corrupt.write_bytes(raw)
    with pytest.raises(Exception,match='[Cc][Rr][Cc]'):audit.audit_file(corrupt)


def test_public_whitelist_omits_private_header_paths_hashes_and_exact_times(folder):
    path=folder/'private-source.svo2';recording(path,gnss=True)
    result=audit.audit_file(path)
    public=audit.public_aggregate([{'source_path':str(path),'sha256':'a'*64,'size_bytes':path.stat().st_size,'audit':result}])
    serialized=json.dumps(public)
    for private in ['SYNTHETIC','private-source','first_log_time_ns','last_log_time_ns','a'*64,'source_path']:
        assert private not in serialized
    assert public['separately_named_gnss_recordings_found']==1
    assert not public['original_expiry_confirmed']
