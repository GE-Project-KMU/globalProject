"""Synthetic non-personal MCAP/H264 fixtures: contracts, not accuracy evidence."""
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace as N
import io,json,shutil,struct,uuid
import av,numpy as np,pytest
from mcap.writer import Writer,CompressionType
from agent.paths import ROOT,sha256
from agent.recorded_mcap import RecordedMcapReader,RecordedMcapError,unpack_video_message,_bounded_chunk,MAX_RECORD

@pytest.fixture
def folder():
    path=ROOT/'.cache/test-recorded-mcap'/uuid.uuid4().hex;path.mkdir(parents=True)
    yield path
    if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('unsafe_cleanup')
    shutil.rmtree(path)

@lru_cache
def packets(count=3,bframes=0):
    encoder=av.CodecContext.create('libx264','w');encoder.width=128;encoder.height=64;encoder.pix_fmt='yuv420p'
    encoder.time_base=Fraction(1,30);encoder.framerate=Fraction(30,1);encoder.max_b_frames=bframes;encoder.gop_size=10
    encoder.options={'preset':'ultrafast',**({'tune':'zerolatency'} if not bframes else {})}
    output=[]
    for i in range(count):
        pixels=np.zeros((64,128,3),np.uint8);pixels[:,:64]=[10+i,20,30];pixels[:,64:]=[150+i,160,170]
        frame=av.VideoFrame.from_ndarray(pixels,format='bgr24');frame.pts=i
        output.extend(bytes(p) for p in encoder.encode(frame))
    output.extend(bytes(p) for p in encoder.encode(None));assert len(output)==count
    return output

def envelope(payload):return struct.pack('<II',len(payload)+60,len(payload))+payload+b'\x00'*56

def write(path,*,payloads=None,times=None,library='libmcap stereolabs',version='5.5.0',compression=CompressionType.NONE,enable_crcs=True,enable_data_crcs=True,topic='Camera_SN123/side_by_side',encoding='zed_sdk_encoded',schema_encoding=None):
    payloads=list(payloads if payloads is not None else packets());times=times if times is not None else [1_700_000_000_123_456_789+i*33_333_333 for i in range(len(payloads))]
    stream=io.BytesIO();writer=Writer(stream,compression=compression,enable_crcs=enable_crcs,enable_data_crcs=enable_data_crcs)
    writer.start(profile='',library=library);head_schema=writer.register_schema(name='sl_type',encoding='jsonschema',data=b'{}')
    head=writer.register_channel(topic='svo_header',message_encoding='json',schema_id=head_schema)
    schema=writer.register_schema(name=topic,encoding=schema_encoding or encoding,data=b'')
    video=writer.register_channel(topic=topic,message_encoding=encoding,schema_id=schema)
    writer.add_message(head,log_time=max(0,times[0]-1),publish_time=max(0,times[0]-1),data=json.dumps({'ZED_SDK_version':version,'version':'2.0.5'}).encode())
    for i,payload in enumerate(payloads):writer.add_message(video,log_time=times[i],publish_time=max(0,times[i]-7),sequence=i,data=envelope(payload))
    writer.finish();path.write_bytes(stream.getvalue());return times

@pytest.mark.parametrize('compression',[CompressionType.NONE,CompressionType.LZ4,CompressionType.ZSTD])
def test_actual_memory_decode_exact_message_timestamps_and_eof(folder,compression):
    path=folder/'synthetic.svo2';times=write(path,compression=compression);before=sha256(path)
    with RecordedMcapReader(path) as reader:
        samples=list(reader)
        assert [s['timestamp_ns'] for s in samples]==times
        assert [s['mcap_publish_time_ns'] for s in samples]==[t-7 for t in times]
        for sample in samples:
            assert sample['decoder_pts']==sample['mcap_log_time_ns']==sample['timestamp_ns']
            assert sample['decoder_time_base']==[1,1_000_000_000]
            assert sample['left'].shape==sample['right'].shape==(64,64,3)
            assert sample['left'].mean()<sample['right'].mean()-80
            assert not np.shares_memory(sample['left'],sample['right'])
            assert sample['utc'] is None and sample['gnss'] is None and sample['depth_m'] is None
        assert reader.eof and reader.read() is None and reader.integrity['fully_consumed']
        assert reader.integrity['chunks_with_crc']>0 and reader.integrity['data_end_crc_present']
    assert sha256(path)==before and sorted(p.name for p in folder.iterdir())==['synthetic.svo2']

def test_h264_delayed_frames_flush_match_message_identity(folder):
    path=folder/'delayed.svo2';times=write(path,payloads=packets(6,2))
    with RecordedMcapReader(path) as reader:
        samples=list(reader)
        assert len(samples)==6 and sorted(s['timestamp_ns'] for s in samples)==times
        assert not reader.pending and reader.flushed and reader.eof
        assert any(s['decoder_time_base'] is None and s['timestamp_match_basis'].startswith('EOF_flush_exact') for s in samples)

@pytest.mark.parametrize('mutator',[lambda p:struct.pack('<I',len(p)-5)+p[4:],lambda p:p[:4]+struct.pack('<I',len(p)-65)+p[8:],lambda p:p[:8]+b'FAIL'+p[12:],lambda p:p[:12]+b'\xff'+p[13:],lambda p:p[:-1]])
def test_observed_envelope_rejects_length_startcode_nal_or_trailer(mutator):
    with pytest.raises(RecordedMcapError):unpack_video_message(mutator(envelope(packets()[0])))

def test_chunk_crc_corruption_rejected_not_eof(folder):
    path=folder/'corrupt.svo2';write(path);data=bytearray(path.read_bytes());payload=packets()[0];index=data.index(payload);data[index+20]^=1;path.write_bytes(data)
    with pytest.raises(Exception,match='crc|CRC'):
        with RecordedMcapReader(path) as reader:list(reader)

def test_data_section_crc_checked_at_complete_eof(folder):
    path=folder/'data-crc.svo2';write(path,enable_crcs=False);data=bytearray(path.read_bytes());index=data.index(b'"ZED_SDK_version"');data[index+1]=ord('X');path.write_bytes(data)
    # A semantically invalid header also fails closed, before it can be called EOF.
    with pytest.raises(RecordedMcapError):RecordedMcapReader(path)
    path2=folder/'data-crc-only.svo2';write(path2,enable_crcs=False)
    data=bytearray(path2.read_bytes());marker=struct.pack('<II',len(packets()[0])+60,len(packets()[0]))+packets()[0]
    index=data.index(marker)+len(marker);data[index]^=1;path2.write_bytes(data)
    with pytest.raises(Exception,match='crc|CRC'):
        with RecordedMcapReader(path2) as reader:list(reader)

@pytest.mark.parametrize('kwargs',[{'library':'other'},{'version':'4.1.0'},{'topic':'camera/video'},{'schema_encoding':'other'}])
def test_profile_gate_does_not_generalize_proprietary_encoding(folder,kwargs):
    path=folder/'unsupported.svo2';write(path,**kwargs)
    with pytest.raises(RecordedMcapError):RecordedMcapReader(path)

@pytest.mark.parametrize('times',[[10,10,20],[20,10,30],[0,1,2],[2**63,2**63+1,2**63+2]])
def test_duplicate_regressed_zero_or_int64_overflow_time_rejected(folder,times):
    path=folder/'bad-time.svo2';write(path,times=times)
    with pytest.raises(RecordedMcapError,match='log_time'):
        with RecordedMcapReader(path) as reader:list(reader)

def test_truncated_footer_is_failure_not_successful_eof(folder):
    path=folder/'truncated.svo2';write(path);path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(Exception):
        with RecordedMcapReader(path) as reader:list(reader)

def test_prefix_read_does_not_claim_whole_file_integrity(folder):
    path=folder/'prefix.svo2';write(path)
    with RecordedMcapReader(path) as reader:
        assert reader.read() is not None
        assert reader.integrity['fully_consumed'] is False and reader.eof is False
    with pytest.raises(RecordedMcapError,match='closed'):reader.read()

def test_chunk_decompression_limits_before_allocation():
    with pytest.raises(RecordedMcapError,match='limit'):_bounded_chunk(N(uncompressed_size=MAX_RECORD+1))
    with pytest.raises(RecordedMcapError,match='length'):_bounded_chunk(N(uncompressed_size=3,compression='',data=b'xx',uncompressed_crc=0))

def test_frame_timestamp_loss_or_duplicate_is_failure(folder):
    path=folder/'frame-timestamps.svo2';write(path)
    with RecordedMcapReader(path) as reader:
        with pytest.raises(RecordedMcapError,match='timestamp_missing'):reader._frames([N(is_corrupt=False,pts=None,time_base=None)])
        with pytest.raises(RecordedMcapError,match='unmatched'):reader._frames([N(is_corrupt=False,pts=123,time_base=Fraction(1,10**9))])
        with pytest.raises(RecordedMcapError,match='corrupt'):reader._frames([N(is_corrupt=True)])
