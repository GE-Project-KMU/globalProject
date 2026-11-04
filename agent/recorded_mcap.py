"""Experimental SDK-independent 2D reader for one observed Stereolabs MCAP profile.

The proprietary framing below is observational, not a published SVO decoder.
MCAP message clocks are preserved; neither SDK IMAGE timestamps, UTC, GNSS,
rectification nor metric depth is inferred. No raw image/video files are written.
"""
from collections import deque
from fractions import Fraction
import io,json,re,struct,zlib
import numpy as np
from .paths import workspace_path

MAX_RECORD=64*1024*1024
MAX_PACKET=16*1024*1024
NS_BASE=Fraction(1,1_000_000_000)
PROFILE='stereolabs-mcap-5.5.0-zed_sdk_encoded-h264-envelope-v1-experimental'

class RecordedMcapError(RuntimeError):pass

def unpack_video_message(data):
    """Observed [remaining_u32, h264_size_u32, AnnexB, opaque56] only."""
    if not 69<=len(data)<=MAX_PACKET:raise RecordedMcapError('unsupported_video_message_size')
    remaining,payload_size=struct.unpack_from('<II',data)
    if remaining!=len(data)-4 or payload_size!=len(data)-64:
        raise RecordedMcapError('unsupported_video_envelope_lengths')
    payload=data[8:8+payload_size]
    if len(data[8+payload_size:])!=56 or not payload.startswith(b'\x00\x00\x00\x01'):
        raise RecordedMcapError('unsupported_video_envelope_startcode_or_trailer')
    if (payload[4]&0x80) or not 1<=(payload[4]&31)<=12:raise RecordedMcapError('unsupported_h264_nal_header')
    return payload

def _bounded_chunk(chunk):
    if not 0<=chunk.uncompressed_size<=MAX_RECORD:raise RecordedMcapError('chunk_uncompressed_size_limit')
    if chunk.compression=='':data=chunk.data
    elif chunk.compression=='zstd':
        import zstandard
        # The streaming read remains bounded even if frame content-size lies.
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(chunk.data)) as stream:data=stream.read(MAX_RECORD+1)
    elif chunk.compression=='lz4':
        import lz4.frame
        dec=lz4.frame.LZ4FrameDecompressor();data=dec.decompress(chunk.data,max_length=MAX_RECORD+1)
        if not dec.eof or dec.unused_data:raise RecordedMcapError('invalid_or_oversized_lz4_chunk')
    else:raise RecordedMcapError('unsupported_chunk_compression')
    if len(data)!=chunk.uncompressed_size or len(data)>MAX_RECORD:raise RecordedMcapError('chunk_uncompressed_length_mismatch')
    if chunk.uncompressed_crc and zlib.crc32(data)!=chunk.uncompressed_crc:raise RecordedMcapError('chunk_crc_mismatch')
    return data

class RecordedMcapReader:
    def __init__(self,source):
        self.source=workspace_path(source,exists=True)
        if not self.source.is_file():raise ValueError('recording_file_required')
        self.stream=None;self.codec=None;self.closed=False;self.eof=False;self.flushed=False
        self.queue=deque();self.pending={};self.last_log_time=None;self.video_messages=0;self.decoded_frames=0
        self.integrity={'crc_validation_requested':True,'chunks_checked':0,'chunks_with_crc':0,'data_end_seen':False,'data_end_crc_present':False,'fully_consumed':False}
        self.metadata={'profile':PROFILE,'experimental':True,'sdk_independent':True,'utc':None,'gnss':None,'depth_m':None,
            'rectification':'unknown','timestamp_reference':'MCAP_log_time_assigned_to_packet_PTS; epoch_and_sdk_image_clock_unverified',
            'stereo_split_basis':'side_by_side_topic_and_even_decoded_width_only','sdk_image_timestamp_verified':False}
        try:
            import av
            self.av=av;self.stream=self.source.open('rb');self.messages=self._messages()
            self.next_message=next(self.messages,None)
            if self.next_message is None:raise RecordedMcapError('video_messages_missing')
            self.codec=av.CodecContext.create('h264','r');self.codec.thread_count=1;self.codec.options={'err_detect':'explode'}
            self.metadata['decoder']={'library':'PyAV','version':av.__version__,'codec':'h264','packet_type':'av.Packet','frame_type':'av.VideoFrame',
                'libavcodec_version':list(av.library_versions['libavcodec']),'thread_count':1,'error_detection':'explode',
                'validated_platform_scope':'observed Windows local environment only; no cross-platform validation claim'}
        except BaseException:
            self.close();raise

    def _messages(self):
        from mcap.stream_reader import StreamReader
        from mcap.records import Header,Schema,Channel,Message,Chunk,DataEnd
        from mcap.data_stream import ReadDataStream
        schemas={};channels={};video_id=None;header_seen=False;zed_header_seen=False
        def records():
            reader=StreamReader(self.stream,emit_chunks=True,validate_crcs=True,record_size_limit=MAX_RECORD)
            for record in reader.records:
                if isinstance(record,Chunk):
                    data=_bounded_chunk(record);self.integrity['chunks_checked']+=1
                    self.integrity['chunks_with_crc']+=int(record.uncompressed_crc!=0)
                    sub=ReadDataStream(io.BytesIO(data))
                    while sub.count<len(data):
                        if len(data)-sub.count<9:raise RecordedMcapError('truncated_chunk_record')
                        opcode=sub.read1();length=sub.read8();start=sub.count
                        if length>len(data)-sub.count:raise RecordedMcapError('invalid_chunk_record_length')
                        if opcode==3:item=Schema.read(sub)
                        elif opcode==4:item=Channel.read(sub)
                        elif opcode==5:
                            if length<22:raise RecordedMcapError('invalid_chunk_message_length')
                            item=Message.read(sub,length)
                        else:raise RecordedMcapError('unsupported_record_inside_chunk')
                        if sub.count-start!=length:raise RecordedMcapError('chunk_record_length_mismatch')
                        yield item
                else:yield record
        for record in records():
            if isinstance(record,Header):
                if header_seen or record.library!='libmcap stereolabs' or record.profile!='':raise RecordedMcapError('unsupported_mcap_header_profile')
                header_seen=True
            elif not header_seen:raise RecordedMcapError('mcap_header_required')
            elif isinstance(record,Schema):
                if record.id in schemas and record!=schemas[record.id]:raise RecordedMcapError('schema_redefined')
                schemas[record.id]=record
            elif isinstance(record,Channel):
                if record.id in channels and record!=channels[record.id]:raise RecordedMcapError('channel_redefined')
                channels[record.id]=record
                if record.message_encoding=='zed_sdk_encoded':
                    schema=schemas.get(record.schema_id)
                    if (not re.fullmatch(r'Camera_SN[0-9]+/side_by_side',record.topic) or record.metadata or schema is None or
                        schema.name!=record.topic or schema.encoding!='zed_sdk_encoded' or schema.data!=b''):
                        raise RecordedMcapError('unsupported_video_channel_or_schema')
                    if video_id is not None and video_id!=record.id:raise RecordedMcapError('multiple_video_streams_unsupported')
                    video_id=record.id
            elif isinstance(record,Message):
                channel=channels.get(record.channel_id)
                if channel is None:raise RecordedMcapError('message_channel_missing')
                if channel.topic=='svo_header':
                    if len(record.data)>1024*1024 or channel.message_encoding!='json':raise RecordedMcapError('unsupported_svo_header')
                    value=json.loads(record.data)
                    if not isinstance(value,dict) or value.get('ZED_SDK_version')!='5.5.0' or value.get('version')!='2.0.5':raise RecordedMcapError('unsupported_observed_svo_version')
                    zed_header_seen=True;self.metadata['recording_sdk_version']='5.5.0';self.metadata['observed_header_version']='2.0.5'
                elif record.channel_id==video_id:
                    if not zed_header_seen:raise RecordedMcapError('svo_header_before_video_required')
                    yield record
            elif isinstance(record,DataEnd):
                self.integrity['data_end_seen']=True;self.integrity['data_end_crc_present']=record.data_section_crc!=0
        if not self.integrity['data_end_seen']:raise RecordedMcapError('mcap_data_end_missing')
        self.integrity['fully_consumed']=True

    def _frames(self,frames):
        for frame in frames:
            if frame.is_corrupt:raise RecordedMcapError('decoded_frame_corrupt')
            if frame.pts is None:raise RecordedMcapError('decoded_frame_timestamp_missing')
            if frame.time_base is None:
                # PyAV flush can preserve pts while omitting frame time_base.
                # No clock is guessed: require exact identity with a still-pending
                # original message; ALL submitted packet timestamps use NS_BASE.
                if not self.flushed or frame.pts not in self.pending:raise RecordedMcapError('decoded_frame_timestamp_missing')
                timestamp=Fraction(frame.pts)
                base=None;basis='EOF_flush_exact_original_message_PTS_identity_with_explicit_packet_ns_base'
            else:
                timestamp=Fraction(frame.pts)*frame.time_base*1_000_000_000
                base=[frame.time_base.numerator,frame.time_base.denominator];basis='decoder_frame_time_base'
            if timestamp.denominator!=1:raise RecordedMcapError('decoded_frame_timestamp_not_exact_ns')
            source=self.pending.pop(int(timestamp),None)
            if source is None:raise RecordedMcapError('decoded_frame_timestamp_unmatched_or_duplicate')
            if frame.width<2 or frame.width%2 or frame.width>7680 or not 1<=frame.height<=2160:raise RecordedMcapError('unsupported_side_by_side_dimensions')
            image=frame.to_ndarray(format='bgr24');half=frame.width//2
            if image.shape!=(frame.height,frame.width,3):raise RecordedMcapError('decoded_frame_shape_invalid')
            self.queue.append({'left':image[:,:half].copy(),'right':image[:,half:].copy(),
                **source,'timestamp_ns':int(timestamp),'decoder_pts':frame.pts,'decoder_time_base':base,'timestamp_match_basis':basis,
                'utc':None,'gnss':None,'depth_m':None,'rectification':'unknown'})
            self.decoded_frames+=1
            if len(self.queue)>32:raise RecordedMcapError('decoded_queue_limit')

    def read(self):
        if self.closed:raise RecordedMcapError('recorded_mcap_reader_closed')
        if self.eof:return None
        try:
            while not self.queue:
                if self.next_message is None:
                    if not self.flushed:
                        self.flushed=True;self._frames(self.codec.decode(None))
                        if self.pending:raise RecordedMcapError('video_messages_without_decoded_frame_at_eof')
                        if self.queue:break
                    self.eof=True;return None
                message=self.next_message
                timestamp=message.log_time
                if (type(timestamp) is not int or not 0<timestamp<2**63 or
                    self.last_log_time is not None and timestamp<=self.last_log_time):raise RecordedMcapError('video_log_time_nonpositive_duplicate_or_regressed')
                payload=unpack_video_message(message.data)
                self.pending[timestamp]={'mcap_log_time_ns':timestamp,'mcap_publish_time_ns':message.publish_time,
                    'mcap_sequence':message.sequence,'video_message_index':self.video_messages}
                self.video_messages+=1;self.last_log_time=timestamp
                if len(self.pending)>32:raise RecordedMcapError('decoder_pending_message_limit')
                packet=self.av.Packet(payload);packet.pts=packet.dts=timestamp;packet.time_base=NS_BASE
                self._frames(self.codec.decode(packet))
                self.next_message=next(self.messages,None)
            return self.queue.popleft()
        except BaseException:
            self.close();raise

    def close(self):
        if not self.closed:
            self.closed=True
            if self.stream is not None:self.stream.close()
            self.codec=None;self.queue.clear();self.pending.clear()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __iter__(self):return self
    def __next__(self):
        value=self.read()
        if value is None:raise StopIteration
        return value
