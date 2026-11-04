"""Read-only, checksum-bound MCAP metadata audit; no video/sensor decoding."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
MAGIC=b'\x89MCAP0\r\n'


def scoped(value):
    path=Path(value);path=(path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('outside_workspace')
    return path


def sha(path):
    digest=hashlib.sha256()
    with scoped(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def channel_kind(topic,encoding):
    lowered=topic.lower()
    if any(token in lowered for token in ('gnss','gps','nmea','navsat')):return 'separately_named_gnss'
    if encoding=='zed_sdk_encoded':return 'zed_custom_encoded_video'
    if topic=='svo_header' and encoding=='json':return 'svo_json_header'
    if any(token in lowered for token in ('imu','sensor','barometer','magnetometer')):return 'custom_sensor_topic'
    if 'pose' in lowered or 'odometr' in lowered:return 'pose_or_odometry_topic'
    return 'other_declared_topic'


def clock_evidence(first,last,count,reversals):
    return {'first_log_time_ns':first,'last_log_time_ns':last,'message_count':count,
        'log_time_span_ns':None if first is None else last-first,
        'file_order_log_time_reversals':reversals,'clock_source':'MCAP message log_time',
        'epoch_and_device_clock_verified':False,'sdk_image_timestamp_verified':False,
        'actual_utc_capture_time_verified':False,'original_expiry':None,
        'retention_registered':False}


def audit_file(path):
    from mcap.reader import make_reader,NonSeekingReader
    path=scoped(path)
    with path.open('rb') as stream:
        if stream.read(8)!=MAGIC:raise ValueError('mcap_magic_required')
        stream.seek(0);reader=make_reader(stream,validate_crcs=True)
        header=reader.get_header();summary=reader.get_summary()
        if summary is None or summary.statistics is None:raise ValueError('summary_statistics_required')
        channels={identifier:{'id':identifier,'topic':c.topic,'message_encoding':c.message_encoding,
            'schema_id':c.schema_id,'kind':channel_kind(c.topic,c.message_encoding),
            'declared_message_count':summary.statistics.channel_message_counts.get(identifier,0)} for identifier,c in summary.channels.items()}
        schemas=[{'id':s.id,'name':s.name,'encoding':s.encoding,'schema_bytes':len(s.data)} for s in summary.schemas.values()]
        stats={'message_count':summary.statistics.message_count,'message_start_time_ns':summary.statistics.message_start_time,
            'message_end_time_ns':summary.statistics.message_end_time,'channel_count':summary.statistics.channel_count,
            'schema_count':summary.statistics.schema_count,'chunk_count':summary.statistics.chunk_count}
    observed={};json_headers=[]
    with path.open('rb') as stream:
        # File order preserves clock reversals; streaming avoids sorting payloads.
        reader=NonSeekingReader(stream,validate_crcs=True,record_size_limit=512*1024*1024)
        for _,channel,message in reader.iter_messages(log_time_order=False):
            if channel.id not in channels:raise ValueError('message_channel_missing_from_summary')
            row=observed.setdefault(channel.id,{'count':0,'min_log':message.log_time,'max_log':message.log_time,
                'min_publish':message.publish_time,'max_publish':message.publish_time,'previous_log':None,
                'reversals':0,'payload_bytes':0})
            row['count']+=1;row['payload_bytes']+=len(message.data)
            row['min_log']=min(row['min_log'],message.log_time);row['max_log']=max(row['max_log'],message.log_time)
            row['min_publish']=min(row['min_publish'],message.publish_time);row['max_publish']=max(row['max_publish'],message.publish_time)
            if row['previous_log'] is not None and message.log_time<row['previous_log']:row['reversals']+=1
            row['previous_log']=message.log_time
            if channel.topic=='svo_header' and channel.message_encoding=='json':
                if len(message.data)>2_000_000 or len(json_headers)>=10:raise ValueError('svo_header_bound_exceeded')
                value=json.loads(message.data)
                if not isinstance(value,dict):raise ValueError('json_header_object_required')
                json_headers.append(value)
    if sum(row['count'] for row in observed.values())!=stats['message_count']:raise ValueError('summary_message_count_mismatch')
    for identifier,channel in channels.items():
        row=observed.get(identifier)
        if (row['count'] if row else 0)!=channel['declared_message_count']:raise ValueError('summary_channel_count_mismatch')
        channel['observed']=row
    video=[row for identifier,row in observed.items() if channels[identifier]['kind']=='zed_custom_encoded_video']
    first=min((r['min_log'] for r in video),default=None);last=max((r['max_log'] for r in video),default=None)
    return {'header':{'profile':header.profile,'library':header.library},'schemas':schemas,'channels':list(channels.values()),
        'statistics':stats,'svo_json_headers_private':json_headers,
        'zed_sdk_versions':sorted({v['ZED_SDK_version'] for v in json_headers if isinstance(v.get('ZED_SDK_version'),str)}),
        'svo_header_versions':sorted({v['version'] for v in json_headers if isinstance(v.get('version'),str)}),
        'video_recorded_time':clock_evidence(first,last,sum(r['count'] for r in video),sum(r['reversals'] for r in video)),
        'separately_named_gnss_topic_found':any(c['kind']=='separately_named_gnss' for c in channels.values()),
        'custom_sensor_binary_decoded':False,'video_pixels_decoded':False,
        'crc_validation_requested':True,'crc_scope':'Official NonSeekingReader checks available nonzero chunk/data CRCs; absent CRC is not invented.'}


def public_aggregate(rows):
    spans=[r['audit']['video_recorded_time']['log_time_span_ns']/1e9 for r in rows if r['audit']['video_recorded_time']['log_time_span_ns'] is not None]
    kinds=Counter(c['kind'] for row in rows for c in row['audit']['channels'])
    return {'format':'recorded-mcap-metadata-public-aggregate-v1','recording_count':len(rows),
        'total_source_bytes':sum(r['size_bytes'] for r in rows),'distinct_source_count':len({r['sha256'] for r in rows}),
        'recorded_video_time_span_seconds':{'sum_of_file_spans':sum(spans),'minimum_file_span':min(spans) if spans else None,'maximum_file_span':max(spans) if spans else None},
        'recorded_video_message_count':sum(r['audit']['video_recorded_time']['message_count'] for r in rows),
        'sdk_versions':sorted({v for row in rows for v in row['audit']['zed_sdk_versions']}),
        'svo_header_versions':sorted({v for row in rows for v in row['audit']['svo_header_versions']}),
        'channel_kinds_counted_across_recordings':dict(kinds),
        'separately_named_gnss_recordings_found':sum(r['audit']['separately_named_gnss_topic_found'] for r in rows),
        'gnss_status':'Only declared topics audited; custom binary not decoded. No claim that GNSS is absent everywhere.',
        'clock_provenance':'MCAP message log_time span only; not externally verified camera UTC or SDK IMAGE timestamp.',
        'user_capture_date_statement':'User tentatively reports Oct28; separate from unverified device-clock evidence.',
        'actual_utc_capture_time_verified':False,'original_expiry_confirmed':False,'automatic_retention_registration':False,
        'source_bytes_written':False,'video_pixels_decoded':False,'custom_sensor_binary_decoded':False,
        'crc_validation_requested_all':all(r['audit']['crc_validation_requested'] for r in rows),
        'sixty_minute_real_drive_acceptance_claim':False,'duration_limit':'Sum of file log-time spans does not prove continuous driving, unique scene duration or field acceptance; repeated playback cannot extend it.'}


def run(manifest,output):
    manifest=scoped(manifest);output=scoped(output)
    if output.exists():raise ValueError('new_audit_output_required')
    snapshot=manifest.read_bytes();source=json.loads(snapshot)
    recordings=source['recordings']
    if len(recordings)!=16 or len({r['sha256'] for r in recordings})!=16:raise ValueError('handoff_inventory_mismatch')
    # Validate all originals before reading any metadata payload.
    for row in recordings:
        path=scoped(row['source_path'])
        if path.stat().st_size!=row['size_bytes'] or sha(path)!=row['sha256']:raise ValueError('source_changed_before_audit')
    output.mkdir(parents=True)
    rows=[]
    for index,row in enumerate(recordings):
        rows.append({'source_path':row['source_path'],'sha256':row['sha256'],'size_bytes':row['size_bytes'],'audit':audit_file(row['source_path'])})
        print(json.dumps({'metadata_recordings_completed':index+1,'total':16}),flush=True)
    for row in recordings:
        if scoped(row['source_path']).stat().st_size!=row['size_bytes'] or sha(row['source_path'])!=row['sha256']:raise ValueError('source_changed_during_audit')
    if manifest.read_bytes()!=snapshot:raise ValueError('handoff_manifest_changed_during_audit')
    private={'format':'recorded-mcap-private-audit-v1','handoff_manifest_sha256':hashlib.sha256(snapshot).hexdigest(),
        'audited_at':datetime.now(timezone.utc).isoformat(),'source_unchanged':True,'recordings':rows,
        'retention_registration_performed':False,'original_expiry':None,'source_bytes_written':False}
    (output/'private-audit.json').write_text(json.dumps(private,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    aggregate=public_aggregate(rows)
    (output/'aggregate.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(aggregate,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest');parser.add_argument('output')
    args=parser.parse_args();run(args.manifest,args.output)
