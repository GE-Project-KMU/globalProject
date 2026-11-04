"""Read-only bounded observed-profile MCAP video probe; private metadata only."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
from agent.recorded_mcap import RecordedMcapReader
def probe(source,max_frames):
    source=workspace_path(source,exists=True);before=source.stat();frames=[]
    try:
        with RecordedMcapReader(source) as reader:
            for _ in range(max_frames):
                value=reader.read()
                if value is None:break
                frames.append({key:value[key] for key in ('timestamp_ns','mcap_log_time_ns','mcap_publish_time_ns','mcap_sequence','video_message_index','decoder_pts','decoder_time_base','timestamp_match_basis')}
                    |{'left_shape':list(value['left'].shape),'right_shape':list(value['right'].shape)})
            result={'status':'experimental_2d_decode_verified_bounded','metadata':reader.metadata,'frames':frames,'integrity':reader.integrity,
                'eof':reader.eof,'video_messages_consumed':reader.video_messages,'frame_limit':max_frames}
    except Exception as exc:result={'status':'blocked_or_failed','reason':str(exc),'exception':type(exc).__name__,'frames_before_failure':len(frames),'frame_limit':max_frames}
    checksum=sha256(source);after=source.stat();result.update(source_path=source.relative_to(ROOT).as_posix(),source_sha256=checksum,source_size=before.st_size,source_mtime_ns=before.st_mtime_ns,
        source_stat_unchanged=before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,
        uploaded=False,raw_images_saved=False,training_started=False,source_capture_clock_verified=False,automatic_expiry_registration=False)
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--max-frames',type=int,default=10);p.add_argument('--all-files',action='store_true');p.add_argument('--output',required=True);p.add_argument('--source-inventory');a=p.parse_args()
    if not 1<=a.max_frames<=10:raise ValueError('initial_probe_bound_1_to_10_required')
    source=workspace_path(a.source,exists=True);files=sorted(source.glob('*.svo2')) if a.all_files else [source]
    if not files or a.all_files and not source.is_dir():raise ValueError('recording_files_required')
    results=[probe(path,a.max_frames) for path in files]
    inventory_sha=None
    if a.source_inventory:
        inventory_path=workspace_path(a.source_inventory,exists=True);inventory=json.loads(inventory_path.read_text(encoding='utf-8'));inventory_sha=sha256(inventory_path)
        indexed={workspace_path(row['source_path'],exists=True):row for row in inventory['recordings']}
        for result in results:
            row=indexed.get(workspace_path(result['source_path']))
            result['original_inventory_checksum_matches']=bool(row and row['sha256']==result['source_sha256'] and row['size_bytes']==result['source_size'])
    report={'profile':'experimental_sdk_independent_2d','reader_sha256':sha256(ROOT/'agent/recorded_mcap.py'),'probe_sha256':sha256(Path(__file__)),
        'source_inventory_sha256':inventory_sha,'results':results,'all_bounded_decodes_passed':all(r['status']=='experimental_2d_decode_verified_bounded' and r['source_stat_unchanged'] and r.get('original_inventory_checksum_matches',True) for r in results)}
    output=workspace_path(a.output)
    if output.is_relative_to(source if source.is_dir() else source.parent):raise ValueError('probe_output_must_not_modify_source_directory')
    atomic_json(output,report)
    print(json.dumps({'files':len(results),'passed':sum(r['status']=='experimental_2d_decode_verified_bounded' for r in results),'frames':sum(len(r.get('frames',[])) for r in results),'all_passed':report['all_bounded_decodes_passed']}))
    return 0 if report['all_bounded_decodes_passed'] else 2
if __name__=='__main__':raise SystemExit(main())
