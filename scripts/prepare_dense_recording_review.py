"""Dense local road survey and lossless candidate extraction, without ML predictions."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.paths import atomic_json, workspace_path
from agent.recorded_mcap import RecordedMcapReader
from scripts.prepare_visual_recording_review import (
    DEFAULT_PERMISSION, PRIVATE, PermissionGuard, digest, reviewed_inventory, write_png,
)

SESSION = PRIVATE / 'session-v2-dense'
REPORT = ROOT / 'data/recorded-drive-validation/private-full-decode.json'


def choose_frame(timestamp, first, next_tick, interval_ns):
    """Fixed source-clock grid; avoid cumulative sampling drift."""
    if timestamp < first:
        raise ValueError('nonmonotonic_source_clock')
    if timestamp < next_tick:
        return False, next_tick
    return True, first + ((timestamp-first)//interval_ns + 1)*interval_ns


def road_panel(image, clip, index, elapsed):
    import cv2
    import numpy as np
    h, w = image.shape[:2]
    y = math.floor(h*.60)
    crop = image[y:, :]
    thumb = cv2.resize(crop, (640, round(crop.shape[0]*640/w)), interpolation=cv2.INTER_AREA)
    panel = np.zeros((thumb.shape[0]+24, 640, 3), np.uint8)
    panel[24:] = thumb
    cv2.putText(panel, f'C{clip:02} F{index:06} +{elapsed:.3f}s', (6,17), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
    return panel, [0,y,w,h-y]


def inventory(guard):
    raw=REPORT.read_bytes()
    report=json.loads(raw)
    rows=reviewed_inventory(report)
    if digest(ROOT/'agent/recorded_mcap.py',guard)!=report['reader_sha256']:
        raise ValueError('validated_reader_changed')
    return raw, rows


def check_source(row, guard):
    source=workspace_path(row['source_path'],exists=True)
    if source.stat().st_size!=row['expected_bytes'] or digest(source,guard)!=row['expected_sha256']:
        raise ValueError('source_checksum_mismatch')
    return source


def survey(permission=DEFAULT_PERMISSION, interval_ms=250):
    import numpy as np
    if type(interval_ms) is not int or not 50<=interval_ms<=1000:
        raise ValueError('invalid_sampling_interval')
    guard=PermissionGuard(permission)
    raw,rows=inventory(guard)
    SESSION.mkdir(exist_ok=False)
    receipt={'format':'private-dense-road-survey-v1','status':'running',
        'expires_at':guard.deadline.isoformat(),'permission_sha256':guard.sha256,
        'interval_ms':interval_ms,'model_predictions_used':False,'human_reviewed':False,
        'independent_evaluation_eligible':False,'all_frames_visually_reviewed':False,
        'crop_normalized_xyxy':[0,.60,1,1], 'tile_width':640, 'sheet_columns':3,
        'sheet_rows':4, 'clips':[]}
    try:
        for clip,row in enumerate(rows,1):
            source=check_source(row,guard)
            panels,tiles,sheets,samples=[],[],[],[]
            first=next_tick=previous=None
            count=0
            def flush():
                if not panels:return
                n=len(panels)
                padded=panels+[np.zeros_like(panels[0]) for _ in range(12-n)]
                sheet=np.concatenate([np.concatenate(padded[i:i+3],axis=1) for i in range(0,12,3)])
                meta=write_png(SESSION/f'clip-{clip:02}-sheet-{len(sheets)+1:03}.png',sheet,guard)
                sheets.append({'image':meta,'tiles':list(tiles),'real_tiles':n})
                panels.clear();tiles.clear()
            with RecordedMcapReader(source,processing_guard=guard) as reader:
                for frame in reader:
                    guard()
                    stamp=frame['mcap_log_time_ns']
                    if frame['video_message_index']!=count or (previous is not None and stamp<=previous):
                        raise ValueError('frame_identity_or_monotonicity_failed')
                    previous=stamp
                    if first is None:first=next_tick=stamp
                    selected,next_tick=choose_frame(stamp,first,next_tick,interval_ms*1_000_000)
                    if selected or count==row['frames']-1:
                        elapsed=(stamp-first)/1e9
                        panel,crop=road_panel(frame['left'],clip,count,elapsed)
                        item={'clip':clip,'frame_index':count,'elapsed_seconds':elapsed,
                            'mcap_log_time_ns':stamp,'crop_xywh_px':crop,
                            'original_width':frame['left'].shape[1],'original_height':frame['left'].shape[0],
                            'sheet_number':len(sheets)+1,'tile_index':len(panels)}
                        panels.append(panel);tiles.append(item);samples.append(item)
                        if len(panels)==12:flush()
                    count+=1
                if count!=row['frames'] or not reader.eof or not reader.integrity.get('fully_consumed') or reader.video_messages!=count or reader.decoded_frames!=count:
                    raise ValueError('full_decode_contract_failed')
            flush()
            check_source(row,guard)
            guard.unchanged()
            if REPORT.read_bytes()!=raw:raise ValueError('source_inventory_changed')
            result={'clip':clip,'source_sha256':row['expected_sha256'],'full_decoded_frames':count,
                'EOF_verified':True,'source_hash_before_after_verified':True,
                'sampled_frames':len(samples),'max_sample_gap_seconds':max([b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(samples,samples[1:])],default=0),
                'sheets':sheets}
            atomic_json(SESSION/f'clip-{clip:02}-index.json',result)
            receipt['clips'].append(result)
            atomic_json(SESSION/'survey-index.json',receipt)
            print(json.dumps({'clip_complete':clip,'frames':count,'sampled':len(samples),'sheets':len(sheets)}),flush=True)
        receipt.update(status='completed',total_decoded_frames=sum(c['full_decoded_frames'] for c in receipt['clips']),
            total_survey_frames=sum(c['sampled_frames'] for c in receipt['clips']),total_sheets=sum(len(c['sheets']) for c in receipt['clips']))
    except BaseException as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,partial_outputs_preserved=True)
        raise
    finally:atomic_json(SESSION/'survey-index.json',receipt)
    return receipt


def extract_requests(request_path, permission=DEFAULT_PERMISSION):
    """Each request is clip + explicit frame indexes; PNGs keep native left pixels."""
    guard=PermissionGuard(permission)
    raw,rows=inventory(guard)
    request_path=workspace_path(request_path,exists=True)
    if not request_path.is_relative_to(SESSION):raise ValueError('request_outside_session')
    requested=json.loads(request_path.read_text(encoding='utf-8'))
    output=workspace_path(SESSION/requested['output_directory'])
    if output.parent!=SESSION or output==SESSION:raise ValueError('invalid_output_directory')
    plans=requested['clips']
    if not isinstance(plans,list) or not plans:raise ValueError('empty_request')
    seen=set()
    for plan in plans:
        clip=plan['clip']; frames=plan['frames']
        if type(clip) is not int or not 1<=clip<=len(rows) or clip in seen:raise ValueError('invalid_clip')
        seen.add(clip)
        if not isinstance(frames,list) or not frames or len(frames)!=len(set(frames)) or any(type(f) is not int or not 0<=f<rows[clip-1]['frames'] for f in frames):raise ValueError('invalid_frames')
    output.mkdir(exist_ok=False)
    results=[]
    for plan in plans:
        clip=plan['clip']; row=rows[clip-1]; targets=set(plan['frames'])
        source=check_source(row,guard); samples=[]; count=0; previous=None
        with RecordedMcapReader(source,processing_guard=guard) as reader:
            for frame in reader:
                guard()
                stamp=frame['mcap_log_time_ns']
                if frame['video_message_index']!=count or (previous is not None and stamp<=previous):raise ValueError('frame_identity_or_monotonicity_failed')
                previous=stamp
                if count in targets:
                    image=write_png(output/f'clip-{clip:02}-frame-{count:06}-left.png',frame['left'],guard)
                    samples.append({'clip':clip,'frame_index':count,'image':image,
                        'mcap_log_time_ns':frame['mcap_log_time_ns'],'source_recording_sha256':row['expected_sha256']})
                count+=1
            if count!=row['frames'] or not reader.eof or not reader.integrity.get('fully_consumed') or reader.video_messages!=count or reader.decoded_frames!=count or len(samples)!=len(targets):raise ValueError('candidate_decode_incomplete')
        check_source(row,guard)
        results.extend(samples)
        guard.unchanged()
        if REPORT.read_bytes()!=raw:raise ValueError('inventory_changed')
        print(json.dumps({'clip_extracted':clip,'samples':len(samples)}),flush=True)
    result={'format':'private-dense-candidates-v1','expires_at':guard.deadline.isoformat(),
        'permission_sha256':guard.sha256,'model_predictions_used':False,'samples':results}
    atomic_json(output/'index.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--permission',default=str(DEFAULT_PERMISSION))
    p.add_argument('--interval-ms',type=int,default=250)
    p.add_argument('--extract-requests')
    a=p.parse_args()
    if a.extract_requests:extract_requests(a.extract_requests,a.permission)
    else:survey(a.permission,a.interval_ms)
