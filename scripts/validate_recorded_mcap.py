"""Deadline-guarded full-memory decoding; original files are read-only.

Private reports contain source hashes/clocks. Public output is aggregate only.
No pixels, per-frame logs, model inputs, weights, uploads or training are written.
"""
import argparse,collections,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
from agent.recorded_mcap import RecordedMcapReader,PROFILE
MAX_AUTHORIZED=datetime.fromisoformat('2026-11-05T23:59:00+09:00').astimezone(timezone.utc)
PRIVATE_ROOT=ROOT/'data/recorded-drive-validation'

class ProcessingDeadlineExceeded(RuntimeError):pass

def private_output_path(value):
    path=workspace_path(value)
    if path==PRIVATE_ROOT or not path.is_relative_to(PRIVATE_ROOT) or path.suffix!='.json':raise ValueError('private_report_must_be_under_recorded_drive_validation')
    if path.exists():raise ValueError('fresh_output_required')
    return path

def public_output_path(value):
    path=workspace_path(value);public_root=ROOT/'docs/reports'
    if not path.is_relative_to(public_root) or path.suffix!='.json':raise ValueError('public_aggregate_report_directory_required')
    if path.exists():raise ValueError('fresh_public_report_required')
    return path

def control_metadata_unchanged(permission,inventory,report):
    return sha256(permission)==report['permission_sha256'] and sha256(inventory)==report['source_inventory_sha256']

class DeadlineGuard:
    def __init__(self,deadline,*,now=None):
        if deadline.tzinfo is None or deadline.astimezone(timezone.utc)>MAX_AUTHORIZED:raise ValueError('deadline_exceeds_explicit_user_authorization')
        self.deadline=deadline.astimezone(timezone.utc);self.now=now or (lambda:datetime.now(timezone.utc));self.checks=0
        self()
    def __call__(self):
        current=self.now()
        if current.tzinfo is None:raise ValueError('timezone_aware_processing_clock_required')
        self.checks+=1
        if current.astimezone(timezone.utc)>=self.deadline:raise ProcessingDeadlineExceeded('recording_processing_permission_deadline_reached')

def permission_guard(path,*,now=None):
    path=workspace_path(path,exists=True);raw=path.read_bytes();receipt=json.loads(raw.decode('utf-8'))
    if (receipt.get('format')!='recorded-drive-validation-permission-v1' or receipt.get('scope')!='this recording validation only' or
        receipt.get('source')!='explicit user reply in this chat'):raise ValueError('explicit_recording_validation_permission_receipt_required')
    for key in ['original_capture_date_changed','base_retention_policy_changed','automatic_delete_authorized','automatic_retention_registration','external_upload_authorized','training_authorized_by_this_exception','raw_pixel_outputs_planned']:
        if receipt.get(key) is not False:raise ValueError('permission_scope_mismatch:'+key)
    deadline=datetime.fromisoformat(receipt['authorized_until']);utc=datetime.fromisoformat(receipt['authorized_until_utc'])
    if deadline!=utc:raise ValueError('permission_deadline_disagreement')
    guard=DeadlineGuard(deadline,now=now);guard.permission_sha256=hashlib.sha256(raw).hexdigest()
    return guard,receipt

def guarded_hash(path,guard):
    guard();digest=hashlib.sha256()
    with workspace_path(path,exists=True).open('rb') as stream:
        while True:
            guard();chunk=stream.read(1024*1024)
            if not chunk:break
            digest.update(chunk)
    guard();return digest.hexdigest()

class Quality:
    """Small grayscale samples; flags are heuristic and never exclude frames."""
    def __init__(self):
        self.count=0;self.flags=collections.Counter();self.sum_mean=0.;self.sum_std=0.;self.sum_edge=0.;self.minimum=255.;self.maximum=0.;self.previous=None
    def add(self,image):
        if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or image.size==0:raise ValueError('invalid_decoded_bgr_image')
        gray=cv2.cvtColor(cv2.resize(image,(160,90),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        mean=float(gray.mean());std=float(gray.std());edge=float(cv2.Laplacian(gray,cv2.CV_32F).var())
        self.flags['mean_lt_5']+=mean<5;self.flags['mean_gt_250']+=mean>250
        self.flags['std_lt_2']+=std<2;self.flags['std_lt_0_25']+=std<.25;self.flags['laplacian_variance_lt_10']+=edge<10
        if self.previous is not None:self.flags['identical_sample_to_previous']+=np.array_equal(gray,self.previous)
        self.previous=gray;self.count+=1;self.sum_mean+=mean;self.sum_std+=std;self.sum_edge+=edge;self.minimum=min(self.minimum,mean);self.maximum=max(self.maximum,mean)
    def summary(self):
        return {'frames':self.count,'sample_size':[160,90],'flags':{key:int(value) for key,value in self.flags.items()},
            'mean_gray_min':self.minimum if self.count else None,'mean_gray_max':self.maximum if self.count else None,
            'mean_gray_average':self.sum_mean/self.count if self.count else None,'gray_std_average':self.sum_std/self.count if self.count else None,
            'laplacian_variance_average':self.sum_edge/self.count if self.count else None,'heuristics_only_no_GT_no_auto_exclusion':True}

def validate_one(row,guard,*,reader_factory=RecordedMcapReader):
    result={'source_path':row['source_path'],'expected_sha256':row['sha256'],'expected_bytes':row['size_bytes'],'frames':0,'timestamp_mismatches':0,
        'resolution_changes':0,'frame_time_base_missing_exact_flush_matches':0,'resolutions':{},'quality':{},'status':'in_progress'}
    quality={'left':Quality(),'right':Quality()};resolutions=collections.Counter();reader=None;previous=None
    try:
        guard();source=workspace_path(row['source_path'],exists=True)
        result['sha256_before']=guarded_hash(source,guard)
        if result['sha256_before']!=row['sha256'] or source.stat().st_size!=row['size_bytes']:raise ValueError('source_intake_checksum_or_size_mismatch')
        with reader_factory(source,processing_guard=guard) as reader:
            for sample in reader:
                guard()
                if sample['timestamp_ns']!=sample['mcap_log_time_ns']:result['timestamp_mismatches']+=1
                if sample['left'].shape!=sample['right'].shape:raise ValueError('stereo_shapes_disagree')
                shape=sample['left'].shape;resolution=f'{shape[1]}x{shape[0]}'
                if previous is not None and previous!=resolution:result['resolution_changes']+=1
                previous=resolution;resolutions[resolution]+=1
                for side,value in quality.items():value.add(sample[side])
                result['frames']+=1
                if sample['decoder_time_base'] is None:result['frame_time_base_missing_exact_flush_matches']+=1
                value=sample['mcap_log_time_ns'];result['log_time_min_ns']=min(result.get('log_time_min_ns',value),value);result['log_time_max_ns']=max(result.get('log_time_max_ns',value),value)
                guard()
            result.update(eof=reader.eof,video_messages=reader.video_messages,decoder_frames=reader.decoded_frames,integrity=dict(reader.integrity),decoder_metadata=reader.metadata['decoder'])
        guard();result['sha256_after']=guarded_hash(source,guard)
        result['source_checksum_unchanged']=result['sha256_after']==result['sha256_before']==row['sha256']
        result['full_contract_passed']=bool(result['eof'] and result['integrity']['fully_consumed'] and result['frames']>0 and
            result['frames']==result['video_messages']==result['decoder_frames'] and not result['timestamp_mismatches'] and not result['resolution_changes'] and result['source_checksum_unchanged'])
        result['status']='full_decode_verified' if result['full_contract_passed'] else 'full_contract_failed'
    except Exception as exc:
        result.update(status='deadline_stopped' if isinstance(exc,ProcessingDeadlineExceeded) else 'failed',reason=str(exc),exception=type(exc).__name__,full_contract_passed=False)
        if reader is not None:
            result.update(eof=reader.eof,video_messages=reader.video_messages,decoder_frames=reader.decoded_frames,integrity=dict(reader.integrity))
    result['resolutions']=dict(resolutions);result['quality']={side:value.summary() for side,value in quality.items()}
    return result

def aggregate(report):
    rows=report['results'];flags={side:collections.Counter() for side in ['left','right']};resolutions=collections.Counter()
    for row in rows:
        resolutions.update(row['resolutions'])
        for side in flags:flags[side].update(row['quality'][side]['flags'])
    return {'format':'recorded-mcap-full-decode-aggregate-v1','profile':PROFILE,'status':'full_decode_verified' if report.get('complete') and report.get('control_metadata_unchanged') is True and all(r.get('full_contract_passed') for r in rows) else 'incomplete_or_failed',
        'source_files_expected':report['source_files_expected'],'source_files_processed':len(rows),'source_files_passed':sum(r.get('full_contract_passed',False) for r in rows),
        'decoded_frames':sum(r['frames'] for r in rows),'timestamp_mismatches':sum(r['timestamp_mismatches'] for r in rows),
        'source_before_after_intake_hash_matches':sum(r.get('source_checksum_unchanged',False) for r in rows),'full_container_EOF_files':sum(r.get('eof',False) for r in rows),
        'fully_consumed_container_files':sum(r.get('integrity',{}).get('fully_consumed',False) for r in rows),
        'chunks_checked':sum(r.get('integrity',{}).get('chunks_checked',0) for r in rows),'chunks_with_nonzero_CRC_checked':sum(r.get('integrity',{}).get('chunks_with_crc',0) for r in rows),
        'files_with_nonzero_data_end_CRC_checked':sum(r.get('integrity',{}).get('data_end_crc_present',False) for r in rows if r.get('integrity',{}).get('fully_consumed')),
        'left_and_right_resolution_frame_counts':dict(resolutions),'frame_time_base_missing_exact_flush_matches':sum(r['frame_time_base_missing_exact_flush_matches'] for r in rows),
        'quality_flags':{side:dict(value) for side,value in flags.items()},'quality_flags_are_heuristic_not_invalidity_or_GT':True,
        'reader_sha256':report['reader_sha256'],'validator_sha256':report['validator_sha256'],'processing_guard_checks':report['processing_guard_checks'],
        'processing_authorized_until':'2026-11-05T23:59:00+09:00','permission_exception_for_this_validation_only':True,
        'raw_pixel_outputs':False,'original_files_written':False,'uploaded':False,'training_started':False,'automatic_expiry_registration':False,'automatic_delete':False,
        'capture_clock_verified':False,'base_retention_policy_changed':False,'sdk_image_clock_verified':False,'rectification_verified':False,'GNSS_available':False,'metric_depth_available':False,
        'limits':['Observational proprietary profile, not general SVO/SVO2 support','Current Windows/PyAV CPU environment only','CRC verification covers nonzero provided checksums; absent CRCs are not authentication','Heuristic image statistics are not model accuracy or physical defect ground truth','No 60-minute field test or GNSS/calibration/metric-depth validation']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--permission',default='data/recorded-drive-validation/permission.json');p.add_argument('--output',required=True);p.add_argument('--public-report');a=p.parse_args()
    guard,receipt=permission_guard(a.permission);inventory_path=workspace_path(a.inventory,exists=True)
    inventory_raw=inventory_path.read_bytes();inventory=json.loads(inventory_raw.decode('utf-8'));inventory_sha=hashlib.sha256(inventory_raw).hexdigest();rows=inventory['recordings']
    if not rows or len({row['source_path'] for row in rows})!=len(rows):raise ValueError('nonempty_unique_source_inventory_required')
    output=private_output_path(a.output)
    public=public_output_path(a.public_report) if a.public_report else None
    for row in rows:
        source=workspace_path(row['source_path'],exists=True)
        if output.is_relative_to(source.parent) or public is not None and public.is_relative_to(source.parent):raise ValueError('report_must_not_modify_source_directory')
    cv2.setNumThreads(2);start=time.monotonic()
    report={'format':'private-recorded-mcap-full-decode-v1','source_files_expected':len(rows),'permission':receipt,'permission_sha256':guard.permission_sha256,
        'source_inventory_sha256':inventory_sha,'reader_sha256':sha256(ROOT/'agent/recorded_mcap.py'),'validator_sha256':sha256(Path(__file__)),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'results':[],'complete':False,'processing_guard_checks':guard.checks}
    atomic_json(output,report)
    for row in rows:
        result=validate_one(row,guard);report['results'].append(result);report['processing_guard_checks']=guard.checks
        atomic_json(output,report)
        print(json.dumps({'files_processed':len(report['results']),'files_passed':sum(r.get('full_contract_passed',False) for r in report['results']),
            'frames':sum(r['frames'] for r in report['results']),'latest_status':result['status']}),flush=True)
        if result['status']=='deadline_stopped':break
    report['complete']=len(report['results'])==len(rows);report['finished_at_utc']=datetime.now(timezone.utc).isoformat();report['elapsed_seconds']=time.monotonic()-start
    report['control_metadata_unchanged']=control_metadata_unchanged(a.permission,inventory_path,report)
    report['processing_guard_checks']=guard.checks;atomic_json(output,report)
    summary=aggregate(report)
    if public:atomic_json(public,summary)
    print(json.dumps({'status':summary['status'],'files':summary['source_files_processed'],'frames':summary['decoded_frames'],'deadline_checks':guard.checks}),flush=True)
    return 0 if summary['status']=='full_decode_verified' else 2
if __name__=='__main__':raise SystemExit(main())
