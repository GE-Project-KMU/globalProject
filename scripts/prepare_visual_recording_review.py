"""Explicitly authorized local visual-review sampling; never labels or runs ML.

Original recordings are read-only. All PNGs, contact sheets, clocks and hashes
remain private. A failed/partial session is preserved and cannot be overwritten.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.paths import workspace_path, atomic_json
from agent.recorded_mcap import RecordedMcapReader

PRIVATE = ROOT / 'data/recorded-drive-ai-labeling'
DEFAULT_PERMISSION = PRIVATE / 'permission-20261108.json'
MAX_DEADLINE = datetime.fromisoformat('2026-11-14T14:59:00+00:00')


def digest(path, guard=lambda: None):
    guard()
    result = hashlib.sha256()
    with workspace_path(path, exists=True).open('rb') as stream:
        while True:
            guard()
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            result.update(chunk)
    guard()
    return result.hexdigest()


class PermissionGuard:
    def __init__(self, path, *, now=None):
        self.path = workspace_path(path, exists=True)
        self.raw = self.path.read_bytes()
        self.permission = json.loads(self.raw)
        p = self.permission
        if (p.get('format') != 'recorded-drive-ai-labeling-permission-v1'
                or p.get('scope') != 'visual_ai_annotation_only'
                or p.get('source') != 'explicit user reply in this chat'
                or p.get('source_directory') != '데이터'):
            raise ValueError('explicit_visual_review_permission_required')
        for key in ['visual_review_authorized', 'assistant_visual_review_authorized',
                    'raw_pixel_outputs_authorized', 'AI_labeling_authorized']:
            if p.get(key) is not True:
                raise ValueError('permission_flag_required:' + key)
        for key in ['training_authorized', 'training_run_authorized',
                    'evaluation_ground_truth_eligible', 'external_upload_authorized',
                    'third_party_storage_upload_authorized', 'automatic_delete_authorized',
                    'automatic_retention_registration', 'base_retention_policy_changed',
                    'original_capture_date_changed', 'human_reviewed']:
            if p.get(key) is not False:
                raise ValueError('permission_scope_mismatch:' + key)
        self.deadline = datetime.fromisoformat(p['authorized_until_utc'])
        local = datetime.fromisoformat(p['authorized_until'])
        if (self.deadline.tzinfo is None or local.tzinfo is None
                or local != self.deadline or self.deadline > MAX_DEADLINE):
            raise ValueError('permission_deadline_invalid_or_extended')
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.sha256 = hashlib.sha256(self.raw).hexdigest()
        self.checks = 0
        self.next_permission_check = time.monotonic() + 1
        self()

    def __call__(self):
        self.checks += 1
        now = self.now()
        if now.tzinfo is None:
            raise ValueError('timezone_aware_processing_clock_required')
        if now >= self.deadline:
            raise RuntimeError('visual_review_permission_expired')
        if time.monotonic() >= self.next_permission_check:
            if self.path.read_bytes() != self.raw:
                raise ValueError('permission_changed_during_processing')
            self.next_permission_check = time.monotonic() + 1

    def unchanged(self):
        self()
        if self.path.read_bytes() != self.raw:
            raise ValueError('permission_changed_during_processing')


def selection_indices(count):
    if type(count) is not int or count < 5:
        raise ValueError('at_least_five_original_frames_required')
    return [(count - 1) * k // 4 for k in range(5)]


def reviewed_inventory(report):
    rows = report.get('results', [])
    if (report.get('complete') is not True or report.get('source_files_expected') != 16
            or len(rows) != 16 or len({r['source_path'] for r in rows}) != 16
            or sum(r['frames'] for r in rows) != 14753):
        raise ValueError('original_full_validation_16_files_14753_frames_required')
    source_root = ROOT / '데이터'
    for r in rows:
        path = workspace_path(r['source_path'], exists=True)
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError('original_source_directory_required')
        if (r.get('full_contract_passed') is not True or r.get('eof') is not True
                or r.get('integrity', {}).get('fully_consumed') is not True
                or r.get('timestamp_mismatches') != 0
                or not (r['frames'] == r['video_messages'] == r['decoder_frames'])
                or r['sha256_before'] != r['expected_sha256']
                or r['sha256_after'] != r['expected_sha256']):
            raise ValueError('original_validation_contract_not_verified')
        selection_indices(r['frames'])
    return rows


def write_png(path, image, guard):
    import cv2
    import numpy as np
    guard()
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('original_BGR_uint8_image_required')
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('PNG_encoding_failed')
    guard()
    with path.open('xb') as stream:
        stream.write(encoded.tobytes())
    guard()
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(encoded).hexdigest(),
            'width': int(image.shape[1]), 'height': int(image.shape[0])}


def extract_one(row, clip, output, guard, *, reader_factory=RecordedMcapReader,
                include_right=False):
    import cv2
    import numpy as np
    source = workspace_path(row['source_path'], exists=True)
    if (source.stat().st_size != row['expected_bytes']
            or digest(source, guard) != row['expected_sha256']):
        raise ValueError('source_before_checksum_or_size_mismatch')
    selected = selection_indices(row['frames'])
    panels, samples, count = [], [], 0
    with reader_factory(source, processing_guard=guard) as reader:
        for frame in reader:
            guard()
            if (frame['video_message_index'] != count
                    or frame['timestamp_ns'] != frame['mcap_log_time_ns']):
                raise ValueError('frame_identity_or_timestamp_mismatch')
            if count in selected:
                prefix = f'clip-{clip:02d}-frame-{count:06d}'
                item = {key: frame[key] for key in ['video_message_index',
                    'mcap_log_time_ns', 'mcap_publish_time_ns', 'decoder_pts',
                    'decoder_time_base', 'timestamp_match_basis']}
                item.update(clip=clip, frame_index=count, all_trip_group='trip-1',
                            utc=None, gnss=None, depth_m=None, rectification='unknown',
                            expires_at=guard.deadline.isoformat(), labels=[],
                            human_reviewed=False, model_predictions_used=False)
                item['left'] = write_png(output / (prefix + '-left.png'), frame['left'], guard)
                if include_right:
                    item['right'] = write_png(output / (prefix + '-right.png'), frame['right'], guard)
                h, w = frame['left'].shape[:2]
                thumb = cv2.resize(frame['left'], (480, max(1, round(h * 480 / w))))
                panel = np.zeros((thumb.shape[0] + 30, 480, 3), np.uint8)
                panel[30:] = thumb
                cv2.putText(panel, f'clip {clip:02d} / frame {count}', (8, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                panels.append(panel)
                samples.append(item)
            count += 1
        if (count != row['frames'] or reader.eof is not True
                or reader.integrity.get('fully_consumed') is not True
                or reader.video_messages != count or reader.decoded_frames != count
                or len(samples) != 5):
            raise ValueError('full_EOF_frame_count_contract_failed')
    if digest(source, guard) != row['expected_sha256']:
        raise ValueError('source_after_checksum_mismatch')
    if len({p.shape for p in panels}) != 1:
        raise ValueError('sample_resolution_changes_unsupported')
    sheet = write_png(output / f'clip-{clip:02d}-overview.png', np.concatenate(panels, axis=1), guard)
    return {'clip': clip, 'source_path': row['source_path'],
            'source_sha256': row['expected_sha256'], 'full_decoded_frames': count,
            'EOF_verified': True, 'source_before_after_hash_verified': True,
            'overview': sheet, 'samples': samples}


def prepare(permission=DEFAULT_PERMISSION, *, include_right=False, approved=False,
            report_path=ROOT / 'data/recorded-drive-validation/private-full-decode.json',
            reader_factory=RecordedMcapReader, now=None):
    if not approved:
        raise ValueError('explicit_local_extraction_approval_required')
    guard = PermissionGuard(permission, now=now)
    if include_right and guard.permission.get('right_view_authorized') is not True:
        raise ValueError('separate_right_view_approval_required')
    report_path = workspace_path(report_path, exists=True)
    raw = report_path.read_bytes()
    report = json.loads(raw)
    rows = reviewed_inventory(report)
    reader_sha = digest(ROOT / 'agent/recorded_mcap.py', guard)
    if reader_sha != report['reader_sha256']:
        raise ValueError('previously_validated_reader_source_changed')
    output = workspace_path(PRIVATE / 'session-v1')
    output.mkdir(parents=False, exist_ok=False)
    receipt = {'format': 'private-recorded-visual-review-v1', 'status': 'starting',
               'permission_sha256': guard.sha256,
               'original_validation_sha256': hashlib.sha256(raw).hexdigest(),
               'reader_sha256': reader_sha, 'extractor_sha256': digest(Path(__file__), guard),
               'expires_at': guard.deadline.isoformat(), 'all_trip_group': 'trip-1',
               'model_predictions_used': False, 'labels_written': 0,
               'human_reviewed': False, 'independent_TEST_eligible': False,
               'training_authorized': False, 'export_enabled': False,
               'external_upload_authorized': False, 'automatic_delete': False,
               'original_capture_clock_verified': False, 'GNSS': None, 'depth_m': None,
               'right_included': include_right, 'clips': []}
    target = output / 'frame-index.json'
    try:
        atomic_json(target, receipt)
        for i, row in enumerate(rows, 1):
            guard.unchanged()
            if report_path.read_bytes() != raw:
                raise ValueError('original_validation_metadata_changed')
            receipt['status'] = 'running'
            receipt['clips'].append(extract_one(row, i, output, guard,
                reader_factory=reader_factory, include_right=include_right))
            atomic_json(target, receipt)
        guard.unchanged()
        if (report_path.read_bytes() != raw
                or digest(ROOT / 'agent/recorded_mcap.py', guard) != reader_sha
                or digest(Path(__file__), guard) != receipt['extractor_sha256']):
            raise ValueError('processing_metadata_or_reader_changed')
        receipt.update(status='completed', sampled_left_frames=80, source_files=16,
                       full_decoded_frames=sum(c['full_decoded_frames'] for c in receipt['clips']))
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                       partial_outputs_preserved=True, automatic_retry=False)
        raise
    finally:
        receipt['processing_guard_checks'] = guard.checks
        atomic_json(target, receipt)
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--permission', default=str(DEFAULT_PERMISSION))
    p.add_argument('--extract-approved', action='store_true')
    p.add_argument('--include-right', action='store_true')
    a = p.parse_args()
    result = prepare(a.permission, approved=a.extract_approved, include_right=a.include_right)
    print(json.dumps({'status': 'completed', 'private_index': str(result)}))
