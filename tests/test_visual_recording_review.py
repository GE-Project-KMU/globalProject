"""Synthetic-only contracts: no original recordings, ML, GPU or network."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import prepare_visual_recording_review as m


def permission(tmp_path, **changes):
    d = {'format': 'recorded-drive-ai-labeling-permission-v1',
         'scope': 'visual_ai_annotation_only', 'source': 'explicit user reply in this chat',
         'source_directory': '데이터', 'authorized_until': '2026-11-14T23:59:00+09:00',
         'authorized_until_utc': '2026-11-14T14:59:00+00:00'}
    for k in ['visual_review_authorized', 'assistant_visual_review_authorized',
              'raw_pixel_outputs_authorized', 'AI_labeling_authorized']:
        d[k] = True
    for k in ['training_authorized', 'training_run_authorized', 'evaluation_ground_truth_eligible',
              'external_upload_authorized', 'third_party_storage_upload_authorized',
              'automatic_delete_authorized', 'automatic_retention_registration',
              'base_retention_policy_changed', 'original_capture_date_changed', 'human_reviewed']:
        d[k] = False
    d.update(changes)
    path = tmp_path / 'synthetic-permission.json'
    path.write_text(json.dumps(d), encoding='utf-8')
    return path


NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)


def test_explicit_scope_deadline_and_mutation(tmp_path):
    path = permission(tmp_path)
    clock = [NOW]
    g = m.PermissionGuard(path, now=lambda: clock[0])
    g.unchanged()
    path.write_text('{}')
    with pytest.raises(ValueError, match='permission_changed'):
        g.unchanged()
    clock[0] = m.MAX_DEADLINE
    with pytest.raises(RuntimeError, match='expired'):
        g()


@pytest.mark.parametrize('change', [
    {'training_run_authorized': True}, {'external_upload_authorized': True},
    {'raw_pixel_outputs_authorized': False}, {'human_reviewed': True},
    {'authorized_until': '2026-11-15T23:59:00+09:00',
     'authorized_until_utc': '2026-11-15T14:59:00+00:00'},
    {'scope': 'training'}])
def test_unauthorized_scope_or_extension_refused(tmp_path, change):
    with pytest.raises(ValueError):
        m.PermissionGuard(permission(tmp_path, **change), now=lambda: NOW)


def test_balanced_selection_uses_original_indices():
    assert m.selection_indices(101) == [0, 25, 50, 75, 100]
    assert m.selection_indices(7) == [0, 1, 3, 4, 6]
    with pytest.raises(ValueError):
        m.selection_indices(4)


class FakeReader:
    def __init__(self, source, processing_guard):
        self.guard = processing_guard
        self.eof = False
        self.integrity = {'fully_consumed': False}
        self.video_messages = self.decoded_frames = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        for i in range(7):
            self.guard()
            self.video_messages += 1
            self.decoded_frames += 1
            yield {'left': np.full((12, 20, 3), i, np.uint8),
                   'right': np.full((12, 20, 3), 100 + i, np.uint8),
                   'video_message_index': i, 'timestamp_ns': 1000 + i,
                   'mcap_log_time_ns': 1000 + i, 'mcap_publish_time_ns': 2000 + i,
                   'decoder_pts': 1000 + i, 'decoder_time_base': [1, 1000000000],
                   'timestamp_match_basis': 'synthetic_exact'}
        self.eof = True
        self.integrity['fully_consumed'] = True


def setup_clip(tmp_path):
    source = tmp_path / 'synthetic-source.bin'
    source.write_bytes(b'no recording pixels: synthetic fixture')
    output = tmp_path / 'private-session'
    output.mkdir()
    row = {'source_path': str(source), 'expected_bytes': source.stat().st_size,
           'expected_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'frames': 7}
    guard = m.PermissionGuard(permission(tmp_path), now=lambda: NOW)
    return row, output, guard


def test_full_decode_selection_PNG_identity_and_no_default_right(tmp_path):
    row, output, guard = setup_clip(tmp_path)
    result = m.extract_one(row, 1, output, guard, reader_factory=FakeReader)
    assert result['EOF_verified'] and result['full_decoded_frames'] == 7
    assert [s['frame_index'] for s in result['samples']] == [0, 1, 3, 4, 6]
    assert len(list(output.glob('*.png'))) == 6
    assert not list(output.glob('*right*'))
    for sample in result['samples']:
        png = m.ROOT / sample['left']['path']
        image = cv2.imread(str(png))
        assert image.shape == (12, 20, 3)
        assert np.all(image == sample['frame_index'])
        assert hashlib.sha256(png.read_bytes()).hexdigest() == sample['left']['sha256']
        assert sample['utc'] is None and sample['depth_m'] is None
        assert sample['labels'] == [] and sample['human_reviewed'] is False
    assert (m.ROOT / result['overview']['path']).exists()
    with pytest.raises(FileExistsError):
        m.extract_one(row, 1, output, guard, reader_factory=FakeReader)


def test_hash_failure_prevents_decoder_construction(tmp_path):
    row, output, guard = setup_clip(tmp_path)
    row['expected_sha256'] = '0' * 64
    def forbidden(*a, **kw):
        pytest.fail('decoder constructed before original hash validation')
    with pytest.raises(ValueError, match='source_before'):
        m.extract_one(row, 1, output, guard, reader_factory=forbidden)
    assert list(output.iterdir()) == []


def test_missing_EOF_rejected_even_if_all_selected_samples_written(tmp_path):
    row, output, guard = setup_clip(tmp_path)
    class NoEOF(FakeReader):
        def __iter__(self):
            yield from super().__iter__()
            self.eof = False
    with pytest.raises(ValueError, match='full_EOF'):
        m.extract_one(row, 1, output, guard, reader_factory=NoEOF)


def test_guard_is_applied_through_last_original_frame(tmp_path):
    row, output, guard = setup_clip(tmp_path)
    calls = [0]
    class StopReader(FakeReader):
        def __iter__(self):
            for sample in super().__iter__():
                calls[0] += 1
                if calls[0] == 3:
                    guard.now = lambda: m.MAX_DEADLINE
                yield sample
    with pytest.raises(RuntimeError, match='expired'):
        m.extract_one(row, 1, output, guard, reader_factory=StopReader)


def test_no_approval_or_right_permission_has_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'PRIVATE', tmp_path)
    with pytest.raises(ValueError, match='explicit_local'):
        m.prepare(approved=False)
    with pytest.raises(ValueError, match='right_view'):
        m.prepare(permission(tmp_path), approved=True, include_right=True, now=lambda: NOW)
    assert not (tmp_path / 'session-v1').exists()


def test_full_synthetic_16_clip_orchestration_count_and_fresh_session_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    private = tmp_path / 'data/recorded-drive-ai-labeling'
    private.mkdir(parents=True)
    monkeypatch.setattr(m, 'PRIVATE', private)
    source_root = tmp_path / '데이터'
    source_root.mkdir()
    agent = tmp_path / 'agent'
    agent.mkdir()
    (agent / 'recorded_mcap.py').write_bytes(b'synthetic reader identity only')
    rows = []
    for i in range(16):
        p = source_root / f'fixture-{i}.bin'
        p.write_bytes(f'synthetic source {i}'.encode())
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        count = 5 if i < 15 else 14753 - 75
        rows.append({'source_path': str(p), 'expected_bytes': p.stat().st_size,
                     'expected_sha256': h, 'sha256_before': h, 'sha256_after': h,
                     'frames': count, 'video_messages': count, 'decoder_frames': count,
                     'full_contract_passed': True, 'eof': True,
                     'integrity': {'fully_consumed': True}, 'timestamp_mismatches': 0})
    report = tmp_path / 'synthetic-full-validation.json'
    report.write_text(json.dumps({'complete': True, 'source_files_expected': 16,
        'reader_sha256': hashlib.sha256((agent / 'recorded_mcap.py').read_bytes()).hexdigest(),
        'results': rows}), encoding='utf-8')
    counts = {r['source_path']: r['frames'] for r in rows}
    class CountReader(FakeReader):
        def __init__(self, source, processing_guard):
            super().__init__(source, processing_guard)
            self.count = counts[str(source)]
        def __iter__(self):
            template = next(iter(FakeReader.__iter__(self)))
            self.video_messages = self.decoded_frames = 0
            for i in range(self.count):
                self.guard()
                self.video_messages += 1
                self.decoded_frames += 1
                yield {**template, 'video_message_index': i, 'timestamp_ns': 1000+i,
                       'mcap_log_time_ns': 1000+i, 'decoder_pts': 1000+i}
            self.eof = True
            self.integrity['fully_consumed'] = True
    consent = permission(private)
    path = m.prepare(consent, approved=True, report_path=report,
                     reader_factory=CountReader, now=lambda: NOW)
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['status'] == 'completed'
    assert result['full_decoded_frames'] == 14753
    assert result['sampled_left_frames'] == 80
    assert len(list(path.parent.glob('*-left.png'))) == 80
    assert len(list(path.parent.glob('*-overview.png'))) == 16
    assert result['labels_written'] == 0 and result['independent_TEST_eligible'] is False
    with pytest.raises(FileExistsError):
        m.prepare(consent, approved=True, report_path=report,
                  reader_factory=CountReader, now=lambda: NOW)
