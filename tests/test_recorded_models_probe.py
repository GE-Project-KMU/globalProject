"""Permission, frozen-sample and no-pixel-output contracts using synthetic data."""
from datetime import datetime, timezone
import json
import shutil
import uuid

import numpy as np
import pytest
from agent.paths import ROOT, atomic_json, sha256
from scripts import probe_recorded_models as tool


@pytest.fixture
def folder(monkeypatch):
    original_guard = tool.PeriodGuard
    monkeypatch.setattr(tool, 'PeriodGuard', lambda value: original_guard(value, now=lambda: datetime(2026, 11, 4, tzinfo=timezone.utc)))
    path = ROOT/'data/recorded-drive-validation/.tests'/uuid.uuid4().hex
    path.mkdir(parents=True)
    yield path
    assert path.resolve().is_relative_to((ROOT/'data/recorded-drive-validation/.tests').resolve())
    shutil.rmtree(path)


def receipt(deadline='2026-11-05T14:59:00+00:00'):
    return {'scope': 'this recording validation only', 'authorized_until_utc': deadline}


def test_permission_rejects_exact_deadline_and_missing_timezone():
    instant = datetime(2026, 11, 5, 14, 59, tzinfo=timezone.utc)
    with pytest.raises(PermissionError, match='period_ended'):
        tool.PeriodGuard(receipt(instant.isoformat()), now=lambda: instant)
    with pytest.raises(ValueError, match='timezone_aware'):
        tool.PeriodGuard(receipt('2099-01-01T00:00:00'))
    with pytest.raises(ValueError, match='permission_required'):
        tool.PeriodGuard({'scope': 'unrelated permission'})
    with pytest.raises(ValueError, match='exceeds_explicit'):
        tool.PeriodGuard(receipt('2099-01-01T00:00:00+00:00'))


def test_permission_is_rechecked_during_hash_stream(folder):
    source = folder/'bytes'; source.write_bytes(b'x' * 2_000_000)
    calls = []
    def guard():
        calls.append(1)
        if len(calls) == 3:
            raise PermissionError('expired')
    with pytest.raises(PermissionError):
        tool.checked_sha(source, guard)
    assert len(calls) == 3 and source.stat().st_size == 2_000_000


@pytest.mark.parametrize('count,expected', [(1, [0]), (3, [0, 1, 2]), (10, [0, 2, 4, 6, 9]), (100, [0, 24, 49, 74, 99])])
def test_temporal_sampling_is_fixed_before_predictions(count, expected):
    assert tool.selected_indices(count) == expected


def plan_fixture(folder):
    source = folder/'private-SN999-recording.svo2'; source.write_bytes(b'synthetic nonvideo')
    source_name = source.relative_to(ROOT).as_posix()
    inventory = folder/'inventory.json'
    row = {'source_path': source_name, 'sha256': sha256(source), 'size_bytes': source.stat().st_size}
    atomic_json(inventory, {'recordings': [row]})
    audit = folder/'audit.json'
    atomic_json(audit, {'handoff_manifest_sha256': sha256(inventory), 'source_unchanged': True,
                       'recordings': [row | {'audit': {'video_recorded_time': {'message_count': 3}}}]})
    permission = folder/'permission.json'; atomic_json(permission, receipt())
    models = {}
    for task in ('pothole', 'lane', 'redaction'):
        weights = folder/(task+'.onnx'); weights.write_bytes(b'fake weights '+task.encode())
        manifest = folder/(task+'.json')
        atomic_json(manifest, {'task': task, 'weights': weights.relative_to(ROOT).as_posix(), 'sha256': sha256(weights)})
        models[task] = manifest
    plan_path = folder/'plan.json'
    tool.create_plan(inventory, audit, permission, models, plan_path)
    return plan_path, source, models


class FakeModel:
    backend = 'synthetic-test'
    backend_version = 'not-real-inference'
    def __init__(self, manifest):
        self.task = tool.read_json(manifest)['task']
    def predict(self, image):
        if self.task == 'lane':
            return np.zeros(image.shape[:2], np.uint8)
        return [{'class': 'pothole' if self.task == 'pothole' else 'face', 'confidence': .5, 'bbox_xywh_px': [1, 1, 2, 2]}]


class FakeReader:
    def __init__(self, path, processing_guard):
        self.guard = processing_guard; self.eof = False
        self.integrity = {'fully_consumed': False}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __iter__(self):
        for index in range(3):
            self.guard()
            yield {'video_message_index': index, 'timestamp_ns': 100+index,
                   'left': np.zeros((16, 16, 3), np.uint8)}
        self.eof = True; self.integrity['fully_consumed'] = True


def test_fake_pipeline_preserves_original_and_excludes_private_data_from_aggregate(folder):
    plan, source, _ = plan_fixture(folder)
    before = sha256(source)
    public = folder/'public.json'
    result = tool.run_plan(plan, public_report=public, model_factory=FakeModel, reader_factory=FakeReader)
    assert result['inferred_left_frames'] == 3 and result['all_files_decoded_through_eof']
    assert result['candidate_boxes_by_class'] == {'pothole': 3, 'face': 3}
    assert not result['accuracy_measured'] and not result['human_privacy_approval_granted']
    assert not result['server_observations_created'] and not result['raw_pixel_files_written']
    assert sha256(source) == before
    serialized = public.read_text()
    assert source.name not in serialized and before not in serialized and 'timestamp_ns' not in serialized
    assert not list(folder.glob('*.png')) and not list(folder.glob('*.mp4'))
    with pytest.raises(FileExistsError):
        tool.run_plan(plan, model_factory=FakeModel, reader_factory=FakeReader)


def test_changed_source_fails_before_decoder_open(folder):
    plan, source, _ = plan_fixture(folder)
    source.write_bytes(b'changed')
    def must_not_open(*args, **kwargs):
        pytest.fail('decoder opened changed source')
    with pytest.raises(ValueError, match='recording_source_changed'):
        tool.run_plan(plan, model_factory=FakeModel, reader_factory=must_not_open)
    assert not (folder/'private-results.json').exists()


def test_changed_model_fails_before_model_load(folder):
    plan, _, models = plan_fixture(folder)
    models['pothole'].write_text('{}')
    def must_not_load(*args, **kwargs):
        pytest.fail('model loaded after its freeze changed')
    with pytest.raises(ValueError, match='model_changed_since_plan'):
        tool.run_plan(plan, model_factory=must_not_load)


def test_frozen_sampling_cannot_be_replaced_after_scores(folder):
    plan, _, _ = plan_fixture(folder)
    value = tool.read_json(plan)
    value['sources'][0]['selected_video_message_indices'] = [0]
    atomic_json(plan, value)
    with pytest.raises(ValueError, match='frozen_sampling_rule'):
        tool.run_plan(plan, model_factory=FakeModel, reader_factory=FakeReader)


def test_permission_change_and_expiry_block_replay(folder):
    plan, _, _ = plan_fixture(folder)
    value = tool.read_json(plan)
    atomic_json(value['permission'], receipt('2000-01-01T00:00:00+00:00'))
    with pytest.raises(PermissionError):
        tool.run_plan(plan, model_factory=FakeModel, reader_factory=FakeReader)
    atomic_json(value['permission'], receipt('2026-11-05T14:58:00+00:00'))
    with pytest.raises(ValueError, match='permission_changed'):
        tool.run_plan(plan, model_factory=FakeModel, reader_factory=FakeReader)


def test_period_end_during_inference_stops_next_model_and_output(folder, monkeypatch):
    plan, _, _ = plan_fixture(folder)
    state = {'ended': False, 'calls': []}
    def guard():
        if state['ended']:
            raise PermissionError('recording_validation_period_ended')
    monkeypatch.setattr(tool, 'PeriodGuard', lambda value: guard)
    class EndingModel(FakeModel):
        def predict(self, image):
            state['calls'].append(self.task)
            prediction = super().predict(image)
            state['ended'] = True
            return prediction
    with pytest.raises(PermissionError, match='period_ended'):
        tool.run_plan(plan, model_factory=EndingModel, reader_factory=FakeReader)
    assert state['calls'] == ['pothole']
    assert not (folder/'private-results.json').exists()


def test_private_plan_and_results_cannot_be_written_into_docs(folder):
    _, _, models = plan_fixture(folder)
    with pytest.raises(ValueError, match='private_recording_validation'):
        tool.create_plan(folder/'inventory.json', folder/'audit.json', folder/'permission.json', models,
                         ROOT/'docs/reports/forbidden-private-plan.json')
    with pytest.raises(ValueError, match='private_recording_validation'):
        tool.run_plan(ROOT/'docs/reports/forbidden-private-plan.json')
    assert not (ROOT/'docs/reports/forbidden-private-plan.json').exists()


def test_plan_change_during_inference_cannot_publish_misbound_result(folder):
    plan, _, _ = plan_fixture(folder)
    class MutatingModel(FakeModel):
        def predict(self, image):
            value = tool.read_json(plan); value['tampered'] = True
            atomic_json(plan, value)
            return super().predict(image)
    with pytest.raises(ValueError, match='plan_changed_during_validation'):
        tool.run_plan(plan, model_factory=MutatingModel, reader_factory=FakeReader)
    assert not (folder/'private-results.json').exists()
    assert not (folder/'aggregate.json').exists()
