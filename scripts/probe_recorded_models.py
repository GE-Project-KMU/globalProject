"""Frozen, local-only recording inference probe; no pixels are persisted."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from agent.paths import workspace_path, atomic_json, sha256
from agent.models import OnnxModel, frame_quality, runtime_metadata
from agent.recorded_mcap import RecordedMcapReader
MAX_AUTHORIZED = datetime.fromisoformat('2026-11-05T14:59:00+00:00')


class PeriodGuard:
    def __init__(self, receipt, now=None):
        self.now = now or (lambda: datetime.now(timezone.utc))
        if receipt.get('scope') != 'this recording validation only':
            raise ValueError('recording_validation_permission_required')
        self.deadline = datetime.fromisoformat(receipt['authorized_until_utc'])
        if self.deadline.tzinfo is None:
            raise ValueError('timezone_aware_permission_deadline_required')
        if self.deadline > MAX_AUTHORIZED:
            raise ValueError('deadline_exceeds_explicit_user_authorization')
        self()

    def __call__(self):
        current = self.now()
        if current.tzinfo is None:
            raise ValueError('timezone_aware_processing_clock_required')
        if current >= self.deadline:
            raise PermissionError('recording_validation_period_ended')


def checked_sha(path, guard):
    import hashlib
    value = hashlib.sha256()
    guard()
    with workspace_path(path, exists=True).open('rb') as stream:
        while True:
            guard()
            block = stream.read(1024 * 1024)
            if not block:
                break
            value.update(block)
    guard()
    return value.hexdigest()


def selected_indices(count):
    if type(count) is not int or count < 1:
        raise ValueError('positive_audited_message_count_required')
    return sorted({k * (count - 1) // 4 for k in range(5)})


def read_json(path):
    return json.loads(workspace_path(path, exists=True).read_text(encoding='utf-8'))


def create_plan(inventory_path, audit_path, permission_path, models, output):
    output = workspace_path(output)
    if output.exists():
        raise FileExistsError('new_plan_path_required')
    receipt = read_json(permission_path)
    guard = PeriodGuard(receipt)
    inventory, audit = read_json(inventory_path), read_json(audit_path)
    if audit['handoff_manifest_sha256'] != sha256(inventory_path) or not audit['source_unchanged']:
        raise ValueError('audit_inventory_binding_required')
    audited = {r['source_path']: r for r in audit['recordings']}
    sources = []
    for row in inventory['recordings']:
        a = audited.get(row['source_path'])
        if not a or (a['sha256'], a['size_bytes']) != (row['sha256'], row['size_bytes']):
            raise ValueError('audited_source_identity_mismatch')
        count = a['audit']['video_recorded_time']['message_count']
        sources.append({'source_path': row['source_path'], 'sha256': row['sha256'],
                        'size_bytes': row['size_bytes'], 'audited_video_messages': count,
                        'selected_video_message_indices': selected_indices(count)})
    if not sources or len(sources) != len(audited):
        raise ValueError('complete_audited_inventory_required')
    frozen_models = {}
    for task, path in models.items():
        manifest = read_json(path)
        if task != manifest['task']:
            raise ValueError('model_task_mismatch')
        if checked_sha(manifest['weights'], guard) != manifest['sha256']:
            raise ValueError('model_weight_checksum_mismatch')
        frozen_models[task] = {'manifest': workspace_path(path).relative_to(ROOT).as_posix(),
                              'manifest_sha256': sha256(path), 'weights': manifest['weights'],
                              'weights_sha256': manifest['sha256'],
                              'confidence': manifest.get('confidence'),
                              'paint_threshold': manifest.get('paint_threshold')}
    plan = {'format': 'recorded-drive-model-probe-plan-v1',
            'permission': workspace_path(permission_path).relative_to(ROOT).as_posix(),
            'permission_sha256': sha256(permission_path), 'authorized_until_utc': receipt['authorized_until_utc'],
            'inventory_sha256': sha256(inventory_path), 'audit_sha256': sha256(audit_path),
            'models': frozen_models, 'sources': sources,
            'sampling': 'five equally spaced message indices including endpoints; floor(k*(N-1)/4), deduplicated',
            'view': 'left_only', 'labels_available': False, 'model_selection_performed': False,
            'training_performed': False, 'raw_pixel_files_written': False,
            'script_sha256': sha256(Path(__file__)),
            'reader_sha256': sha256(ROOT/'agent/recorded_mcap.py')}
    atomic_json(output, plan)
    return plan


def check_source(row, guard):
    path = workspace_path(row['source_path'], exists=True)
    stat = path.stat()
    if stat.st_size != row['size_bytes'] or checked_sha(path, guard) != row['sha256']:
        raise ValueError('recording_source_changed')
    return path, stat


def summarize(results, plan):
    samples = [s for r in results for s in r['samples']]
    timings = {task: {'median_ms': float(np.median([s['latency_ms'][task] for s in samples])),
                     'p95_ms': float(np.percentile([s['latency_ms'][task] for s in samples], 95))}
               for task in plan['models']}
    counts = Counter(d['class'] for s in samples for d in s['pothole_candidates'] + s['privacy_candidates'])
    quality = Counter(reason for s in samples for reason in s['quality']['reasons'])
    return {'format': 'recorded-drive-model-probe-aggregate-v1',
            'status': 'bounded_local_inference_completed_not_accuracy_validation',
            'source_files': len(results), 'inferred_left_frames': len(samples),
            'all_original_checksums_unchanged': all(r['original_unchanged'] for r in results),
            'all_files_decoded_through_eof': all(r['eof'] and r['integrity']['fully_consumed'] for r in results),
            'sample_selection': plan['sampling'], 'model_weights_sha256': {k: v['weights_sha256'] for k, v in plan['models'].items()},
            'frozen_plan_sha256': None, 'latency': timings,
            'candidate_boxes_by_class': dict(counts),
            'frames_with_pothole_candidates': sum(bool(s['pothole_candidates']) for s in samples),
            'frames_with_privacy_candidates': sum(bool(s['privacy_candidates']) for s in samples),
            'frames_passing_basic_image_quality': sum(s['quality']['usable'] for s in samples),
            'basic_quality_reason_counts': dict(quality),
            'lane_paint_pixel_fraction_mean': float(np.mean([s['lane_paint_pixel_fraction'] for s in samples])),
            'lane_paint_pixel_fraction_min': min(s['lane_paint_pixel_fraction'] for s in samples),
            'lane_paint_pixel_fraction_max': max(s['lane_paint_pixel_fraction'] for s in samples),
            'authorized_until_utc': plan['authorized_until_utc'],
            'labels_available': False, 'accuracy_measured': False, 'privacy_recall_measured': False,
            'physical_depth_or_wear_measured': False, 'utc_or_gnss_generated': False,
            'raw_pixel_files_written': False, 'server_observations_created': False,
            'human_privacy_approval_granted': False, 'automatic_deletion_registered': False,
            'training_performed': False, 'model_thresholds_changed': False,
            'limits': ['Candidates are unreviewed predictions, not confirmed defects or unique objects',
                       'Lane paint pixel fraction is image coverage, not wear or physical road retention',
                       'No GT labels: no AP, precision, recall or masking-completeness claim',
                       'Five temporal samples per file are an execution probe, not a representative accuracy set',
                       '2D observed-profile MCAP decoder; no SDK IMAGE clock, rectification, GNSS or metric depth verification',
                       'CPU timings are measurements on this local machine; no deployment throughput claim']}


def run_plan(plan_path, *, public_report=None, model_factory=OnnxModel, reader_factory=RecordedMcapReader):
    plan_path = workspace_path(plan_path, exists=True)
    output = plan_path.parent/'private-results.json'
    public_report = workspace_path(public_report) if public_report else None
    if output.exists() or public_report and public_report.exists():
        raise FileExistsError('new_validation_output_required')
    plan = read_json(plan_path)
    receipt = read_json(plan['permission'])
    guard = PeriodGuard(receipt)
    if sha256(plan['permission']) != plan['permission_sha256'] or receipt['authorized_until_utc'] != plan['authorized_until_utc']:
        raise ValueError('permission_changed_since_plan')
    if sha256(Path(__file__)) != plan['script_sha256'] or sha256(ROOT/'agent/recorded_mcap.py') != plan['reader_sha256']:
        raise ValueError('probe_source_changed_since_plan')
    models = {}
    for task, spec in plan['models'].items():
        if checked_sha(spec['manifest'], guard) != spec['manifest_sha256'] or checked_sha(spec['weights'], guard) != spec['weights_sha256']:
            raise ValueError('model_changed_since_plan')
        models[task] = model_factory(spec['manifest'])
    if set(models) != {'pothole', 'lane', 'redaction'}:
        raise ValueError('three_required_models_missing')
    started, results = time.perf_counter(), []
    for row in plan['sources']:
        path, before = check_source(row, guard)
        wanted = set(row['selected_video_message_indices'])
        if sorted(wanted) != selected_indices(row['audited_video_messages']):
            raise ValueError('frozen_sampling_rule_mismatch')
        samples, count = [], 0
        with reader_factory(path, processing_guard=guard) as reader:
            for frame in reader:
                guard()
                count += 1
                index = frame['video_message_index']
                if index not in wanted:
                    continue
                image = frame['left']
                predictions, latencies = {}, {}
                for task, model in models.items():
                    guard()
                    tick = time.perf_counter()
                    predictions[task] = model.predict(image)
                    latencies[task] = 1000 * (time.perf_counter() - tick)
                    guard()
                mask = predictions['lane']
                if mask.shape != image.shape[:2] or not np.isin(mask, [0, 1]).all():
                    raise ValueError('lane_mask_contract_failed')
                samples.append({'video_message_index': index, 'timestamp_ns': frame['timestamp_ns'],
                                'left_shape': list(image.shape), 'quality': frame_quality(image),
                                'pothole_candidates': predictions['pothole'], 'privacy_candidates': predictions['redaction'],
                                'lane_paint_pixel_fraction': float(mask.mean()), 'latency_ms': latencies,
                                'human_reviewed': False})
            integrity, eof = dict(reader.integrity), reader.eof
        after_path, after = check_source(row, guard)
        if after_path != path or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('recording_stat_changed_during_probe')
        if count != row['audited_video_messages'] or len(samples) != len(wanted) or not eof or not integrity['fully_consumed']:
            raise ValueError('audited_full_decode_or_sample_count_mismatch')
        results.append({'source_path': row['source_path'], 'source_sha256': row['sha256'],
                        'original_unchanged': True, 'decoded_frames': count, 'eof': eof, 'integrity': integrity, 'samples': samples})
        print(json.dumps({'completed_files': len(results), 'sampled_frames': sum(len(r['samples']) for r in results)}), flush=True)
    if not results:
        raise ValueError('recordings_required')
    aggregate = summarize(results, plan)
    aggregate['frozen_plan_sha256'] = sha256(plan_path)
    aggregate['elapsed_seconds_including_decode_and_checksums'] = time.perf_counter() - started
    aggregate['runtime'] = runtime_metadata(models)
    guard()
    atomic_json(output, {'plan_sha256': sha256(plan_path), 'results': results, 'aggregate': aggregate})
    atomic_json(plan_path.parent/'aggregate.json', aggregate)
    if public_report:
        atomic_json(public_report, aggregate)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    for name in ('inventory', 'audit', 'permission', 'pothole', 'lane', 'redaction', 'output'):
        plan.add_argument('--'+name, required=True)
    run = sub.add_parser('run')
    run.add_argument('plan'); run.add_argument('--public-report')
    args = parser.parse_args()
    if args.command == 'plan':
        result = create_plan(args.inventory, args.audit, args.permission, {k: getattr(args, k) for k in ('pothole', 'lane', 'redaction')}, args.output)
        print(json.dumps({'planned_files': len(result['sources']), 'planned_samples': sum(len(s['selected_video_message_indices']) for s in result['sources'])}))
    else:
        result = run_plan(args.plan, public_report=args.public_report)
        print(json.dumps({'status': result['status'], 'frames': result['inferred_left_frames']}))


if __name__ == '__main__':
    main()
