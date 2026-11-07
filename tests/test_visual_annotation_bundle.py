from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import uuid

import pytest

from scripts import validate_visual_annotation_bundle as v

NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)
END = '2026-11-14T14:59:00Z'


@pytest.fixture
def draft(tmp_path):
    # conftest confines tmp_path to ROOT/.cache; fixtures contain no real pixels.
    permission = tmp_path / 'permission.txt'
    permission.write_text('Synthetic explicit approval: source + derived until deadline.', encoding='utf-8')
    frame = tmp_path / 'frame.bin'
    frame.write_bytes(b'synthetic-file-not-an-image')
    group = str(uuid.uuid4())
    return {'format':'visual-ai-annotation-bundle-v1', 'purpose':'human_review_only',
            'expires_at':END, 'source_expires_at':END,
            'permission_path':str(permission), 'permission_sha256':hashlib.sha256(permission.read_bytes()).hexdigest(),
            'leakage_group_id':group, 'training_export_enabled':False, 'evaluation_ground_truth_eligible':False,
            'samples':[{'sample_id':str(uuid.uuid4()), 'dataset_schema_version':'1.0',
                        'image':{'width':4,'height':3,'coord_space':'left_original','path':str(frame),
                                 'sha256':hashlib.sha256(frame.read_bytes()).hexdigest()},
                        'annotation_status':'draft','annotator_id':'AIassistant','reviewer_id':None,
                        'human_reviewed':False,'detection_eligible':False,'labels_complete':False,'negative_eligible':False,
                        'leakage_group_id':group, 'expires_at':END,'source_expires_at':END,
                        'potholes':[]}]}


def run(draft, **kw):
    return v.validate_bundle(draft, now=NOW, **kw)


def obj(**extra):
    return {'annotation_id':str(uuid.uuid4()),'bbox_xywh_px':[1,1,2,1],'reason':'synthetic visual basis',**extra}


def test_valid_draft_is_not_approval_and_hash_opt_in(draft, monkeypatch):
    original = v.digest
    calls = []
    def spy(path):
        calls.append(path.name)
        return original(path)
    monkeypatch.setattr(v, 'digest', spy)
    answer = run(draft)
    assert answer['valid'] and answer['human_reviewed'] is False
    assert answer['training_export_enabled'] is False and answer['evaluation_ground_truth_eligible'] is False
    assert calls == ['permission.txt']
    assert run(draft, verify_frame_hashes=True)['valid']
    assert calls[-2:] == ['permission.txt','frame.bin']


@pytest.mark.parametrize('coordinates', [[-1,0,1,1],[0,0,5,1],[0,0,1,0],[True,0,1,1],[float('nan'),0,1,1],[0,0,float('inf'),1]])
def test_real_bbox_bounds(draft, coordinates):
    draft['samples'][0]['privacy_boxes'] = [obj(**{'class':'face','certainty':'confirmed','bbox_xywh_px':coordinates})]
    assert not run(draft)['valid']


def test_uncertain_cannot_be_negative_or_implicitly_approved(draft):
    s = draft['samples'][0]
    s['uncertain_regions'] = [obj(target='pothole')]
    assert run(draft)['errors'][0]['code'] == 'uncertain_requires_held'
    s['annotation_status'] = 'held'
    assert run(draft)['valid']
    s['negative_eligible'] = True
    assert not run(draft)['valid']


@pytest.mark.parametrize('key,value', [('annotation_status','qc_passed'),('qc_status','AIqc_passed'),('human_reviewed',True),
                                      ('labels_complete',True),('training_export',True),('evaluation_ground_truth_eligible',True),
                                      ('reviewer_id','human'),('split','test')])
def test_ai_cannot_grant_review_or_evaluation(draft,key,value):
    draft['samples'][0][key] = value
    assert not run(draft)['valid']


def test_privacy_uncertain_requires_held_and_group_is_same_trip(draft):
    s = draft['samples'][0]
    s['privacy_boxes'] = [obj(**{'class':'license_plate','certainty':'uncertain'})]
    assert not run(draft)['valid']
    s['annotation_status']='held'
    assert run(draft)['valid']
    s['leakage_group_id']=str(uuid.uuid4())
    assert not run(draft)['valid']


def test_stereo_pair_must_share_group_and_resolution(draft,tmp_path):
    left = draft['samples'][0]
    left.update(stereo_pair_id=str(uuid.uuid4()),view='left',trip_id='same-trip')
    right = deepcopy(left)
    right.update(sample_id=str(uuid.uuid4()),view='right')
    p = tmp_path / 'right.bin'
    p.write_bytes(b'different-synthetic-file')
    right['image'].update(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    draft['samples'].append(right)
    assert run(draft)['valid']
    right['image']['height']=2
    assert not run(draft)['valid']


def masks():
    # Column-major foreground pixel (row=1,col=0), not row-major (0,1).
    one = {'size':[3,4], 'counts':[1,1,10]}
    return {k:deepcopy(one) for k in ('outline','paint','valid')} | {'gap':{'size':[3,4],'counts':[12]}}


def test_original_resolution_rle_and_valid_subset(draft):
    s = draft['samples'][0]
    s['lane_masks']=masks()
    assert run(draft)['valid']
    from agent.dataset import decode_rle
    assert decode_rle(s['lane_masks']['paint'])[1,0] == 1
    assert decode_rle(s['lane_masks']['paint'])[0,1] == 0
    s['lane_masks']['valid']['counts']=[2,1,9]
    assert run(draft)['errors'][0]['code']=='valid_mask_outside_manual_outline'


@pytest.mark.parametrize('change', ['size','bool_count','wrong_sum','missing','paint_outside'])
def test_bad_rle_rejected_before_large_decode(draft,change):
    m = masks()
    draft['samples'][0]['lane_masks']=m
    if change=='size': m['paint']['size']=[1,12]
    elif change=='bool_count': m['paint']['counts']=[True,1,10]
    elif change=='wrong_sum': m['paint']['counts']=[11]
    elif change=='missing': del m['gap']
    else: m['paint']['counts']=[2,1,9]
    assert not run(draft)['valid']


@pytest.mark.parametrize('points', [[[0,0],[3,2],[0,2],[3,0]], [[0,0],[1,0],[2,0]],
                                    [[0,0],[4.1,0],[0,1]],[[True,0],[2,0],[0,2]],[[0,0],[2,0],[0,2],[0,0]]])
def test_polygon_geometry(draft,points):
    draft['samples'][0]['lane_polygons']=[{'kind':'lane_outline','points':points,'reason':'synthetic'}]
    assert not run(draft)['valid']


def test_valid_polygon_and_hard_negative(draft):
    s = draft['samples'][0]
    s['lane_polygons']=[{'kind':'lane_outline','points':[[0,0],[4,0],[4,3],[0,3]],'reason':'synthetic'}]
    s['hard_negatives']=[obj(type='manhole')]
    assert run(draft)['valid']


def test_permission_expiry_and_frame_binding(draft):
    draft['permission_sha256']='0'*64
    assert not run(draft)['valid']
    draft['permission_sha256']=v.digest(v.scoped_file(draft['permission_path']))
    draft['source_expires_at']='2026-11-14T15:00:00Z'
    assert not run(draft)['valid']
    draft['source_expires_at']=END
    assert not v.validate_bundle(draft, now=v.DEADLINE)['valid']
    draft['samples'][0]['image']['sha256']='a'*64
    assert run(draft)['valid']  # metadata-only explicitly makes no byte-hash claim
    assert not run(draft,verify_frame_hashes=True)['valid']


def test_outside_rejected_before_resolve_or_read(draft,monkeypatch):
    original = v.Path.resolve
    def trap(path,*a,**kw):
        assert 'outside-workspace-fixture' not in str(path)
        return original(path,*a,**kw)
    monkeypatch.setattr(v.Path,'resolve',trap)
    draft['samples'][0]['image']['path']=str(v.ROOT.parent / 'outside-workspace-fixture' / 'x.bin')
    answer = run(draft)
    assert not answer['valid']
    assert 'outside-workspace-fixture' not in json.dumps(answer)


def test_frame_tamper_duplicate_and_annotation_compatibility(draft):
    s = draft['samples'][0]
    s['potholes']=[obj(physical_object_id=str(uuid.uuid4()),certainty='confirmed')]
    assert run(draft)['valid']
    s['potholes'][0]['physical_object_id']='not-a-uuid'
    assert not run(draft)['valid']
    s['potholes']=[]
    duplicate=deepcopy(s)
    duplicate['sample_id']=str(uuid.uuid4())
    draft['samples'].append(duplicate)
    assert not run(draft)['valid']


@pytest.mark.parametrize('where,key', [('bundle','human_reviewed'),('bundle','AIqc_passed'),('sample','AIqc_passed'),('sample','qc_passed')])
def test_explicit_ai_approval_flags_are_rejected(draft,where,key):
    target = draft if where == 'bundle' else draft['samples'][0]
    target[key] = True
    assert not run(draft)['valid']


def test_direct_cli_with_clean_pythonpath_and_unrelated_workspace_cwd(tmp_path):
    import os
    import subprocess
    import sys
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['TEMP'] = str(tmp_path)
    env['TMP'] = str(tmp_path)
    env['TMPDIR'] = str(tmp_path)
    env['PYTHONPYCACHEPREFIX'] = str(tmp_path / 'pycache')
    result = subprocess.run(
        [sys.executable, str(v.ROOT / 'scripts' / 'validate_visual_annotation_bundle.py'), '--help'],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '--verify-frame-hashes' in result.stdout
    assert 'ModuleNotFoundError' not in result.stderr
