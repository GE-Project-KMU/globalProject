"""Build only the separately predeclared US100 blind cohort from verified cache."""
from __future__ import annotations
import argparse,copy,hashlib,json,sys,zipfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,sha256,atomic_json
from scripts import prepare_rdd2022_blind as original
from scripts import prepare_rdd2022_us_offline as diagnostic
from scripts.acquire_rdd2022_blind_api import SELECTION_SHA,PROTOCOL_SHA

RELEASE='data/training/research-pothole-rdd2022-us100-v7'
PLAN='us100-offline-protocol-v2.json'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def select(rows):
    result=[]
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('unique_candidate_ids_required')
    for positive in (True,False):
        available=sorted((r for r in rows if bool(r['objects'])==positive),key=lambda r:r['sample_id'])
        if len(available)<50:raise ValueError('insufficient_balanced_support')
        result.extend(copy.deepcopy(available[:50]))
    return result

def plan():
    folder=original.scoped_output(original.DEST);target=folder/PLAN
    if target.exists():raise ValueError('offline_protocol_already_frozen')
    feasibility=folder/'offline-us-feasibility.json';qc=read(feasibility)
    if qc['status']!='feasible' or not qc['all_original_US_candidates_recovered'] or qc['candidate_image_CRC_SHA_verified']!=240:raise ValueError('complete_original_US_candidates_required')
    if qc['near_QC']['near_pairs'] or qc['history_removed']:raise ValueError('explicit_new_QC_revision_required')
    if sha256(Path(diagnostic.__file__))!=qc['diagnostic_source_sha256']:raise ValueError('diagnostic_source_changed')
    if original.aware(qc['source_expires_at'])<=datetime.now(timezone.utc):raise ValueError('original_source_expired')
    payload={'format':'RDD2022-US100-offline-blind-v2','country':'United_States','samples':100,'positive_images':50,'negative_images':50,
             'selection':'Original fixed-seed 240 US candidates; lexicographically first 50 UUIDs per class after historical and connected-duplicate QC.',
             'same_label_component_rule':'Keep lexicographically smallest original UUID.','mixed_label_component_rule':'Exclude entire component.',
             'current_verified_candidate_near_pairs':0,'nonzero_near_pairs_require_explicit_revision':True,
             'one_frame_per_country_filename20_group':True,'independent_physical_route_verified':False,
             'first_two_country_200_protocol_remains_failed':True,'this_is_not_a_two_country_or_200_sample_cohort':True,
             'additional_network_requests':0,'model_inferences_before_freeze':0,'train_or_selection_use':False,
             'expires_at':qc['source_expires_at'],'original_selection_source_sha256':SELECTION_SHA,'original_protocol_sha256':PROTOCOL_SHA,
             'offline_feasibility_sha256':sha256(feasibility),'offline_diagnostic_source_sha256':qc['diagnostic_source_sha256'],
             'builder_source_sha256':sha256(Path(__file__)),'historical_exclusions_sha256':sha256(folder/'history-exclusions.json')}
    previous=folder/'us100-offline-protocol.json'
    payload.update(pre_build_revision_reason='Materialize validated source bytes before file-backed research release validation; selection, UUIDs, source bytes and deadlines unchanged.',
                   previous_pre_build_protocol_sha256=sha256(previous))
    atomic_json(target,payload)
    print(json.dumps({'status':'US100_offline_protocol_frozen','protocol_sha256':sha256(target),'builder_source_sha256':payload['builder_source_sha256'],'samples':100,'model_inferences':0,'network_requests':0}))

def build():
    import cv2,numpy as np
    from scripts.improve_pothole_data import dhash
    from scripts.audit_rdd2022_blind import near_components
    cv2.setNumThreads(1)
    folder=original.scoped_output(original.DEST);payload=read(folder/PLAN)
    if payload['builder_source_sha256']!=sha256(Path(__file__)):raise ValueError('frozen_builder_source_changed')
    if payload['offline_diagnostic_source_sha256']!=sha256(Path(diagnostic.__file__)):raise ValueError('frozen_diagnostic_source_changed')
    if sha256(Path(original.__file__))!=SELECTION_SHA or sha256(folder/'selection-protocol.json')!=PROTOCOL_SHA:raise ValueError('original_source_or_protocol_changed')
    if payload['offline_feasibility_sha256']!=sha256(folder/'offline-us-feasibility.json') or payload['historical_exclusions_sha256']!=sha256(folder/'history-exclusions.json'):raise ValueError('frozen_private_evidence_changed')
    expiry=original.aware(payload['expires_at'])
    if expiry<=datetime.now(timezone.utc):raise ValueError('original_source_expired')
    release=workspace_path(RELEASE)
    if release.exists():raise ValueError('new_US100_release_directory_required')
    _,_,_,current_history=original.history_snapshot(release)
    if current_history!=read(folder/'history-exclusions.json')['releases']:raise ValueError('historical_exclusions_changed')
    qc=read(folder/'offline-us-feasibility.json');rows=select(qc['candidate_identity_rows']);buffers={}
    source=read(folder/'source.json')
    if original.aware(source['expires_at'])!=expiry:raise ValueError('original_deadline_changed')
    with diagnostic.OfflineRange(folder=folder) as remote:
        with zipfile.ZipFile(remote) as outer:
            country,binding=original.open_country(remote,outer,'United_States')
            with country as archive:
                for row in rows:
                    original.safe_member(row['original_member']);info=archive.getinfo(row['original_member'])
                    if row['country']!='United_States' or not 0<info.file_size<=24*1024*1024:raise ValueError('frozen_US_source_required')
                    raw=archive.read(info)
                    if hashlib.sha256(raw).hexdigest()!=row['sha256'] or info.CRC!=row['publisher_inner_image_crc32']:raise ValueError('original_image_CRC_SHA_changed')
                    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                    if image is None or image.shape[:2]!=(row['image']['height'],row['image']['width']):raise ValueError('source_geometry_changed')
                    if f'{dhash(image):016x}'!=row['perceptual_dhash']:raise ValueError('original_visual_identity_changed')
                    row['_flip_dhash']=dhash(cv2.flip(image,1));buffers[row['sample_id']]=raw
        blocks=len(remote.verified_cached_blocks)
    pairs,components,mixed=near_components(rows)
    if pairs or len({r['sha256'] for r in rows})!=100 or len({r['leakage_group_id'] for r in rows})!=100:raise ValueError('selected_US100_duplicate_QC_failed')
    evidence=folder/'license-evidence.json'
    for row in rows:
        row.pop('_flip_dhash');row.update(path=(release/'source'/(row['sample_id']+'.jpg')).relative_to(ROOT).as_posix(),
            annotation_version='RDD2022-official-VOC-literal-D40-US100-offline-v2',privacy_status='restricted_research',
            qc_status='publisher_annotations_machine_checked',release_scope='offline_research',human_privacy_reviewed=False,
            not_for_deployment=True,project_acceptance_eligible=False,independent_route_verified=False,detection_labels_verified=True,
            allowed_use='CC BY 4.0 private offline evaluation only; no human privacy/deployment approval',expires_at=expiry.isoformat(),
            source_expires_at=expiry.isoformat(),source_url=original.SOURCE,license_evidence_path=evidence.relative_to(ROOT).as_posix(),license_evidence_sha256=sha256(evidence))
    if expiry<=datetime.now(timezone.utc):raise ValueError('source_expired_before_release')
    materialize_and_validate(rows,buffers,release,expiry)
    atomic_json(release/'release.json',rows)
    summary={'format':'RDD2022-US100-offline-blind-v2','status':'frozen','samples':100,'country':'United_States','positive_images':50,'negative_images':50,
             'network_requests':0,'model_inferences':0,'never_training_or_model_selection':True,'original_two_country_200_target_met':False,
             'original_US_candidates_recovered':240,'selected_individual_inner_CRC_and_SHA_verified':100,'verified_cached_blocks_read':blocks,
             'unique_image_SHA_count':100,'unique_filename20_groups':100,'within_cohort_near_pairs':len(pairs),'whole_outer_MD5_verified':False,
             'whole_country_CRC_verified':False,'physical_route_independence_verified':False,'human_privacy_reviewed':False,'not_for_deployment':True,
             'source_expires_at':expiry.isoformat(),'protocol_sha256':sha256(folder/PLAN),'builder_source_sha256':sha256(Path(__file__)),
             'diagnostic_source_sha256':payload['offline_diagnostic_source_sha256'],'release_sha256':sha256(release/'release.json')}
    atomic_json(release/'split-evidence.json',summary);print(json.dumps(summary))

def materialize_and_validate(rows,buffers,release,expiry):
    """No manifest is published until file-backed validation sees actual bytes."""
    from agent.dataset import validate_research_release
    if release.exists():raise ValueError('new_US100_release_directory_required')
    if expiry<=datetime.now(timezone.utc):raise ValueError('source_expired_before_materialization')
    (release/'source').mkdir(parents=True,exist_ok=False)
    try:
        for row in rows:
            path=workspace_path(row['path'])
            if not path.is_relative_to(release/'source') or path.parent!=release/'source':raise ValueError('materialized_source_scope_required')
            raw=buffers[row['sample_id']]
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('materialized_source_SHA_mismatch')
            with path.open('xb') as stream:stream.write(raw)
        if expiry<=datetime.now(timezone.utc):raise ValueError('source_expired_during_materialization')
        check=validate_research_release(rows)
        if not check['valid']:raise ValueError('US100_research_release_validation_failed')
    except BaseException as error:
        atomic_json(release/'incomplete.json',{'status':'incomplete','error_type':type(error).__name__,
                    'source_expires_at':expiry.isoformat(),'expires_at':expiry.isoformat(),'release_published':False,
                    'original_source_deleted':False,'model_inferences':0,'network_requests':0})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','build']);args=parser.parse_args()
    try:(plan if args.action=='plan' else build)()
    except BaseException as error:
        print(json.dumps({'status':'offline_US100_stopped','error_type':type(error).__name__}),file=sys.stderr);sys.exit(1)
