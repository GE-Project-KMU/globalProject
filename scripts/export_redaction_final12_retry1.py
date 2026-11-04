"""Retry1 of the same predeclared final12 EMA export; correct TRAIN/VAL membership.

Declaration reads metadata only. Actual recovery/weights loading, synthetic
prefix inference and ONNX export require completed original training and a
separate explicit CPU export approval. This tool never trains or reads images.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.paths import atomic_json, workspace_path, sha256

PARENT = ROOT / 'data/training/redaction-rescue-v8'
OUT = ROOT / 'data/training/redaction-rescue-v8-final12-retry1'
POLICY = ROOT / 'docs/reports/PRIVACY-RESCUE-V8-FINAL12-RETRY1-EXPORT-PLAN.json'
TRAINER = ROOT / 'scripts/train_redaction_rescue_v8.py'
TRAINER_SHA = '7bc192db04ff79ff0cad9c3b440aa8357c99f387eecf4a7093ce3f59d9475f65'
TRAIN_PLAN_SHA = '6bbd59191a2c352a6a2bf7dc0fbd15f25ced0f500be6010fa3011a4e0f94164d'
SUPERVISION_PLAN = ROOT / 'docs/reports/PRIVACY-RESCUE-V8-RETRY1-SUPERVISION-PLAN.json'
COMPLETION = PARENT / 'external-train-supervisor-retry1.json'
BASE = ROOT / 'data/training/redaction-specialists-v4-efficient/best.pt'
BASE_SHA = '6c985f7e3afb0fa061cbb82d7ec7b304eb5f9fbfcd0cf9f775dbab28873c82ad'
DEV = ROOT / 'data/training/research-redaction-cycle2-v7/development.json'
RELEASE = ROOT / 'data/training/research-redaction-cycle2-v7/release.json'
DEV_SHA = '37cdd77565438c1254768347a7d7abe37c686ba20239c3ade2fda23cd98a8d0c'
RELEASE_SHA = '5ae56bd60700ce17a6af8e92edf630a2722ebc6148e177b798afa172cbff50dc'
PREDECLARATION = ROOT / 'docs/reports/PRIVACY-FAMILY-V9-PREDECLARATION.json'
PREDECLARATION_SHA = '2717f0a737aa73d72afca8cc567519d0630c3ae2ebe77b980d608a9018228117'
OLD_OUT = ROOT/'data/training/redaction-rescue-v8-final12'
OLD_POLICY = ROOT/'docs/reports/PRIVACY-RESCUE-V8-FINAL12-EXPORT-PLAN.json'
OLD_POLICY_SHA = 'bca93d5d7163f4746c436ba2da4c277fe8b0356ac7cdfc8d0e4b91d857ad3daf'
OLD_EXPORTER_SHA = 'e8de9f51024417d4b4b36a8791870e144786ce582fea9adca1a493b88107287a'
OLD_WRAPPER_SHA = '3b449b4ccf14814ae9383d9817b9b45d0db61c198aecad0b40bdf5471ca6c0b1'
SOURCE_RECOVERY_SHA = '0d72cb7f8655cef70c2f99f3fab45e884479eab945aada56454bd51bb9697e98'
AUDIT = OLD_OUT/'failure-parent-preservation-audit.json'
AUDIT_SHA = '7a4a7b84d6dafb8d11b775271031c14974c51fecf58caf4930fcfc41b6ca230c'
STOP = ROOT/'.cache/privacy-family-v9-operator/V10-revision-stop.json'
STOP_SHA = 'dfd8b3769f36fe4ea31a53467c160c8078bca985ff14a64dc3a0c2b8992a415e'
BLOCK_NAMES = ('fresh-attempt.json', 'regression-attempt.json', 'fresh-comparison.json',
               'regression-comparison.json', 'parallel-fresh-v8', 'parallel-regression-v8',
               'parallel-fresh-v9', 'parallel-regression-v9', 'parallel-fresh-v10', 'parallel-regression-v10')


def read(path):
    return json.loads(workspace_path(path, exists=True).read_text(encoding='utf-8'))


def ref(path):
    return {'path': workspace_path(path, exists=True).relative_to(ROOT).as_posix(), 'sha256': sha256(path)}


def ensure_preTEST():
    for family in ('v8', 'v9', 'v10'):
        folder = ROOT / f'data/training/privacy-family-{family}-evaluation'
        if any((folder / name).exists() for name in BLOCK_NAMES):
            raise ValueError('family_TEST_already_started_fixed12_export_forbidden')


def alive(value):
    deadline = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
        raise ValueError('original_source_deadline_expired_or_naive')


def pinned_inputs():
    validate_retry_lineage()
    if sha256(PREDECLARATION) != PREDECLARATION_SHA:
        raise ValueError('original_family_predeclaration_must_remain_unchanged')
    if sha256(TRAINER) != TRAINER_SHA or sha256(PARENT / 'plan.json') != TRAIN_PLAN_SHA:
        raise ValueError('original_training_source_or_learning_plan_changed')
    learning = read(PARENT / 'plan.json')
    if sha256(ROOT / 'docs/reports/PRIVACY-RESCUE-V8-PLAN.json') != TRAIN_PLAN_SHA:
        raise ValueError('original_public_learning_plan_changed')
    for path, digest in learning['dependencies_sha256'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError('original_factory_or_data_source_changed')
    if sha256(DEV) != DEV_SHA or sha256(RELEASE) != RELEASE_SHA:
        raise ValueError('original_release_or_development_changed')
    alive(learning['source_expires_at'])
    return learning


def validate_retry_lineage():
    expected = {ROOT/'scripts/export_redaction_final12.py': OLD_EXPORTER_SHA,
                ROOT/'scripts/supervise_final12_cpu_export.py': OLD_WRAPPER_SHA,
                OLD_POLICY: OLD_POLICY_SHA, AUDIT: AUDIT_SHA, STOP: STOP_SHA}
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError('original_failed_export_or_independent_audit_changed')
    failed = read(OLD_OUT/'root-export-execution.json')
    if (failed.get('status') != 'failed' or failed.get('child_exit_code') != 1
            or failed.get('own_child_running_after_cleanup') is not False or failed.get('sources_unchanged') is not True
            or failed.get('export_source_sha256') != OLD_EXPORTER_SHA or failed.get('wrapper_source_sha256') != OLD_WRAPPER_SHA
            or failed.get('export_plan_sha256') != OLD_POLICY_SHA or failed.get('source_recovery_sha256') != SOURCE_RECOVERY_SHA):
        raise ValueError('bound_original_failed_CPU_export_required')
    require_child_gone(failed)
    text = (ROOT/'.cache/cycle2-final12-cpu-export.log').read_text(encoding='utf-8')
    if 'complete_original_TRAIN_VAL_only_provenance_required' not in text:
        raise ValueError('original_membership_gate_failure_log_required')
    return [OLD_POLICY, ROOT/'scripts/export_redaction_final12.py', ROOT/'scripts/supervise_final12_cpu_export.py',
            OLD_OUT/'root-export-execution.json', OLD_OUT/'export-attempt.json',
            ROOT/'.cache/cycle2-final12-cpu-export.log', AUDIT, STOP]


def original_training_membership(rows, provenance, source_plan_sha256, training_source_sha256):
    """Exact frozen trainer predicate and complete source-identity mapping, no pixels."""
    from agent.governance import identities, effective_expiry
    if source_plan_sha256 != TRAIN_PLAN_SHA or training_source_sha256 != TRAINER_SHA:
        raise ValueError('original_trainer_source_plan_binding_required')
    if any(row['split'] not in ('train','validation') for row in rows):
        raise ValueError('original_development_only_metadata_required')
    train = [r for r in rows if r['split']=='train' and r['evaluation_class']=='face']
    validation = [r for r in rows if r['split']=='validation']
    if (len(rows)!=5501 or len(train)!=3000
            or sum(r['evaluation_class']=='face' for r in validation)!=400
            or sum(r['evaluation_class']=='license_plate' for r in validation)!=108
            or sum(r['split']=='train' and r['evaluation_class']=='license_plate' for r in rows)!=1993):
        raise ValueError('exact_original_faceTRAIN3000_allVAL508_required')
    selected = sorted(train,key=lambda r:r['sample_id'])+sorted(validation,key=lambda r:r['sample_id'])
    assets = provenance.get('assets', [])
    if (provenance.get('release_sha256') != RELEASE_SHA or provenance.get('test_split_exported') is not False
            or len(assets)!=3508 or len({a['sample_id'] for a in assets})!=3508
            or len({r['sample_id'] for r in selected})!=3508):
        raise ValueError('complete_original3508_provenance_required')
    by_id = {r['sample_id']:r for r in selected}
    if set(by_id)!={a['sample_id'] for a in assets}:
        raise ValueError('exact_selected_sample_membership_required')
    canonical = []
    for asset in assets:
        row = by_id[asset['sample_id']]
        expected = {'sample_id':row['sample_id'],'source_id':row['source_id'],
                    'observation_id':row.get('observation_id'),'source_path':Path(row['path']).as_posix(),
                    'source_sha256':row['sha256'],'expires_at':effective_expiry(row),'identity_hashes':identities(row)}
        if any(asset.get(key)!=value for key,value in expected.items()):
            raise ValueError('complete_original_source_identity_or_TTL_mismatch')
        canonical.append(expected)
    digest=hashlib.sha256(json.dumps(sorted(canonical,key=lambda r:r['sample_id']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return selected, digest


def membership_audit():
    pinned_inputs()
    if sha256(PARENT/'training-config.json') != TRAIN_PLAN_SHA:
        raise ValueError('actual_original_training_config_binding_required')
    rows, provenance = read(DEV), read(PARENT/'provenance.json')
    selected, membership = original_training_membership(rows,provenance,TRAIN_PLAN_SHA,TRAINER_SHA)
    return {'format':'fixed12-retry1-metadata-membership-audit-v1','passed':True,
            'export_source_sha256':sha256(Path(__file__)),'training_source_sha256':TRAINER_SHA,'source_plan_sha256':TRAIN_PLAN_SHA,
            'development_sha256':DEV_SHA,'parent_provenance_sha256':sha256(PARENT/'provenance.json'),
            'development_rows':len(rows),'selected_training_and_validation_rows':len(selected),'parent_assets':len(provenance['assets']),
            'unused_plate_TRAIN_rows_excluded':1993,'complete_identity_set_sha256':membership,
            'predicate':'(split=train and evaluation_class=face) or split=validation',
            'images_or_recovery_or_models_read':False,'GPU_used':False,'TEST_images':0}


def declare():
    ensure_preTEST(); learning=pinned_inputs(); proof=membership_audit()
    receipt,status,metadata=read(COMPLETION),read(PARENT/'training-status.json'),read(PARENT/'recovery-receipt.json')
    validate_completed(receipt,status,metadata,learning);require_child_gone(receipt)
    if metadata['checkpoint_sha256']!=SOURCE_RECOVERY_SHA:
        raise ValueError('same_fixed_epoch12_recovery_required')
    if POLICY.exists() or OUT.exists():
        raise ValueError('fresh_retry1_policy_namespace_required')
    OUT.mkdir(parents=True)
    proof_path=OUT/'metadata-membership-audit.json';atomic_json(proof_path,proof)
    public_proof=ROOT/'docs/reports/PRIVACY-FINAL12-RETRY1-MEMBERSHIP.json'
    if public_proof.exists():raise ValueError('membership_public_proof_already_exists')
    atomic_json(public_proof,proof)
    parent=read(OLD_POLICY);plan=copy.deepcopy(parent)
    lineage=validate_retry_lineage()
    plan.update(format='privacy-fixed-final12-EMA-export-retry1-policy-v1',
        export_source_sha256=sha256(Path(__file__)),parent_export_policy=ref(OLD_POLICY),
        parent_family_predeclaration=ref(PREDECLARATION),
        failed_export_lineage={p.relative_to(ROOT).as_posix():ref(p) for p in lineage},
        revision_reason='Software membership gate fix only: original trainer used faceTRAIN3000 plus allVAL508, not unused plateTRAIN1993. Same predeclared fixed12 EMA; no training, data, threshold or checkpoint reselection.',
        provenance_membership_audit=ref(proof_path),source_recovery_sha256=SOURCE_RECOVERY_SHA,
        original_trainer_subset={'train_face':3000,'validation_face':400,'validation_plate':108,'total':3508},
        original_actual_optimizer_updates=71969,new_selection_or_training_performed=False,
        retry_after_training_completed=True,automatic_retry_allowed=False)
    plan['source_sha256'].update({p.relative_to(ROOT).as_posix():sha256(p) for p in [Path(__file__),*lineage,PARENT/'training-config.json',proof_path]})
    atomic_json(OUT/'plan.json',plan);atomic_json(POLICY,plan)
    print(json.dumps({'status':'fixed12_retry1_policy_frozen_export_not_started','export_plan_sha256':sha256(POLICY),'export_source_sha256':sha256(Path(__file__))}))



def validate_completed(receipt, status, recovery_metadata, learning):
    if (receipt.get('phase') != 'train' or receipt.get('status') != 'completed'
            or type(receipt.get('child_exit_code')) is not int or receipt['child_exit_code'] != 0
            or receipt.get('own_child_running_after_cleanup') is not False or receipt.get('sources_unchanged') is not True
            or receipt.get('training_source_sha256') != TRAINER_SHA or receipt.get('learning_plan_sha256') != TRAIN_PLAN_SHA
            or type(receipt.get('TEST_images')) is not int or receipt['TEST_images'] != 0):
        raise ValueError('completed_original_training_and_child_cleanup_required')
    if (status.get('status') != 'completed' or type(status.get('epochs')) is not int or status['epochs'] != 12
            or type(status.get('TEST_images')) is not int or status['TEST_images'] != 0 or status.get('family_eligible') is not True
            or status.get('source_expires_at') != learning['source_expires_at']):
        raise ValueError('complete_original_12epochs_TEST0_required')
    if (type(recovery_metadata.get('completed_epoch')) is not int or recovery_metadata['completed_epoch'] != 12
            or recovery_metadata.get('plan_sha256') != TRAIN_PLAN_SHA or recovery_metadata.get('automatic_resume_allowed') is not False):
        raise ValueError('actual_epoch12_recovery_metadata_required')


def require_child_gone(receipt):
    import psutil
    pid, created = receipt.get('own_child_pid'), receipt.get('own_child_creation_time')
    if type(pid) is not int or pid <= 0 or type(created) not in (int, float) or not math.isfinite(created) or created <= 0:
        raise ValueError('original_child_identity_required')
    try:
        process = psutil.Process(pid)
        if process.create_time() == created and process.is_running():
            raise ValueError('original_training_child_still_exists')
    except psutil.NoSuchProcess:
        pass


def state_digest(state, legacy_only=False):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        if legacy_only and not name.startswith(('face.', 'plate.')):
            continue
        if hasattr(value, 'detach'):
            value = value.detach().cpu().contiguous()
        result.update(name.encode()); result.update(str(value.dtype).encode()); result.update(str(tuple(value.shape)).encode())
        result.update((value.numpy() if hasattr(value, 'numpy') else value).tobytes())
    return result.hexdigest()


def validate_recovery(value, expected_keys):
    if (value.get('format') != 'rescue-epoch-recovery-v1' or value.get('plan_sha256') != TRAIN_PLAN_SHA
            or type(value.get('completed_epoch')) is not int or value['completed_epoch'] != 12
            or type(value.get('updates')) is not int or value['updates'] < 1
            or value.get('automatic_resume_allowed') is not False):
        raise ValueError('complete_fixed12_EMA_recovery_required')
    if set(value.get('EMA', {})) != set(expected_keys):
        raise ValueError('complete_EMA_fullstate_required_no_partial_transplant')
    history = value.get('selection_history', [])
    if [r.get('epoch') for r in history] != list(range(13)) or history[-1].get('updates') != value['updates']:
        raise ValueError('original_complete_training_update_history_required')
    return value['EMA']


def derived_provenance(parent, parent_ref, manifest_path):
    if (parent.get('format') != 'training-provenance-v1' or parent.get('release_sha256') != RELEASE_SHA
            or parent.get('test_split_exported') is not False or not parent.get('assets')
            or parent.get('data_revoked') or parent.get('needs_revalidation')):
        raise ValueError('valid_parent_training_provenance_required')
    value = copy.deepcopy(parent)
    for asset in value['assets']:
        alive(asset['expires_at'])
        if asset.get('revoked_at'):
            raise ValueError('revoked_parent_asset_forbidden')
        # Child revocation must not delete any parent assets or labels.
        for key in ('files', 'source_files', 'label_refs', 'delete_refs'):
            asset[key] = []
    value.update(parent_training_provenance=parent_ref, models=[manifest_path],
                 derived_artifact_kind='fixed_epoch12_EMA_CPU_export_without_training',
                 derived_at=datetime.now(timezone.utc).isoformat(), research_only=True,
                 production_approved=False)
    return value


def export(approved=False, expected_plan_sha256=None):
    if approved is not True or not expected_plan_sha256 or sha256(POLICY) != expected_plan_sha256:
        raise ValueError('separate_explicit_CPU_export_approval_and_pinned_policy_required')
    ensure_preTEST(); learning = pinned_inputs(); policy = read(POLICY)
    if read(OUT / 'plan.json') != policy or policy['export_source_sha256'] != sha256(Path(__file__)):
        raise ValueError('frozen_export_source_and_public_private_policy_required')
    for relative, digest in policy['source_sha256'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError('frozen_export_dependency_changed')
    if sha256(SUPERVISION_PLAN) != policy['supervision_plan_sha256']:
        raise ValueError('original_supervision_plan_changed')
    receipt, status, recovery_metadata = read(COMPLETION), read(PARENT / 'training-status.json'), read(PARENT / 'recovery-receipt.json')
    validate_completed(receipt, status, recovery_metadata, learning); require_child_gone(receipt)
    if (receipt.get('supervision_plan_sha256') != sha256(SUPERVISION_PLAN)
            or receipt.get('trainer_report_sha256') != sha256(PARENT / 'training-status.json')):
        raise ValueError('original_training_completion_report_binding_required')
    recovery_path = PARENT / 'epoch-recovery.pt'
    if sha256(recovery_path) != SOURCE_RECOVERY_SHA or recovery_metadata['checkpoint_sha256'] != SOURCE_RECOVERY_SHA or sha256(BASE) != BASE_SHA:
        raise ValueError('actual_recovery_and_frozen_legacy_checkpoint_hash_required')
    artifacts = ['best.pt', 'model.onnx', 'model.json', 'provenance.json', 'export-attempt.json', 'export-receipt.json', 'candidate-freeze.json']
    if any((OUT / name).exists() for name in artifacts):
        raise ValueError('fresh_export_namespace_no_overwrite_or_automatic_retry')
    before = {str(p.relative_to(ROOT)): sha256(p) for p in [COMPLETION, PARENT/'training-status.json', PARENT/'recovery-receipt.json',
              recovery_path, PARENT/'provenance.json', PARENT/'model.json', PARENT/'best.pt', BASE, POLICY]}
    old_before = read(OLD_OUT/'export-attempt.json')['input_sha256']
    for relative,digest in old_before.items():
        if sha256(workspace_path(ROOT/relative,exists=True)) != digest:
            raise ValueError('original_parent_artifact_changed_since_failed_attempt')
    atomic_json(OUT/'export-attempt.json', {'status': 'started', 'epoch': 12, 'EMA': True, 'TEST_images': 0, 'export_plan_sha256': expected_plan_sha256, 'input_sha256': before})
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from agent.redaction_rescue_network import build_redaction_rescue_network
    from agent.redaction_tiled_network import build_redaction_tiled_network
    from agent.governance import begin_training, blocked_identities
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise ValueError('CPU_only_export_CUDA_must_not_initialize')
    model = build_redaction_rescue_network().cpu().eval()
    payload = torch.load(recovery_path, map_location='cpu', weights_only=True)
    state = validate_recovery(payload, model.state_dict())
    if payload['updates'] != 71969 or policy.get('original_actual_optimizer_updates') != 71969:
        raise ValueError('same_original71969_updates_required')
    history = read(PARENT/'history.json')
    if history.get('history') != payload['selection_history'] or history.get('TEST_images') != 0:
        raise ValueError('recovery_must_match_original_completed_training_history')
    if any(not isinstance(v, torch.Tensor) or (v.is_floating_point() and not torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('finite_tensor_EMA_fullstate_required')
    base_state = torch.load(BASE, map_location='cpu', weights_only=True)['model']
    legacy_hash = state_digest(base_state, legacy_only=True)
    if state_digest(state, legacy_only=True) != legacy_hash:
        raise ValueError('all_legacy_parameter_and_BN_bytes_must_remain_unchanged')
    model.load_state_dict(state, strict=True)
    original = read(PARENT/'model.json')
    if (original.get('training_complete') is not True or original.get('optimizer_updates') != payload['updates']
            or original.get('legacy_state_sha256') != legacy_hash or original.get('dataset_release') != 'manifest-sha256:'+RELEASE_SHA
            or original.get('research_only') is not True or original.get('production_approved') is not False
            or original.get('project_acceptance_eligible') is not False or original.get('test_split_used') is not False
            or original.get('data_revoked') is not False or original.get('needs_revalidation') is not False):
        raise ValueError('original_completed_research_manifest_binding_required')
    legacy = build_redaction_tiled_network().cpu().eval(); legacy.load_state_dict(base_state, strict=True)
    # Synthetic bytes only: no source image, NPY or source file pixels loaded.
    pixels = torch.arange(1920*1920, dtype=torch.float32).remainder(251).reshape(1, 1, 1920, 1920).expand(1, 3, 1920, 1920).contiguous()
    with torch.inference_mode():
        output, old = model(pixels), legacy(pixels)
    if list(output.shape) != [1,333300,7] or list(old.shape) != [1,102900,7] or output[:,:102900].contiguous().numpy().tobytes() != old.numpy().tobytes():
        raise ValueError('synthetic_CPU_legacy_raw_prefix_must_be_byte_exact')
    del legacy, output, old
    alive(policy['source_expires_at']); ensure_preTEST()
    parent_provenance = read(PARENT/'provenance.json')
    if sha256(PARENT/'training-config.json') != TRAIN_PLAN_SHA:
        raise ValueError('original_training_config_changed')
    rows,membership = original_training_membership(read(DEV),parent_provenance,TRAIN_PLAN_SHA,TRAINER_SHA)
    audit=read(OUT/'metadata-membership-audit.json')
    if audit['complete_identity_set_sha256'] != membership or audit['parent_provenance_sha256'] != sha256(PARENT/'provenance.json'):
        raise ValueError('predeclared_exact3508_membership_proof_changed')
    blocked = blocked_identities()
    if any(set(a['identity_hashes']) & blocked for a in parent_provenance['assets']):
        raise ValueError('parent_source_identity_revoked')
    manifest_path = OUT/'model.json'
    provenance = derived_provenance(parent_provenance, ref(PARENT/'provenance.json'), manifest_path.relative_to(ROOT).as_posix())
    torch.save({'model': state, 'epoch': 12}, OUT/'best.pt')
    onnx_path = OUT/'model.onnx'
    torch.onnx.export(model, pixels, str(onnx_path), opset_version=17, input_names=['images'], output_names=['detections'], dynamo=False)
    import onnx
    onnx.checker.check_model(str(onnx_path))
    loaded = onnx.load(str(onnx_path), load_external_data=False)
    if [d.dim_value for d in loaded.graph.output[0].type.tensor_type.shape.dim] != [1,333300,7]:
        raise ValueError('fixed333300x7_ONNX_output_required')
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in loaded.graph.initializer):
        raise ValueError('single_owned_ONNX_file_required')
    manifest = copy.deepcopy(original)
    manifest.update(weights=onnx_path.relative_to(ROOT).as_posix(), sha256=sha256(onnx_path),
                    training_provenance=(OUT/'provenance.json').relative_to(ROOT).as_posix(),
                    trained_checkpoint_sha256=sha256(OUT/'best.pt'), confidence=.01, best_epoch=12,
                    checkpoint_selection='preregistered_fixed_final_epoch12_EMA', selected_initial_legacy_fallback=False,
                    rescue_head_selected_after_training=True, final_epoch_EMA=True, source_recovery_sha256=before[str(recovery_path.relative_to(ROOT))],
                    parent_training_manifest=ref(PARENT/'model.json'), export_plan_sha256=expected_plan_sha256,
                    confidence_selection='native_VAL_floor.01_placeholder_before_V9_family_VAL_selection',
                    numerical_verification_pending=True, training_complete=True, test_split_used=False,
                    exported_before_final_test=True, research_only=True, production_approved=False, project_acceptance_eligible=False)
    pp, _ = begin_training(OUT, RELEASE, rows, 'redaction')
    atomic_json(pp, provenance); atomic_json(manifest_path, manifest)
    freeze = {'manifest_sha256': sha256(manifest_path), 'checkpoint_sha256': sha256(OUT/'best.pt'), 'onnx_sha256': sha256(onnx_path),
              'confidence': .01, 'TEST_images': 0, 'training_complete': True, 'epoch': 12, 'fixed_final12_EMA': True,
              'export_plan_sha256': expected_plan_sha256, 'export_source_sha256': sha256(Path(__file__))}
    atomic_json(OUT/'candidate-freeze.json', freeze)
    if any(sha256(ROOT/p) != digest for p,digest in before.items()) or torch.cuda.is_initialized():
        raise ValueError('parent_inputs_changed_or_CUDA_initialized_during_export')
    alive(policy['source_expires_at']); ensure_preTEST()
    result = {'status': 'completed', 'epoch': 12, 'EMA': True, 'TEST_images': 0, 'training_complete': True,
              'manifest_sha256': freeze['manifest_sha256'], 'checkpoint_sha256': freeze['checkpoint_sha256'], 'onnx_sha256': freeze['onnx_sha256'],
              'source_plan_sha256': TRAIN_PLAN_SHA, 'export_plan_sha256': expected_plan_sha256, 'export_source_sha256': sha256(Path(__file__)),
              'source_expires_at': policy['source_expires_at'], 'source_recovery_sha256': before[str(recovery_path.relative_to(ROOT))],
              'source_final12_EMA_state_sha256': state_digest(state),
              'parent_training_provenance_sha256': sha256(PARENT/'provenance.json'), 'parent_training_supervisor_sha256': sha256(COMPLETION),
              'legacy_state_sha256': legacy_hash, 'legacy_raw_prefix_matches_frozen_v4': True,
              'legacy_raw_prefix_comparison_scope': 'synthetic_CPU_Torch_only', 'candidate_freeze_sha256': sha256(OUT/'candidate-freeze.json'),
              'optimizer_updates': payload['updates'], 'state_extracted_without_training': True, 'confidence': .01,
              'original_trainer_subset': {'train_face':3000,'validation_face':400,'validation_plate':108,'total':3508},
              'provenance_membership_sha256':membership,'membership_audit_sha256':sha256(OUT/'metadata-membership-audit.json'),
              'parent_export_policy_sha256':OLD_POLICY_SHA,'same_original_final12_EMA':True,
              'GPU_used': False, 'CUDA_initialized': False, 'source_images_or_existing_NPY_read': False,
              'source_expiries_extended': False, 'original_AP_selection_modified': False, 'strict_numeric_verification_pending': True,
              'parent_inputs_unchanged': True, 'research_only': True, 'production_approved': False}
    atomic_json(OUT/'export-receipt.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['declare-policy','export'])
    parser.add_argument('--cpu-export-approved', action='store_true')
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.action == 'declare-policy':
        declare()
    else:
        export(args.cpu_export_approved, args.expected_plan_sha256)
