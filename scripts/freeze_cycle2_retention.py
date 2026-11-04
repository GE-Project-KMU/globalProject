"""Separate cycle2 retention preparation. Default control audit never scans assets.

Only a pinned complete writer-stop handoff permits a manual freeze/register +
dry-run. There is no purge/apply action and no automatic schedule.
"""
from __future__ import annotations
import argparse,json,os,re,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_retention as r
from scripts import freeze_public_cycle_retention as old
from scripts import cycle2_research_writer_guard as writer_guard

# Frozen source dependencies: exact known nonpersonal controls must stay at original paths.
PROTECTED_CONTROLS={'.cache/cycle2-rescue-v8-preflight.log': '8fc5f869ed5d32d14ab42f0ad0567ec7865382464bb0aacc4c73593b915fa99c', '.cache/cycle2-rescue-v8-preflight-observer.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'data/training/redaction-rescue-v8-env-repair/before.json': 'caa25cc806aa187d49fc97a156dba54142fd61b77e0b9ffeeec27a80cc7a0f7e', 'data/training/redaction-rescue-v8-env-repair/after.json': '01a3232a942f93cf5b1f845f673c52c2e994c22cc8c7806458e65b78e95cde1d', 'data/training/redaction-rescue-v8-env-repair/import-success.json': 'be1fe14662bc1e4a2119010e1a3c49e592c595ff15fa8b0e983cd6273a4a5879', 'data/training/redaction-rescue-v8-env-repair/requirements-snapshot.txt': '850b0fef35c2cca7066f3dbb06ec6a2dcc83889b68e3713113904b3512eccbc6', 'data/training/redaction-rescue-v8-env-repair/install.log': 'f30363bc13659f869f59d25e6c3190e475f718297ff40cf3603fc24e5dd59cfd', 'data/training/redaction-rescue-v8-env-repair/install-compatible.log': 'a6b154a7881bb39af4834474715be31180f68de4981a3decc3646ed79ced4b0b', 'data/training/redaction-rescue-v8-env-repair/publisher-wheel-metadata.json': '736aadfb3f92b08b00175d25c2bd5a13a0418ecc84b94b7b4d2cac0e3a393c66'}

def protected_controls():
    for name,pin in PROTECTED_CONTROLS.items():
        path=r.project_path(name,file=True)
        if r.file_snapshot(path)[0]!=pin:raise ValueError('pinned_nonpersonal_control_changed')
    return len(PROTECTED_CONTROLS)

def preparation_guard():
    writer_guard.require_stopped(r.ROOT)
    protected_controls()

STATE='data/research-retention/'
PINS={**old.PINS,
 'public-cycle-inventory.json':'7aeec19e4445366124c9faa50cc48b164eaf9fe0f4fa758d91fa362d68f41599',
 'public-cycle-frozen-plan.json':'65e8f391acbb4a8d086d8fbbd0a13154851c248cceb286eb1109369970ab3d44'}
LANE='data/public/a2d2-cycle2-v5/source.json'
PRIVACY='data/training/research-redaction-cycle2-v7/release.json'
BLIND='data/research/rdd2022-blind-v7/source.json'
TINY='data/training/pothole-tiny-v7/release.json'
PROTOTYPE='data/training/redaction-rescue-cpu-prototype-v8'
PROTOTYPE_CONTEXT=PROTOTYPE+'/synthetic-artifact-context.json'
PROTOTYPE_CAP='2026-11-04T17:37:03.942455Z'
PROTOTYPE_NPY={'synthetic-input.npy','baseline-torch.npy','candidate-torch.npy'}
DOMAIN_V11='data/research/rdd2022-domain-v11'
DOMAIN_V11_ARCHIVES={'RDD2022_released_through_CRDDC2022.zip','RDD2022_released_through_CRDDC2022.zip.download'}
PROOF_PINS={LANE:'af18e33a8cc42cebc2618776594a50fcc5d8886b3c0a4edab7f9b6b9ff668006',
 PRIVACY:'5ae56bd60700ce17a6af8e92edf630a2722ebc6148e177b798afa172cbff50dc',
 BLIND:'38a5002db939310c8d5a6a6b0977a917be6801ad148403b7647e30c054c78d6b',
 TINY:'43bcbd0c5160276dc8de9507e40db0a8776ff35e74faf971dcdf69fdddd6d52f',
 PROTOTYPE_CONTEXT:'dd7cad3ddb375fee17be393b2c6d0452d5ff603203413c4753cdec8ac9dbc1f4'}
LANE_CAP='2026-11-10T17:07:26.069653Z';PRIVACY_CAP='2026-11-10T16:55:19.285523Z';BLIND_CAP='2026-11-10T17:07:24.286159Z'
POLICIES={
 'data/research/rdd2022-blind-v7':('blind_source',[BLIND],BLIND_CAP),
 DOMAIN_V11:('domain_v11_source',[BLIND,DOMAIN_V11+'/transfer-plan.json'],BLIND_CAP),
 'data/training/research-pothole-rdd2022-blind-v7':('blind_release',[BLIND],BLIND_CAP),
 'data/training/research-pothole-rdd2022-us100-v7':('blind_release',[BLIND,'data/training/research-pothole-rdd2022-us100-v7/split-evidence.json'],BLIND_CAP),
 'data/training/pothole-validation-diagnostic-v7':('pothole_diagnostic',['data/training/pothole-mwpd-v6/release.json','data/training/pothole-mwpd-v6/provenance.json'],'2026-11-10T17:14:46.862846Z'),
 'data/training/pothole-tiny-v7':('pothole_model',[TINY,'data/training/pothole-tiny-v7/provenance.json'],'2026-11-10T17:14:46.862846Z'),
 'data/training/pothole-train-hardness-v8':('pothole_diagnostic',['data/training/pothole-mwpd-v6/release.json','data/training/pothole-mwpd-v6/provenance.json','data/training/pothole-train-hardness-v8/plan.json'],'2026-11-10T17:14:46.862846Z'),
 'data/training/pothole-localization-v8':('pothole_model',['data/training/pothole-mwpd-v6/release.json','data/training/pothole-mwpd-v6/provenance.json','data/training/pothole-localization-v8/plan.json'],'2026-11-10T17:14:46.862846Z'),
 'data/training/pothole-small-v9':('pothole_model',['data/training/pothole-mwpd-v6/release.json','data/training/pothole-mwpd-v6/provenance.json','data/training/pothole-small-v9/plan.json'],'2026-11-10T17:14:46.862846Z'),
 'data/training/pothole-small-v9-retry1':('pothole_model',['data/training/pothole-mwpd-v6/release.json','data/training/pothole-mwpd-v6/provenance.json','data/training/pothole-small-v9-retry1/plan.json'],'2026-11-10T17:14:46.862846Z'),
 'data/training/pothole-family-v9':('family_evaluation',[BLIND,'data/training/pothole-mwpd-v6/release.json','data/training/pothole-mwpd-v6/provenance.json'],BLIND_CAP),
 'data/training/pothole-resolution-v10':('family_evaluation',['data/training/pothole-mwpd-v6/release.json','data/training/pothole-mwpd-v6/provenance.json'],'2026-11-10T17:14:46.862846Z'),
 'data/training/pothole-family-v10':('family_evaluation',[BLIND,'data/training/pothole-mwpd-v6/release.json','data/training/pothole-mwpd-v6/provenance.json'],BLIND_CAP),
 PROTOTYPE:('synthetic_prototype',[PROTOTYPE_CONTEXT],PROTOTYPE_CAP),
 'data/training/redaction-rescue-v8':('privacy_model',[PRIVACY,'data/training/redaction-rescue-v8/plan.json'],PRIVACY_CAP),
 'data/training/redaction-rescue-v8-final12':('privacy_model',[PRIVACY,'data/training/redaction-rescue-v8-final12/plan.json'],PRIVACY_CAP),
 'data/training/privacy-family-v9-evaluation':('family_evaluation',[PRIVACY,'data/training/privacy-family-v9-evaluation/protocol.json'],PRIVACY_CAP),
 'data/training/privacy-family-v8-evaluation':('family_evaluation',[PRIVACY,'data/training/privacy-family-v8-evaluation/protocol.json'],PRIVACY_CAP),
 'data/public/a2d2-cycle2-v5':('lane_source',[LANE],LANE_CAP),
 'data/training/research-redaction-cycle2-v7':('privacy_source',[PRIVACY],PRIVACY_CAP),
 'data/training/lane-cycle2-v5':('lane_model',[LANE,'data/training/lane-cycle2-v5/provenance.json'],LANE_CAP),
 'data/training/lane-cycle2-v5-retry1':('lane_model',[LANE],LANE_CAP),
 'data/training/lane-cycle2-retry1-evaluation':('family_evaluation',[LANE],LANE_CAP),
 'data/training/redaction-p2-cycle2-v7':('privacy_model',[PRIVACY,'data/training/redaction-p2-cycle2-v7/provenance.json'],PRIVACY_CAP),
 'data/training/public-nextcycle':('mixed',[LANE,PRIVACY],PRIVACY_CAP),
 'data/training/cycle2-evaluation':('mixed',[LANE,PRIVACY,BLIND],PRIVACY_CAP),
 'data/training/cycle2-logs':('mixed',[LANE,PRIVACY,BLIND],PRIVACY_CAP),
 'data/gpu-model-sets/io-overlap-v1':('performance',[old.PERF_NEW+'/plan.json'],PRIVACY_CAP),
 'data/gpu-model-sets/cycle2-final-v7':('performance',[LANE,PRIVACY,BLIND,'data/gpu-model-sets/cycle2-final-v7/plan.json'],PRIVACY_CAP)}
OPTIONAL_ROOTS={'data/training/cycle2-logs','data/training/research-pothole-rdd2022-blind-v7',
                'data/training/pothole-small-v9','data/training/redaction-rescue-v8',
                'data/training/lane-cycle2-v5-retry1','data/training/pothole-family-v9',
                'data/training/pothole-family-v10','data/training/privacy-family-v8-evaluation',
                'data/training/lane-cycle2-retry1-evaluation','data/training/redaction-rescue-v8-final12','data/training/privacy-family-v9-evaluation'}
CYCLE='public-cycle2-v7'


def audit_immutable():
    ids=set();counts=Counter()
    for name,pin in PINS.items():
        path=r.project_path(STATE+name,file=True)
        if r.file_snapshot(path)[0]!=pin:raise ValueError('historical_control_snapshot_changed')
        value=r.read(path)
        if name.endswith('inventory.json'):
            counts['inventories']+=1
            active={key for key,item in value['entries'].items() if item['status']=='registered'}
            if ids&active:raise ValueError('historical_inventory_duplicate_identifiers')
            ids.update(active)
            for item in value.get('control_plans',{}).values():
                if item.get('status')!='registered':continue
                name=item['path'].removeprefix(STATE)
                if name not in PINS or item['sha256']!=PINS[name]:raise ValueError('historical_control_plan_unpinned')
        else:counts['plans']+=1
    return ids,{'historical_inventories_verified':counts['inventories'],'historical_plans_verified':counts['plans'],
        'historical_registered_files':len(ids),'historical_assets_scanned':False}


def stopped_receipt(path,pin):
    path=r.state_path(path)
    if r.file_snapshot(path)[0]!=r.valid_hash(pin):raise ValueError('writer_receipt_digest_mismatch')
    value=r.read(path)
    if set(value)!={'cycle','writers_stopped','confirmed_at','roots'} or value['cycle']!=CYCLE or value['writers_stopped'] is not True or value['roots']!=sorted(POLICIES):raise ValueError('explicit_complete_writer_stop_required')
    if r.clock(value['confirmed_at'])>r.clock():raise ValueError('future_writer_stop_refused')
    return path


def category(path,tag):
    if tag=='domain_v11_source' and path.suffix.lower() in r.KINDS['source_archive']:
        return 'source_archive' if path.parent==r.ROOT/DOMAIN_V11 and path.name in DOMAIN_V11_ARCHIVES else None
    if tag=='synthetic_prototype' and path.suffix.lower() in ('.npy','.npz'):
        return 'per_image_prediction' if path.parent==r.ROOT/PROTOTYPE and path.name in PROTOTYPE_NPY else None
    if path.suffix.lower()=='.bin':return 'source_range' if path.parent==r.ROOT/r.RANGE_ROOT else None
    value=old.kind(path,tag)
    if value=='derived_image' and tag in ('blind_source','blind_release','lane_source','privacy_source'):return 'source_image'
    return value


def minimal_control_provenance(path,formats=None):
    if path.suffix.lower()!='.json':return False
    value=r.read(path)
    allowed=formats if formats is not None else {'privacy-family-v9-winner-freeze','privacy-family-v9-predeclared-protocol-v1','pothole-family-v9-validation-freeze-v1','pothole-family-one-shot-comparison-v1',
        'pothole-family-v10-CPU-validation-freeze-v1','pothole-family-v10-one-shot-comparison-v1',
        'lane-cycle2-retry1-completed-CPU-artifact-freeze-v1','lane-cycle2-retry1-one-shot-comparison-v1'}
    if not isinstance(value,dict) or value.get('format') not in allowed:return False
    def individual(item):
        if isinstance(item,dict):
            for key,child in item.items():
                if key in r.PRIVATE_FIELDS-{'path'} or key in {'truth','predictions','detections'}:return True
                if key=='path' and (not isinstance(child,str) or Path(child).suffix.lower() not in {'.json','.py','.pt','.pth','.onnx'}):return True
                if individual(child):return True
        elif isinstance(item,list):return any(individual(child) for child in item)
        elif isinstance(item,str) and Path(item).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp','.npy','.npz'}:return True
        return False
    return not individual(value)


OPERATOR_PHASES={'CPU-fixed12-export','fixed12-VAL508','idle-observer','baseline-VAL508','rescue-VAL508-import','winner-freeze','fresh','regression'}|{'strict-'+role+'-'+label for role in ('baseline','winner') for label in ('face','license_plate')}

def valid_log_source(name):
    if re.fullmatch(r'\.cache/cycle2[a-zA-Z0-9_.-]*\.log',name):return True
    path=Path(name)
    if path.as_posix()!=name:return False
    if path.parent.as_posix() not in {'.cache/privacy-family-v8-operator','.cache/privacy-family-v8-operator-retry1','.cache/privacy-family-v9-operator'}:return False
    return path.name in {phase+suffix for phase in OPERATOR_PHASES for suffix in ('.log','-observer.log')}

def prepare_log_migration(receipt_path,receipt_sha256,manifest_path,manifest_sha256,output_path):
    """Exact operator-supplied cache logs only; write a reversible move PLAN, not a move."""
    preparation_guard()
    stopped_receipt(receipt_path,receipt_sha256);audit_immutable()
    manifest=r.state_path(manifest_path)
    if r.file_snapshot(manifest)[0]!=r.valid_hash(manifest_sha256):raise ValueError('log_manifest_digest_mismatch')
    value=r.read(manifest)
    if set(value)!={'format','logs'} or value['format']!='cycle2-exact-log-list-v1' or not isinstance(value['logs'],list) or not 1<=len(value['logs'])<=100:raise ValueError('exact_log_list_required')
    output=r.state_path(output_path)
    if output.exists():raise ValueError('new_log_migration_plan_required')
    entries=[];snapshots={};seen=set();aggregate=0
    for item in value['logs']:
        if set(item)!={'source','sha256'} or not valid_log_source(item['source']):raise ValueError('exact_cycle2_cache_log_required')
        if item['source'] in seen:raise ValueError('duplicate_log_manifest_entry')
        seen.add(item['source']);source=r.project_path(item['source'],file=True)
        snapshot=r.file_snapshot(source)
        if snapshot[0]!=r.valid_hash(item['sha256']):raise ValueError('log_source_digest_mismatch')
        snapshots[item['source']]=snapshot
        if item['source'] in PROTECTED_CONTROLS:
            if snapshot[0]!=PROTECTED_CONTROLS[item['source']]:raise ValueError('pinned_nonpersonal_control_changed')
            aggregate+=1;continue
        try:r.eligible(source,'identifiable_manifest')
        except r.PreservedResearchFile:aggregate+=1;continue
        prefix='operator-v9--' if '/privacy-family-v9-operator/' in item['source'] else 'operator-retry1--' if '/privacy-family-v8-operator-retry1/' in item['source'] else 'operator-original--' if '/privacy-family-v8-operator/' in item['source'] else ''
        destination='data/training/cycle2-logs/'+prefix+source.name
        if r.project_path(destination).exists():raise ValueError('log_migration_destination_exists')
        entries.append({'from_log':item['source'],'to_log':destination,'sha256':snapshot[0]})
    for name,snapshot in snapshots.items():
        if r.file_snapshot(r.project_path(name,file=True))!=snapshot:raise ValueError('log_changed_during_preparation')
    preparation_guard()
    stopped_receipt(receipt_path,receipt_sha256);audit_immutable()
    r.write(output,{'format':'cycle2-reversible-log-move-plan-v1','writer_receipt_sha256':receipt_sha256,
        'exact_log_manifest_sha256':manifest_sha256,'moves':entries,'aggregate_logs_preserved':aggregate,
        'moves_executed':False,'source_files_deleted':0,'requires_manual_scoped_move_and_result_receipt':True})
    preparation_guard()
    return {'planned_private_log_moves':len(entries),'aggregate_logs_preserved':aggregate,'moves_executed':False,'source_files_deleted':0}


def freeze(receipt_path,receipt_sha256):
    # Handoff first: no source proofs, file lists or asset hashes while active.
    preparation_guard()
    receipt=stopped_receipt(receipt_path,receipt_sha256)
    plan=r.state_path(STATE+'cycle2-frozen-plan.json');inventory=r.state_path(STATE+'cycle2-inventory.json');summary=r.state_path(STATE+'cycle2-summary.json')
    if any(os.path.lexists(path) for path in (plan,inventory,summary)):raise ValueError('new_cycle2_outputs_required')
    with r.inventory_lock(summary):
        previous,history=audit_immutable();proofs={};dates={};snapshots={};filesets={};entries=[];preserved=Counter();unknown=Counter()
        def proof(name):
            if name not in proofs:
                path=r.project_path(name,file=True);digest,identity=r.file_snapshot(path)
                if name in PROOF_PINS and digest!=PROOF_PINS[name]:raise ValueError('original_source_proof_changed')
                values=r.evidence_expiries(r.read(path))
                if not values:raise ValueError('each_proof_requires_fixed_original_expiry')
                proofs[name]={'path':name,'sha256':digest};dates[name]=values;snapshots[name]=(digest,identity)
            return dates[name]
        for root,(tag,evidence,ceiling) in POLICIES.items():
            if root in OPTIONAL_ROOTS and not r.project_path(root).exists():continue
            deadline=min(value for name in evidence for value in proof(name))
            if deadline>r.clock(ceiling):raise ValueError('source_expiry_ceiling_exceeded')
            files=old.list_files(root);filesets[root]=files
            for path in files:
                if tag=='synthetic_prototype' and path==r.ROOT/PROTOTYPE_CONTEXT:preserved['prototype_context_provenance']+=1;continue
                if tag=='family_evaluation' and minimal_control_provenance(path):preserved['nonpersonal_family_control_provenance']+=1;continue
                if tag=='domain_v11_source' and minimal_control_provenance(path,{'rdd2022-domain-v11-bounded-acquisition-plan-v1','rdd2022-domain-v11-full-transfer-plan-v1','rdd2022-domain-v11-annotation-metadata-plan-v1'}):preserved['source_acquisition_control_provenance']+=1;continue
                if tag=='performance' and old.performance_provenance(path):preserved['performance_source_and_aggregate']+=1;continue
                kind=category(path,tag)
                if kind is None:
                    preserved['weights_or_unclassified']+=1
                    if not any(suffix.lower() in r.WEIGHTS for suffix in path.suffixes) and path.suffix.lower() not in ('.py','.sha256'):unknown[path.suffix.lower()]+=1
                    continue
                try:r.eligible(path,kind)
                except r.PreservedResearchFile as error:preserved[str(error)]+=1;continue
                name=path.relative_to(r.ROOT).as_posix()
                if r.hash_text(name) in previous:raise ValueError('candidate_already_in_historical_inventory')
                local_evidence=list(evidence);local_deadline=deadline
                if tag=='performance':
                    for local in ([path] if path.suffix.lower()=='.json' else [path.parent/'analysis.json']):
                        if not os.path.lexists(local):continue
                        local_name=local.relative_to(r.ROOT).as_posix()
                        local=r.project_path(local_name,file=True)
                        if not r.evidence_expiries(r.read(local)):continue
                        local_deadline=min(local_deadline,*proof(local_name))
                        if local_name not in local_evidence:local_evidence.append(local_name)
                digest,identity=r.file_snapshot(path);snapshots[name]=(digest,identity)
                entries.append({'path':name,'kind':kind,'sha256':digest,'expires_at':r.iso(local_deadline),'evidence':[proofs[key] for key in local_evidence]})
        if not entries:raise ValueError('new_cycle2_candidates_required')
        for root,files in filesets.items():
            if old.list_files(root)!=files:raise ValueError('cycle2_file_set_changed')
        for name,snapshot in snapshots.items():
            if r.file_snapshot(r.project_path(name,file=True))!=snapshot:raise ValueError('cycle2_asset_or_proof_changed')
        stopped_receipt(receipt.relative_to(r.ROOT).as_posix(),receipt_sha256);audit_immutable()
        preparation_guard()
        r.roots(list(filesets))
        r.write(plan,{'schema_version':1,'roots':list(filesets),'entries':entries,'writer_receipt_sha256':receipt_sha256})
        registered=r.register(plan.relative_to(r.ROOT).as_posix(),inventory.relative_to(r.ROOT).as_posix())
        dry=r.purge(inventory.relative_to(r.ROOT).as_posix());audit_immutable()
        if set(r.read(inventory)['entries'])!={r.hash_text(entry['path']) for entry in entries}:raise ValueError('new_inventory_exact_coverage_mismatch')
        if any(dry['counts'][key] for key in ('missing','modified_refused','scope_refused','retired')):raise ValueError('new_inventory_dryrun_consistency_failed')
        result={'format':'public-cycle2-retention-summary-v1',**history,'declared_roots':len(POLICIES),'explicit_roots':len(filesets),
            'absent_optional_roots':len(set(POLICIES)-set(filesets)),'registered_files':len(entries),
            'source_range_files':sum(entry['kind']=='source_range' for entry in entries),'deadline_date_counts':dict(Counter(entry['expires_at'][:10] for entry in entries)),
            'preserved_counts':dict(preserved),'unclassified_extensions':dict(unknown),'historical_controls_unchanged':True,
            'source_deadlines_extended':False,'source_files_deleted':0,'model_weights_deleted':0,'automatic_schedule_configured':False,
            'registration':registered,'dry_run':dry,'unclassified_requires_operator_review':bool(unknown)}
        preparation_guard()
        r.write(summary,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['audit','freeze'],nargs='?',default='audit')
    parser.add_argument('--writers-stopped-receipt');parser.add_argument('--receipt-sha256');args=parser.parse_args()
    if args.action=='audit':print(json.dumps(audit_immutable()[1]))
    else:
        if not args.writers_stopped_receipt or not args.receipt_sha256:parser.error('freeze requires complete writer-stop receipt and pinned SHA')
        print(json.dumps(freeze(args.writers_stopped_receipt,args.receipt_sha256)))


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,TypeError) as error:
        reason=str(error) if isinstance(error,ValueError) and re.fullmatch('[a-z0-9_]+',str(error)) else 'metadata_or_filesystem_error'
        print(json.dumps({'status':'cycle2_preparation_refused','reason':reason,'source_files_deleted':0,'automatic_schedule_configured':False}))
        raise SystemExit(2)
