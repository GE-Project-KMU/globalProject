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

STATE='data/research-retention/'
PINS={**old.PINS,
 'public-cycle-inventory.json':'7aeec19e4445366124c9faa50cc48b164eaf9fe0f4fa758d91fa362d68f41599',
 'public-cycle-frozen-plan.json':'65e8f391acbb4a8d086d8fbbd0a13154851c248cceb286eb1109369970ab3d44'}
LANE='data/public/a2d2-cycle2-v5/source.json'
PRIVACY='data/training/research-redaction-cycle2-v7/release.json'
BLIND='data/research/rdd2022-blind-v7/source.json'
TINY='data/training/pothole-tiny-v7/release.json'
PROOF_PINS={LANE:'af18e33a8cc42cebc2618776594a50fcc5d8886b3c0a4edab7f9b6b9ff668006',
 PRIVACY:'5ae56bd60700ce17a6af8e92edf630a2722ebc6148e177b798afa172cbff50dc',
 BLIND:'38a5002db939310c8d5a6a6b0977a917be6801ad148403b7647e30c054c78d6b',
 TINY:'43bcbd0c5160276dc8de9507e40db0a8776ff35e74faf971dcdf69fdddd6d52f'}
LANE_CAP='2026-11-10T17:07:26.069653Z';PRIVACY_CAP='2026-11-10T16:55:19.285523Z';BLIND_CAP='2026-11-10T17:07:24.286159Z'
POLICIES={
 'data/research/rdd2022-blind-v7':('blind_source',[BLIND],BLIND_CAP),
 'data/training/research-pothole-rdd2022-blind-v7':('blind_release',[BLIND],BLIND_CAP),
 'data/training/research-pothole-rdd2022-us100-v7':('blind_release',[BLIND,'data/training/research-pothole-rdd2022-us100-v7/split-evidence.json'],BLIND_CAP),
 'data/training/pothole-validation-diagnostic-v7':('pothole_diagnostic',['data/training/pothole-mwpd-v6/release.json','data/training/pothole-mwpd-v6/provenance.json'],'2026-11-10T17:14:46.862846Z'),
 'data/training/pothole-tiny-v7':('pothole_model',[TINY,'data/training/pothole-tiny-v7/provenance.json'],'2026-11-10T17:14:46.862846Z'),
 'data/public/a2d2-cycle2-v5':('lane_source',[LANE],LANE_CAP),
 'data/training/research-redaction-cycle2-v7':('privacy_source',[PRIVACY],PRIVACY_CAP),
 'data/training/lane-cycle2-v5':('lane_model',[LANE,'data/training/lane-cycle2-v5/provenance.json'],LANE_CAP),
 'data/training/redaction-p2-cycle2-v7':('privacy_model',[PRIVACY,'data/training/redaction-p2-cycle2-v7/provenance.json'],PRIVACY_CAP),
 'data/training/public-nextcycle':('mixed',[LANE,PRIVACY],PRIVACY_CAP),
 'data/training/cycle2-evaluation':('mixed',[LANE,PRIVACY,BLIND],PRIVACY_CAP),
 'data/training/cycle2-logs':('mixed',[LANE,PRIVACY,BLIND],PRIVACY_CAP),
 'data/gpu-model-sets/io-overlap-v1':('performance',[old.PERF_NEW+'/plan.json'],PRIVACY_CAP),
 'data/gpu-model-sets/cycle2-final-v7':('performance',[LANE,PRIVACY,BLIND,'data/gpu-model-sets/cycle2-final-v7/plan.json'],PRIVACY_CAP)}
OPTIONAL_ROOTS={'data/training/cycle2-logs','data/training/research-pothole-rdd2022-blind-v7'}
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
    if path.suffix.lower()=='.bin':return 'source_range' if path.parent==r.ROOT/r.RANGE_ROOT else None
    value=old.kind(path,tag)
    if value=='derived_image' and tag in ('blind_source','blind_release','lane_source','privacy_source'):return 'source_image'
    return value


def prepare_log_migration(receipt_path,receipt_sha256,manifest_path,manifest_sha256,output_path):
    """Exact operator-supplied cache logs only; write a reversible move PLAN, not a move."""
    stopped_receipt(receipt_path,receipt_sha256);audit_immutable()
    manifest=r.state_path(manifest_path)
    if r.file_snapshot(manifest)[0]!=r.valid_hash(manifest_sha256):raise ValueError('log_manifest_digest_mismatch')
    value=r.read(manifest)
    if set(value)!={'format','logs'} or value['format']!='cycle2-exact-log-list-v1' or not isinstance(value['logs'],list) or not 1<=len(value['logs'])<=100:raise ValueError('exact_log_list_required')
    output=r.state_path(output_path)
    if output.exists():raise ValueError('new_log_migration_plan_required')
    entries=[];snapshots={};seen=set();aggregate=0
    for item in value['logs']:
        if set(item)!={'source','sha256'} or not re.fullmatch(r'\.cache/cycle2[a-zA-Z0-9_.-]*\.log',item['source']):raise ValueError('exact_cycle2_cache_log_required')
        if item['source'] in seen:raise ValueError('duplicate_log_manifest_entry')
        seen.add(item['source']);source=r.project_path(item['source'],file=True)
        snapshot=r.file_snapshot(source)
        if snapshot[0]!=r.valid_hash(item['sha256']):raise ValueError('log_source_digest_mismatch')
        snapshots[item['source']]=snapshot
        try:r.eligible(source,'identifiable_manifest')
        except r.PreservedResearchFile:aggregate+=1;continue
        destination='data/training/cycle2-logs/'+source.name
        if r.project_path(destination).exists():raise ValueError('log_migration_destination_exists')
        entries.append({'from_log':item['source'],'to_log':destination,'sha256':snapshot[0]})
    for name,snapshot in snapshots.items():
        if r.file_snapshot(r.project_path(name,file=True))!=snapshot:raise ValueError('log_changed_during_preparation')
    stopped_receipt(receipt_path,receipt_sha256);audit_immutable()
    r.write(output,{'format':'cycle2-reversible-log-move-plan-v1','writer_receipt_sha256':receipt_sha256,
        'exact_log_manifest_sha256':manifest_sha256,'moves':entries,'aggregate_logs_preserved':aggregate,
        'moves_executed':False,'source_files_deleted':0,'requires_manual_scoped_move_and_result_receipt':True})
    return {'planned_private_log_moves':len(entries),'aggregate_logs_preserved':aggregate,'moves_executed':False,'source_files_deleted':0}


def freeze(receipt_path,receipt_sha256):
    # Handoff first: no source proofs, file lists or asset hashes while active.
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
