"""Explicit public-research cycle retention preparation; never deletes files.

Default audit reads only six immutable historical control files. Asset scanning,
hashing and registration require a pinned, explicit writer-stopped receipt.
No user recordings or arbitrary caller-provided roots/tags are accepted.
"""
from __future__ import annotations
import argparse,json,os,re
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_retention as r

STATE='data/research-retention/'
PINS={
 'inventory.json':'a54f568d2c7c2b49427dc8d1b3ac925364f8f5c90160c2ac526b6c4e34b8bc91',
 'improvement-inventory.json':'d9d1802992930412572c7f23cc17aab58d51920ccf1cf67f2ca26ae620d26e1d',
 'followup-inventory.json':'007ed6277aa1c4d1fe97e0cda8f319f0b9e2b224116db00e17002e64dd45680f',
 'frozen-plan.json':'0d8fef1d0f719579faae361ca7ab67c794d7d74b76fd4c31ca5b7c67c1acda75',
 'improvement-frozen-plan.json':'6b380f6653619b7bfa6229ec97aaaaa64e33dab523b64463f9e0d1edf4bf46b7',
 'followup-frozen-plan.json':'889cd5301ae7a4527ef9d5b5f769d866784c35402ec3f72845637a829e4f196c'}
MWPD='data/research/mwpd-v6/source.json'
RDD='data/research/rdd2020/source.json'
LANE='data/public/a2d2-refinement-v3/release.json'
OLD_LANE=['data/public/a2d2/acquisition.json','data/public/a2d2-improvement-v2/acquisition.json']
PRIVACY='data/training/research-redaction-v6/release.json'
OLD_PRIVACY='data/training/research-redaction-v4/release.json'
SCREEN='data/demo-reference/browser-evidence/e8981281-329f-4455-a627-27e6073c7b95'
SCREEN_FOLLOWUP='data/demo-reference/browser-evidence/8dd5ab5e-3d4e-47f2-bd89-eb5d1b3f8e71'
SOAK='data/training/hardware-free-soak-v6'
SOAK_PROOFS=['data/training/pothole-nano-rdd2020-v4/provenance.json',
             'data/training/lane-a2d2-supported-ort131-v3/provenance.json',
             'data/training/redaction-specialists-v4-efficient/provenance.json',SOAK+'/summary.json']
PERF_OLD_PROOFS=['data/cpu-inference-benchmark/public-val-v1/private-plan.json',
                 'data/public/a2d2-improvement-v2/manifest.json',
                 'data/training/research-pothole-rdd2020-v4/release.json',
                 'data/training/research-redaction-v4/release.json']
PERF_NEW='data/gpu-model-sets/final-v6'
PERF_NEW_PROOFS=[PERF_NEW+'/plan.json','data/public/a2d2-refinement-v3/release.json',
                 'data/training/pothole-mwpd-v6/release.json','data/training/research-redaction-v6/release.json']
OPTIONAL_ROOTS={'data/training/research-pothole-clean-v6'}
# Collection roots are exact. The existing retention API only admits the broad
# demo namespace, so entries stay inside the two exact screenshot collections.
POLICIES={
 'data/research/mwpd-v6':('mwpd_source',[MWPD],'2026-11-11'),
 'data/training/research-pothole-mwpd-v6':('pothole_release',[MWPD,'data/training/research-pothole-mwpd-v6/release.json'],'2026-11-11'),
 'data/training/research-pothole-clean-v6':('pothole_clean_evaluation',[RDD,'data/training/research-pothole-clean-v6/release.json'],'2026-11-10'),
 'data/training/pothole-mwpd-v6':('pothole_mixed_model',[MWPD,RDD,'data/training/pothole-mwpd-v6/release.json','data/training/pothole-mwpd-v6/provenance.json'],'2026-11-10'),
 'data/public/a2d2-refinement-v3':('a2d2_source',[LANE,*OLD_LANE],'2026-11-10'),
 'data/training/lane-refinement-v4':('lane_model',[LANE,*OLD_LANE,'data/training/lane-refinement-v4/provenance.json'],'2026-11-10'),
 'data/training/lane-refinement-baseline-v4':('lane_baseline',[LANE,*OLD_LANE],'2026-11-10'),
 'data/training/research-redaction-v6':('redaction_release',[PRIVACY,OLD_PRIVACY],'2026-11-10'),
 'data/training/redaction-refinement-v6':('redaction_model',[PRIVACY,OLD_PRIVACY,'data/training/redaction-refinement-v6/provenance.json'],'2026-11-10'),
 'data/training/refinement-evaluation':('mixed_evaluation',[LANE,*OLD_LANE,PRIVACY,OLD_PRIVACY,MWPD,RDD],'2026-11-10'),
 SCREEN:('rtk_screenshot',['data/public/rtk/acquisition.json',SCREEN+'/private-evidence.json'],'2027-02-01'),
 SCREEN_FOLLOWUP:('rtk_screenshot',['data/public/rtk/acquisition.json',SCREEN_FOLLOWUP+'/private-evidence.json'],'2027-02-01'),
 SOAK:('synthetic_soak',SOAK_PROOFS,'2026-11-10')}
POLICIES.update({'data/cpu-inference-benchmark':('performance_diagnostic',PERF_OLD_PROOFS,'2026-11-10'),
                 'data/gpu-inference-probe':('performance_diagnostic',PERF_OLD_PROOFS,'2026-11-10'),
                 PERF_NEW:('performance_diagnostic',PERF_NEW_PROOFS,'2026-11-10')})

def validate_policies(policies):
    if not isinstance(policies,dict) or set(policies)!=set(POLICIES):raise ValueError('exact_public_cycle_roots_required')
    for root,policy in policies.items():
        if policy!=POLICIES[root]:raise ValueError('fixed_scope_tag_and_evidence_required')
        r.project_path(root)
    return policies

def audit_immutable():
    identifiers=set();inventories=0;plans=0
    for name,pin in PINS.items():
        path=r.project_path(STATE+name,file=True)
        if r.file_snapshot(path)[0]!=pin:raise ValueError('historical_control_snapshot_changed')
        value=r.read(path)
        if name.endswith('inventory.json'):
            inventories+=1
            active={key for key,item in value['entries'].items() if item['status']=='registered'}
            if identifiers&active:raise ValueError('historical_inventory_duplicate_identifiers')
            identifiers.update(active)
            for item in value.get('control_plans',{}).values():
                if item['status']!='registered':continue
                source=item['path']
                if source not in {STATE+name for name in PINS}:raise ValueError('unrecognized_historical_control_plan')
                if r.file_snapshot(r.project_path(source,file=True))[0]!=item['sha256']:raise ValueError('historical_control_plan_changed')
        else:plans+=1
    return identifiers,{'historical_inventories_verified':inventories,'historical_plans_verified':plans,'historical_registered_files':len(identifiers),'historical_source_files_rehashed':False}

def stopped_receipt(path,pin):
    # This is an explicit operator handoff, not a claim that the script can
    # discover every external writer or stop training processes itself.
    path=r.state_path(path)
    if r.file_snapshot(path)[0]!=r.valid_hash(pin):raise ValueError('writer_receipt_digest_mismatch')
    value=r.read(path)
    if set(value)!={'cycle','writers_stopped','confirmed_at','roots'} or value['cycle']!='public-refinement-v6' or value['writers_stopped'] is not True or value['roots']!=sorted(POLICIES):raise ValueError('explicit_complete_writer_stop_handoff_required')
    if r.clock(value['confirmed_at'])>r.clock():raise ValueError('future_writer_handoff_refused')
    return path

def kind(path,tag):
    if any(s.lower() in r.WEIGHTS for s in path.suffixes):return None
    suffix=path.suffix.lower()
    if suffix in r.KINDS['source_archive']:return 'source_archive'
    if suffix in r.KINDS['source_image']:return 'source_image' if tag in {'mwpd_source','a2d2_source'} else 'derived_image'
    if suffix in {'.npy','.npz'}:return 'per_image_prediction'
    if suffix in r.KINDS['identifiable_manifest']:return 'identifiable_manifest'
    if suffix in r.KINDS['source_annotation']:return 'source_annotation'
    return None

def performance_provenance(path):
    if path.name in {'aggregate.json','receipt.json'} or path.suffix.lower() in {'.py','.txt','.sha256'}:return True
    if path.suffix.lower()=='.xml' and path.stat().st_size<=1024*1024:
        import xml.etree.ElementTree as ET
        try:return ET.fromstring(path.read_bytes()).tag.rsplit('}',1)[-1] in {'testsuites','testsuite'}
        except ET.ParseError:return False
    if path.suffix.lower()!='.json':return False
    value=r.read(path)
    if not isinstance(value,dict) or value.get('format') not in {'staging-cuda-model-set-v1','registered-cuda-model-set-v1','verified-local-cuda-bundle-v1'}:return False
    def cases(item):
        if isinstance(item,dict):return any(key=='cases' and bool(child) or cases(child) for key,child in item.items())
        if isinstance(item,list):return any(cases(child) for child in item)
        return False
    return not cases(value)

def list_files(root):
    base=r.project_path(root)
    if not base.is_dir():raise ValueError('completed_public_cycle_root_missing')
    files=[]
    for folder,dirs,names in os.walk(base,followlinks=False):
        for name in sorted(dirs):r.project_path((Path(folder)/name).relative_to(r.ROOT).as_posix())
        for name in sorted(names):
            relative=(Path(folder)/name).relative_to(r.ROOT).as_posix()
            files.append(r.project_path(relative,file=True))
    return sorted(files)

def freeze(receipt_path,receipt_sha256):
    validate_policies(POLICIES)
    receipt=stopped_receipt(receipt_path,receipt_sha256)
    plan=r.state_path(STATE+'public-cycle-frozen-plan.json')
    inventory=r.state_path(STATE+'public-cycle-inventory.json')
    summary=r.state_path(STATE+'public-cycle-summary.json')
    if any(os.path.lexists(p) for p in (plan,inventory,summary)):raise ValueError('new_public_cycle_outputs_required')
    with r.inventory_lock(summary):
        old_ids,history=audit_immutable()
        entries=[];proofs={};proof_dates={};snapshots={};file_sets={};preserved=Counter();unknown=Counter();bytes_total=0
        for root,(tag,evidence,max_date) in POLICIES.items():
            if root in OPTIONAL_ROOTS and not r.project_path(root).exists():continue
            deadlines=[]
            for name in evidence:
                if name not in proofs:
                    path=r.project_path(name,file=True)
                    digest,identity=r.file_snapshot(path);proofs[name]={'path':name,'sha256':digest};snapshots[name]=(digest,identity)
                    proof_dates[name]=r.evidence_expiries(r.read(path))
                dates=proof_dates[name]
                if not dates:raise ValueError('each_source_proof_requires_original_expiry')
                deadlines.extend(dates)
            deadline=min(deadlines)
            if deadline.date().isoformat()>max_date:raise ValueError('original_expiry_date_ceiling_exceeded')
            files=list_files(root);file_sets[root]=files
            if tag=='synthetic_soak':
                completed=r.read(r.project_path(root+'/summary.json',file=True))
                expected=completed.get('saved_PNGs')
                actual=sum(bool(re.fullmatch(r'analysis/[0-9]{9}\.redacted\.png',path.relative_to(r.project_path(root)).as_posix())) for path in files)
                if type(expected) is not int or expected<1 or completed.get('inferred_frames')!=expected or actual!=expected or not r.project_path(root+'/analysis/analysis.json',file=True).is_file():
                    raise ValueError('soak_completed_output_count_mismatch')
            for path in files:
                if tag=='synthetic_soak':
                    local=path.relative_to(r.project_path(root)).as_posix()
                    if local!='analysis/analysis.json' and not re.fullmatch(r'analysis/[0-9]{9}\.redacted\.png',local):
                        preserved['soak_origin_and_execution_provenance']+=1
                        continue
                if tag=='performance_diagnostic' and performance_provenance(path):
                    preserved['performance_aggregate_receipt_and_source_provenance']+=1
                    continue
                relative=path.relative_to(r.ROOT).as_posix();category=kind(path,tag)
                if category is None:
                    preserved['weights' if any(s.lower() in r.WEIGHTS for s in path.suffixes) else 'unclassified_extension']+=1
                    if not any(s.lower() in r.WEIGHTS for s in path.suffixes):unknown[path.suffix.lower()]+=1
                    continue
                try:r.eligible(path,category)
                except r.PreservedResearchFile as error:preserved[str(error)]+=1;continue
                if r.hash_text(relative) in old_ids:raise ValueError('new_candidate_already_in_historical_inventory')
                entry_evidence=list(evidence);entry_deadline=deadline
                if tag=='performance_diagnostic':
                    # A synthetic pipeline can impose a shorter local source
                    # contract (e.g. one hour). Never replace it with Nov10.
                    candidates=[path] if path.suffix.lower()=='.json' else [path.parent/'analysis.json']
                    for candidate in candidates:
                        if not os.path.lexists(candidate):continue
                        candidate=r.project_path(candidate.relative_to(r.ROOT).as_posix(),file=True)
                        name=candidate.relative_to(r.ROOT).as_posix()
                        if name not in proofs:
                            dates=r.evidence_expiries(r.read(candidate))
                            if not dates:continue
                            checksum,fingerprint=r.file_snapshot(candidate);proofs[name]={'path':name,'sha256':checksum};snapshots[name]=(checksum,fingerprint);proof_dates[name]=dates
                        dates=proof_dates[name]
                        if name not in entry_evidence:entry_evidence.append(name)
                        entry_deadline=min(entry_deadline,*dates)
                digest,identity=r.file_snapshot(path);snapshots[relative]=(digest,identity)
                entries.append({'path':relative,'kind':category,'sha256':digest,'expires_at':r.iso(entry_deadline),'evidence':[proofs[p] for p in entry_evidence]});bytes_total+=identity[2]
        if not entries:raise ValueError('new_public_candidates_required')
        # Refuse partial/writing inputs before creating any plan/inventory.
        for root,files in file_sets.items():
            if list_files(root)!=files:raise ValueError('public_cycle_file_set_changed')
        for name,(digest,identity) in snapshots.items():
            if r.file_snapshot(r.project_path(name,file=True))!=(digest,identity):raise ValueError('public_cycle_file_changed')
        stopped_receipt(receipt.relative_to(r.ROOT).as_posix(),receipt_sha256);audit_immutable()
        roots=list(dict.fromkeys('data/demo-reference' if POLICIES[root][0]=='rtk_screenshot' else root for root in file_sets))
        r.roots(roots)
        r.write(plan,{'schema_version':1,'roots':roots,'entries':entries,'collection_scope_tags':{root:POLICIES[root][0] for root in file_sets},'writer_receipt_sha256':receipt_sha256})
        registration=r.register(plan.relative_to(r.ROOT).as_posix(),inventory.relative_to(r.ROOT).as_posix())
        dry=r.purge(inventory.relative_to(r.ROOT).as_posix())
        current=r.read(inventory)
        if set(current['entries'])!={r.hash_text(e['path']) for e in entries}:raise ValueError('new_inventory_coverage_mismatch')
        audit_immutable()
        result={'format':'public-cycle-retention-v1',**history,'declared_collection_roots':len(POLICIES),'explicit_collection_roots':len(file_sets),'absent_optional_roots':len(set(POLICIES)-set(file_sets)),'registered_files':len(entries),'registered_bytes':bytes_total,'deadline_date_counts':dict(Counter(item['expires_at'][:10] for item in current['entries'].values())),
                'historical_controls_unchanged':True,'source_deadlines_extended':False,'source_files_deleted':0,'model_weights_deleted':0,'automatic_schedule_configured':False,'user_recordings_in_scope':False,'preserved_counts':dict(preserved),'unclassified_extensions':dict(unknown),'registration':registration,'dry_run':dry}
        r.write(summary,result)
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['audit','freeze'],nargs='?',default='audit');parser.add_argument('--writers-stopped-receipt');parser.add_argument('--receipt-sha256');args=parser.parse_args()
    if args.action=='audit':print(json.dumps(audit_immutable()[1]))
    else:
        if not args.writers_stopped_receipt or not args.receipt_sha256:parser.error('freeze requires explicit writer-stop handoff and its pinned SHA256')
        print(json.dumps(freeze(args.writers_stopped_receipt,args.receipt_sha256)))

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,TypeError) as error:
        reason=str(error) if isinstance(error,ValueError) and re.fullmatch('[a-z0-9_]+',str(error)) else 'metadata_or_filesystem_error'
        print(json.dumps({'status':'public_cycle_preparation_refused','reason':reason,'source_files_deleted':0,'automatic_schedule_configured':False}))
        raise SystemExit(2)
