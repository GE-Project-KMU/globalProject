"""Metadata-only final model-set input preparation; never freeze or run CUDA.

Full source releases and CPU proof pointers are mandatory. Image bytes are not
opened: original image identities remain for gpu_model_set.freeze to verify.
"""
from __future__ import annotations
import argparse,json,re,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import gpu_model_set as g
from scripts.evaluate_cycle2 import full_gate
from agent.governance import effective_expiry

OUTPUT_ROOT=ROOT/'data/gpu-model-sets/cycle2-final-v7'
SOURCES=(*g.RUNTIME_SOURCES,'scripts/prepare_final_gpu_model_set.py','scripts/evaluate_cycle2.py','scripts/check_lane_cycle2_runtime.py')
COUNTS={'pothole':652,'lane':179,'redaction':508}
CANONICAL_COUNTS={'pothole':3,'lane':3,'redaction':2}
FORMAT='final-cycle2-gpu-input-preparation-v1'
PROTOCOL=ROOT/'docs/reports/FINAL-GPU-MODELSET-PREPARATION-PROTOCOL.json'


def source_hashes():return {name:g.sha256(ROOT/name) for name in SOURCES}


def protocol():
    return {'format':'final-cycle2-gpu-preparation-protocol-v1','source_sha256':source_hashes(),
        'full_VAL_counts':COUNTS,'GPU_subset_counts':g.COUNTS,'required_canonical_counts':CANONICAL_COUNTS,
        'selection':'Required canonical first; remaining rows sorted by (source_id,sample_id) within each fixed group',
        'groups':{'pothole':{'original':10,'mwpd':10},'lane':'two recordings, 50 each from original100+new79',
            'redaction':{'face':20,'license_plate':20}},
        'row_values_changed':False,'image_bytes_read_during_preparation':0,'GPU_sessions_during_preparation':0,
        'TEST_images':0,'source_expiries_extended':False,'old_GPU_proofs_reused':False,'model_or_threshold_changes':False}


def timestamp(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None or parsed<=datetime.now(timezone.utc):raise ValueError('original_source_expired_or_naive')
    return parsed


def pinned(ref):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or not re.fullmatch('[0-9a-f]{64}',ref.get('sha256','')):raise ValueError('exact_metadata_pointer_required')
    return g.pinned(ref)


def binary(ref):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or not re.fullmatch('[0-9a-f]{64}',ref.get('sha256','')) or g.sha256(ref['path'])!=ref['sha256']:raise ValueError('exact_binary_pointer_required')


def ordered(rows):return sorted(rows,key=lambda row:(row.get('source_id',''),row['sample_id']))


def select_group(rows,canonical,count):
    pool=ordered(rows);required=[row for row in pool if row['sha256'] in canonical]
    if len(required)>count:raise ValueError('canonical_group_exceeds_budget')
    chosen=required+[row for row in pool if row['sha256'] not in canonical][:count-len(required)]
    if len(chosen)!=count:raise ValueError('insufficient_validation_group')
    return chosen


def select(task,rows,canonical):
    if task=='pothole':
        if Counter(row.get('development_domain') for row in rows)!={'original':402,'mwpd':250}:raise ValueError('exact_pothole_VAL652_domains_required')
        groups=[([row for row in rows if row['development_domain']==domain and any(obj.get('class')=='pothole' for obj in row.get('objects',[]))],10) for domain in ('original','mwpd')]
    elif task=='lane':
        counts=Counter(row.get('recording') for row in rows)
        if len(counts)!=2 or sorted(counts.values())!=[79,100] or None in counts:raise ValueError('exact_lane_FULLVAL179_recordings_required')
        if any(row['sha256'] in canonical and row.get('positive_pixels',0)<=0 for row in rows):raise ValueError('positive_lane_canonical_GT_required')
        groups=[([row for row in rows if row['recording']==recording],50) for recording in sorted(counts)]
    else:
        if Counter(row.get('evaluation_class') for row in rows)!={'face':400,'license_plate':108}:raise ValueError('exact_privacy_DEV508_classes_required')
        groups=[([row for row in rows if row['evaluation_class']==label and any(obj.get('class')==label for obj in row.get('objects',[]))],20) for label in ('face','license_plate')]
    result=[row for pool,count in groups for row in select_group(pool,canonical,count)]
    if not canonical<={row['sha256'] for row in result}:raise ValueError('canonical_not_positive_or_missing_from_subset')
    return result


def lane_full(entry,manifest,dev,rows):
    full=pinned(entry['full_validation']);plan=pinned(entry['full_validation_plan']);source_plan=pinned(entry['full_validation_source_plan'])
    full_gate(full,entry['manifest']['sha256'],entry['checkpoint']['sha256'],179)
    if source_plan.get('validation_count')!=179 or source_plan.get('development_sha256')!=dev['sha256'] or source_plan.get('release_sha256')!=entry['full_release']['sha256']:raise ValueError('predeclared_lane_FULLVAL179_source_binding_required')
    if full.get('weights_sha256')!=manifest['sha256'] or full.get('plan_sha256')!=entry['full_validation_plan']['sha256'] or full.get('worker_source_sha256')!=g.sha256(ROOT/'scripts/check_lane_cycle2_runtime.py') or full.get('test_accessed') is not False:raise ValueError('full_VAL_source_weight_plan_binding_required')
    if plan.get('manifest_sha256')!=entry['manifest']['sha256'] or plan.get('checkpoint_sha256')!=entry['checkpoint']['sha256'] or plan.get('development_sha256')!=dev['sha256'] or plan.get('rtol')!=.001 or plan.get('atol')!=.0001 or plan.get('test_accessed') is not False:raise ValueError('full_VAL_plan_artifact_development_binding_required')
    expected=[{key:row[key] for key in ('path','sha256','expires_at')} for row in sorted(rows,key=lambda row:row['source_id'])]
    if plan.get('cases')!=expected or any(case.get('image_sha256')!=expected[case['index']]['sha256'] for case in full['cases']):raise ValueError('complete_original_FULLVAL179_image_binding_required')
    return {'report':entry['full_validation'],'plan':entry['full_validation_plan'],'source_plan':entry['full_validation_source_plan']}


def validate_entry(task,entry):
    model=pinned(entry['manifest']);binary(entry['checkpoint']);release=pinned(entry['full_release']);dev=pinned(entry['development_manifest'])
    if model.get('task')!=task or model.get('research_only') is not True or model.get('production_approved') is not False or g.sha256(model['weights'])!=model.get('sha256'):raise ValueError('exact_research_model_weights_required')
    provenance=pinned(entry['training_provenance'])
    if model.get('training_provenance')!=entry['training_provenance']['path'] or provenance.get('release_sha256')!=entry['full_release']['sha256'] or provenance.get('task')!=task or any(value.get(key) is True for value in (model,provenance) for key in ('data_revoked','needs_revalidation')):raise ValueError('valid_exact_training_provenance_required')
    # Lane legacy-compatible export omits this field; its three canonical
    # coupons AND complete full-VAL report/plan must bind the checkpoint below.
    if (task!='lane' or 'trained_checkpoint_sha256' in model) and model.get('trained_checkpoint_sha256')!=entry['checkpoint']['sha256']:raise ValueError('model_checkpoint_binding_required')
    if model.get('source_release_path')!=entry['full_release']['path'] or model.get('dataset_release')!='manifest-sha256:'+entry['full_release']['sha256']:raise ValueError('full_parent_release_binding_required')
    if 'training_development_release_path' in model and (model['training_development_release_path']!=entry['development_manifest']['path'] or model.get('training_development_release_sha256')!=entry['development_manifest']['sha256']):raise ValueError('development_release_binding_required')
    if not isinstance(release,list) or not isinstance(dev,list):raise ValueError('release_rows_required')
    rows=[row for row in dev if row.get('split')=='validation']
    if len(rows)!=COUNTS[task] or len({r['sample_id'] for r in rows})!=len(rows) or len({r['sha256'] for r in rows})!=len(rows):raise ValueError('complete_unique_validation_membership_required')
    parents=[g.canonical(row) for row in release if row.get('split')=='validation']
    if Counter(g.canonical(row) for row in rows)!=Counter(parents):raise ValueError('unaltered_complete_parent_validation_rows_required')
    if any(row.get('split') not in ('train','validation') for row in dev) and entry['development_manifest']!=entry['full_release']:raise ValueError('development_cannot_include_TEST')
    for row in rows:
        timestamp(row['expires_at']);g.public_case_path(row['path'])
        if not re.fullmatch('[0-9a-f]{64}',row['sha256']):raise ValueError('source_image_hash_required')
    source_expiry=min(timestamp(effective_expiry(row)) for row in release)
    for asset in provenance.get('assets',[]):source_expiry=min(source_expiry,timestamp(asset['expires_at']))
    proofs=entry['cpu_verifications']
    if not isinstance(proofs,list) or len(proofs)!=CANONICAL_COUNTS[task]:raise ValueError('all_required_canonical_coupons_required')
    canonical=set()
    for ref in proofs:
        proof=pinned(ref);g.proof_passed(proof,model,entry['manifest']['sha256'])
        if proof.get('checkpoint_sha256')!=entry['checkpoint']['sha256']:raise ValueError('canonical_checkpoint_binding_required')
        canonical.add(proof['image_sha256'])
    if len(canonical)!=CANONICAL_COUNTS[task]:raise ValueError('distinct_canonical_coupons_required')
    selected=select(task,rows,canonical)
    if task=='redaction' and Counter(row['evaluation_class'] for row in rows if row['sha256'] in canonical)!={'face':1,'license_plate':1}:raise ValueError('face_and_plate_canonical_required')
    full=lane_full(entry,model,entry['development_manifest'],rows) if task=='lane' else None
    evidence={'manifest':entry['manifest'],'checkpoint':entry['checkpoint'],'full_release':entry['full_release'],
        'development_manifest':entry['development_manifest'],'cpu_verifications':proofs,'weights_sha256':model['sha256'],
        'training_provenance':entry['training_provenance'],
        'full_VAL_membership_sha256':g.canonical(rows),'subset_membership_sha256':g.canonical(selected),
        'source_expiry_min':source_expiry.isoformat()}
    if full:evidence['full_lane_validation']=full
    return selected,evidence


def prepare(request_path,output):
    request=g.read(request_path);request_ref=g.pointer(request_path)
    if request.get('format')!=FORMAT or set(request.get('models',{}))!=set(g.TASKS) or request.get('source_sha256')!=source_hashes() or request.get('TEST_images')!=0 or type(request.get('TEST_images')) is not int or request.get('GPU_used') is not False:raise ValueError('frozen_metadata_preparation_request_required')
    if request.get('preparation_protocol')!=g.pointer(PROTOCOL) or pinned(request['preparation_protocol'])!=protocol():raise ValueError('predeclared_source_bound_preparation_protocol_required')
    out=g.workspace_path(output)
    if out==OUTPUT_ROOT or not out.is_relative_to(OUTPUT_ROOT) or out.exists():raise ValueError('fresh_final_modelset_child_required')
    subsets={};evidence={}
    # Validate every artifact/proof before creating any output directory.
    for task in g.TASKS:subsets[task],evidence[task]=validate_entry(task,request['models'][task])
    def recheck_refs(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'}:
                if g.sha256(value['path'])!=value['sha256']:raise ValueError('input_changed_during_preparation')
            else:
                for item in value.values():recheck_refs(item)
        elif isinstance(value,list):
            for item in value:recheck_refs(item)
    recheck_refs(request)
    for task,entry in request['models'].items():
        model=pinned(entry['manifest'])
        if g.sha256(model['weights'])!=evidence[task]['weights_sha256']:raise ValueError('weights_changed_during_preparation')
    if source_hashes()!=request['source_sha256'] or g.pointer(request_path)!=request_ref:raise ValueError('preparation_source_or_request_changed')
    specification={'models':{},'format':'final-cycle2-cuda-input-specification-v1','preparation_request':request_ref,
        'preparation_source_sha256':request['source_sha256'],'metadata_only':True,'image_bytes_read':0,'GPU_sessions':0,
        'TEST_images':0,'source_expiries_extended':False,'source_evidence':evidence}
    out.mkdir(parents=True)
    for task in g.TASKS:
        path=out/(task+'-GPU-VAL-subset.json');g.write(path,subsets[task]);entry=request['models'][task]
        specification['models'][task]={'manifest':entry['manifest'],'cpu_verifications':entry['cpu_verifications'],
            'validation_manifest':g.pointer(path),'case_sha256s':[row['sha256'] for row in subsets[task]]}
    g.write(out/'input-specification.json',specification)
    return {'status':'metadata_prepared','counts':g.COUNTS,'image_bytes_read':0,'GPU_sessions':0,'TEST_images':0,
        'input_specification_sha256':g.sha256(out/'input-specification.json'),'plan_frozen':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('request');parser.add_argument('output');args=parser.parse_args()
    print(json.dumps(prepare(args.request,args.output)))


if __name__=='__main__':main()
