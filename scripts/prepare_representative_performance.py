"""Freeze VAL32 metadata for a later exact registered-model GPU IO comparison.

No GPU session, image decode, resize or accuracy evaluation is performed here.
Unknown original capture dates are never replaced with the current time.
"""
from pathlib import Path
from datetime import datetime
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import gpu_model_set as g

GROUPS=('pothole','lane','face','license_plate')


def select_representative(plan,release_rows,canonical):
    grouped={}
    for group in GROUPS:
        task=group if group in ('pothole','lane') else 'redaction'
        allowed={row['sha256'] for row in plan['models'][task]['cases'] if task!='redaction' or row['evaluation_class']==group}
        pool=[row for row in release_rows[task] if row.get('split')=='validation' and row['sha256'] in allowed and
              (task!='redaction' or row.get('evaluation_class')==group)]
        if len({row['sha256'] for row in pool})!=len(pool):raise ValueError('duplicate_representative_source_hash')
        pool.sort(key=lambda row:(row.get('source_id',''),row['sample_id']))
        required=[row for row in pool if row['sha256'] in canonical[task]]
        if len(required)>8:raise ValueError('canonical_cases_exceed_group_budget')
        chosen=required+[row for row in pool if row not in required][:8-len(required)]
        if len(chosen)!=8:raise ValueError('eight_frozen_validation_cases_required_per_group')
        grouped[group]=[{'group':group,**{key:row[key] for key in ('sample_id','source_id','path','sha256','expires_at')}} for row in chosen]
    rows=[grouped[group][index] for index in range(8) for group in GROUPS]
    if len({row['sha256'] for row in rows})!=32:raise ValueError('representative_cases_must_be_unique')
    for task,images in canonical.items():
        if not images<={row['sha256'] for row in rows if row['group']==task or task=='redaction' and row['group'] in ('face','license_plate')}:raise ValueError('representative_canonical_omitted')
    return rows


def freeze(profile_path,output):
    if not g.workspace_path(profile_path,exists=True).is_relative_to((g.ROOT/'data/gpu-runtime-profiles').resolve()):raise ValueError('registered_local_gpu_profile_required')
    profile=g.read(profile_path);plan=g.validate_descriptor(profile,True)
    if not g.workspace_path(profile['plan']['path'],exists=True).is_relative_to((g.ROOT/'data/gpu-model-sets/cycle2-final-v7').resolve()):raise ValueError('new_cycle_model_set_required')
    output=g.workspace_path(output)
    if not output.is_relative_to((g.ROOT/'data/gpu-model-sets/cycle2-final-v7').resolve()):raise ValueError('exact_new_model_performance_root_required')
    if output.exists():raise ValueError('new_representative_plan_required')
    releases={task:g.pinned(entry['validation_manifest']) for task,entry in plan['models'].items()}
    canonical={task:{g.pinned(pointer)['image_sha256'] for pointer in entry['cpu_verifications']} for task,entry in plan['models'].items()}
    rows=select_representative(plan,releases,canonical)
    value={'format':'representative-public-val32-performance-plan-v1','registered_profile':g.pointer(profile_path),
        'inference_plan':profile['plan'],'native_proof':profile['native'],'groups':{group:8 for group in GROUPS},
        'cases':rows,'case_set_sha256':g.canonical(rows),'case_order':'Round-robin pothole/lane/face/license_plate; canonical first then source_id/sample_id.',
        'prepared_input_size':[1280,720],'preparation':'BGR uint8 aspect-preserving letterbox, top-left, padding114; speed input only, no source/GT mutation.',
        'source_expires_at':min(datetime.fromisoformat(row['expires_at'].replace('Z','+00:00')) for row in rows).isoformat(),
        'source_captured_at_provided':False,'original_capture_date_known':False,'replay_timeline':'synthetic frame_index/12; never physical capture/GNSS time',
        'local_material_state':'unreviewed','test_or_user_media_used':False,'accuracy_dataset_or_labels_changed':False,
        'image_read_decode_resize_in_preparation_separately_timed':True,'image_read_decode_resize_in_warmed_scope':False,
        'comparison':'GPU serial/overlap/overlap/serial,32frames each, one run only; PNG/quality/order equality required.',
        'precision':'FP32 unchanged','fresh_model_support_inherited':False,'independent_accuracy_or_field_acceptance_claim':False,
        'gpu_sessions_started':0,'images_read':0}
    g.validate_descriptor(g.read(profile_path),True)
    if value['registered_profile']['sha256']!=g.sha256(profile_path):raise ValueError('registered_profile_changed_during_preparation')
    g.write(output,value)
    return {'representative_cases':32,'case_set_sha256':value['case_set_sha256'],'gpu_sessions_started':0,'images_read':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(freeze(args.profile,args.output)))
