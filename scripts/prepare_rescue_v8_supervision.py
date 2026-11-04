"""Metadata-only freeze for separate Rescue preflight/train external ownership."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256

def prepare():
 source=ROOT/'scripts/train_redaction_rescue_v8.py';public=ROOT/'docs/reports/PRIVACY-RESCUE-V8-PLAN.json';private=ROOT/'data/training/redaction-rescue-v8/plan.json';plan=json.loads(public.read_text(encoding='utf-8'))
 if sha256(source)!='7bc192db04ff79ff0cad9c3b440aa8357c99f387eecf4a7093ce3f59d9475f65' or sha256(public)!='6bbd59191a2c352a6a2bf7dc0fbd15f25ced0f500be6010fa3011a4e0f94164d' or sha256(private)!=sha256(public):raise ValueError('root_frozen_rescue_learning_inputs_required')
 paths=['scripts/train_redaction_rescue_v8.py','docs/reports/PRIVACY-RESCUE-V8-PLAN.json','data/training/redaction-rescue-v8/plan.json','data/training/redaction-rescue-v8/cpu-contract.json','data/training/research-redaction-cycle2-v7/development.json','data/training/research-redaction-cycle2-v7/release.json','data/training/redaction-specialists-v4-efficient/best.pt','scripts/cycle2_device_memory.py','scripts/supervise_cycle2_training.py',*plan['dependencies_sha256']]
 value={'format':'rescue-v8-external-single-GPU-supervision-plan-v1','supervisor_source_sha256':sha256(ROOT/'scripts/supervise_redaction_rescue_v8.py'),'preparation_source_sha256':sha256(Path(__file__)),'training_source_sha256':sha256(source),'learning_plan_sha256':sha256(public),'CPU_contract_sha256':plan['cpu_contract_sha256'],'input_sha256':{p:sha256(ROOT/p) for p in paths},'prerequisites':{
  'Small_retry1':{'receipt':'data/training/pothole-small-v9-retry1/shared-training-supervisor.json','plan_field':'plan_sha256','plan_sha256':sha256(ROOT/'data/training/pothole-small-v9-retry1/plan.json')},
  'lane_retry1':{'receipt':'data/training/public-nextcycle/lane-retry1-training-supervisor.json','plan_field':'learning_plan_sha256','plan_sha256':sha256(ROOT/'docs/reports/LANE-CYCLE2-RETRY1-PLAN.json')}},
 'required_prior_status':'completed','required_prior_exit_code':0,'required_prior_child_identity_absent':True,'initial_device_samples_each_strictly_below_bytes':2*1024**3,'initial_device_samples':2,'device_reported_budget_bytes':9*1024**3,'requested_query_interval_ms':100,'scope':'sampled device-reported FB memory under WDDM; no hard quota or exact process-sum claim','max_GPU_training_jobs':1,'source_deadline':plan['source_expires_at'],'phases':['preflight','train'],'preflight_is_separate_process':True,'preflight_optimizer_step_attempts':8,'ephemeral_actual_update_count_not_exposed_by_trainer':True,'preflight_state_discarded_before_fresh_construct_training':True,'training_requires_preflight_completed_child_absent_and_exact_report_hash':True,'direct_training_python_with_setup_training_environment':True,'direct_child_binding_avoids_untracked_launcher_descendant':True,'CPU_work_may_run_concurrently':True,'timing_is_not_isolated_camera_FPS':True,'no_training_source_or_math_changes':True,'no_parent_file_overwrites':True,'no_automatic_retry_or_resume':True,'GPU_execution_requires_root_phase_slot':True,'TEST_images':0,'user_recordings_used':False,'source_expiries_extended':False}
 target=ROOT/'docs/reports/PRIVACY-RESCUE-V8-SUPERVISION-PLAN.json'
 if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=value:raise ValueError('rescue_external_protocol_already_frozen')
 if not target.exists():atomic_json(target,value)
 print(json.dumps({'supervision_plan_sha256':sha256(target),'supervisor_sha256':value['supervisor_source_sha256'],'GPU_operations':0}))
if __name__=='__main__':prepare()
