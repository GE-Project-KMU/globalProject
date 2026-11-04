"""Freeze exclusive lane retry1 resource protocol; never starts CUDA/training."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import sha256,atomic_json

def prepare():
 plan=ROOT/'docs/reports/LANE-CYCLE2-RETRY1-PLAN.json';learning=json.loads(plan.read_text());source=ROOT/'scripts/train_lane_cycle2_retry1.py';observer=ROOT/'scripts/cycle2_device_memory.py'
 if learning['training_code_sha256']!=sha256(source) or learning['cudnn_benchmark'] is not False or learning['requires_exclusive_GPU_training'] is not True:raise ValueError('frozen_exclusive_retry_backend_required')
 value={'format':'lane-exclusive-retry1-supervision-plan-v1','learning_plan_sha256':sha256(plan),'training_source_sha256':sha256(source),'supervisor_source_sha256':sha256(ROOT/'scripts/supervise_lane_cycle2_retry1.py'),'observer_source_sha256':sha256(observer),'preparation_source_sha256':sha256(Path(__file__)),'parent_failed_supervision_plan_sha256':sha256(ROOT/'docs/reports/LANE-CYCLE2-SHARED-SUPERVISION-PLAN.json'),'parent_abort_report_sha256':sha256(ROOT/'docs/reports/CYCLE2-SHARED-GPU-ABORT.json'),'requires_exclusive_GPU_training':True,'max_GPU_training_jobs':1,'GPU_start_requires_new_root_slot':True,'restart_mode':'full_restart_from_v4_not_resume','optimizer_state_resumed':False,'cudnn_benchmark':False,'data_loss_optimizer_schedule_selection_unchanged':True,'bitwise_training_equivalence_claimed':False,'device_budget_bytes':9*1024**3,'device_scope':'nvidia-smi device-reported WDDM FB memory; sampled, no hard quota','global_OOM_zero_claimed':False,'OOM_detection_scope':'own child log only','only_own_child_can_be_stopped':True,'source_expiries_extended':False,'user_recordings_used':False,'TEST_predictions':0}
 target=ROOT/'docs/reports/LANE-CYCLE2-RETRY1-SUPERVISION-PLAN.json'
 if target.exists() and json.loads(target.read_text())!=value:raise ValueError('retry_supervision_plan_already_frozen')
 if not target.exists():atomic_json(target,value)
 print(json.dumps({'plan_sha256':sha256(plan),'supervision_plan_sha256':sha256(target),'GPU_operations':0,'training_updates':0}))
if __name__=='__main__':prepare()
