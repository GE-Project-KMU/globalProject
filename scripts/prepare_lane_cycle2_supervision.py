"""Metadata-only lane shared-slot protocols; no CUDA initialization or pixels."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import sha256,atomic_json

def freeze(path,value):
 if path.exists():
  if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('lane_shared_protocol_already_frozen')
 else:atomic_json(path,value)
def prepare():
 plan=ROOT/'docs/reports/LANE-CYCLE2-PLAN.json';learning=json.loads(plan.read_text());source=ROOT/'scripts/train_lane_cycle2.py'
 if learning['training_code_sha256']!=sha256(source):raise ValueError('frozen_learning_source_changed')
 common={'learning_plan_sha256':sha256(plan),'training_source_sha256':sha256(source),'learning_source_sha256':sha256(source),'observer_source_sha256':sha256(ROOT/'scripts/cycle2_device_memory.py'),'preparation_source_sha256':sha256(Path(__file__)),'device_budget_bytes':9*1024**3,'GPU_start_requires_new_root_slot':True,'must_wait_for_root_Nano_GPU_release':True,'source_expiries_extended':False,'user_recordings_used':False,'TEST_images_loaded':0,'scope':'device-reported sampled FB memory under WDDM; not exact process sum or hard quota'}
 freeze(ROOT/'docs/reports/LANE-CYCLE2-GPU-PREFLIGHT-PLAN.json',dict(common,format='lane-shared-memory-preflight-plan-v1',probe_source_sha256=sha256(ROOT/'scripts/probe_cycle2_lane_memory.py'),optimizer_steps=0,training_updates=0,max_batches=2,TRAIN_AMP_batch_size=2,VAL_FP32_batch_size=1,input_size=1024,selection='two TRAIN images with highest declared paint fraction and sourceID tie; first positive sourceID-sorted VAL; selected before GPU; no TEST',source_and_mask_sha_verified_before_GPU=True,model_weights_saved=False,model_factory_sha256=learning['factory_sha256'],tensor_math_from_frozen_learning_source=True,device_samples_before_init=2,additional_device_samples_after_batches=2))
 freeze(ROOT/'docs/reports/LANE-CYCLE2-SHARED-SUPERVISION-PLAN.json',dict(common,format='lane-shared-supervision-plan-v1',supervisor_source_sha256=sha256(ROOT/'scripts/supervise_cycle2_lane.py'),command_scope='fixed workspace Python3.11 + train_lane_cycle2.py train only',identity='own direct child PID/create_time/PPID; never other model process',only_own_lane_training_child_can_be_stopped=True,training_math_changed=False,global_OOM_zero_claimed=False,OOM_detection_scope='own child log only'))
 print(json.dumps({'lane_plan_sha256':sha256(plan),'learning_source_unchanged':True,'GPU_operations':0,'training_updates':0}))
if __name__=='__main__':prepare()
