"""Metadata-only aborted-run report; preserves original receipts/checkpoints/logs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def report():
 common=ROOT/'data/training/public-nextcycle';p2=read(common/'P2-shared-training-supervisor.json');lane=read(common/'lane-shared-training-supervisor.json')
 if p2['status']!='failed_supervision' or lane['status']!='failed_supervision':raise ValueError('aborted_receipts_required')
 history=read(ROOT/'data/training/redaction-p2-cycle2-v7/history.json');steps=[]
 for line in (ROOT/'.cache/cycle2-P2-training-supervised.log').read_text(encoding='utf-8').splitlines():
  try:value=json.loads(line)
  except json.JSONDecodeError:continue
  if isinstance(value,dict) and 'step' in value and 'updates' in value:steps.append(value)
 last=steps[-1]
 summary={'format':'cycle2-shared-GPU-abort-v1','status':'aborted_budget_guard_not_completed_training','peak_device_reported_bytes':max(p2['device_memory']['max_device_reported_used_bytes'],lane['device_memory']['max_device_reported_used_bytes']),'device_budget_bytes':9*1024**3,'scope':'sampled nvidia-smi device-reported WDDM FB memory; not exact process sum','own_OOM_signature_observed':{'privacy_P2':p2['own_OOM_signature_observed'],'lane':lane['own_OOM_signature_observed']},'own_child_stopped_after_identity_check':{'privacy_P2':p2['own_child_stopped_after_identity_check'],'lane':lane['own_child_stopped_after_identity_check']},'own_children_running_after_cleanup':{'privacy_P2':p2['own_child_running_after_cleanup'],'lane':lane['own_child_running_after_cleanup']},'original_sources_unchanged':p2['sources_unchanged'] and lane['sources_unchanged'],'original_receipts_sha256':{'privacy_P2':sha256(common/'P2-shared-training-supervisor.json'),'lane':sha256(common/'lane-shared-training-supervisor.json')},'root_reported_third_SmallV9_also_aborted':True,'third_run_independently_verified_by_this_report':False,
 'privacy_P2':{'planned_epochs':12,'completed_epochs':len(history['history']),'interrupted_partial_epoch':last['epoch'],'completed_epoch_optimizer_updates':history['actual_updates'],'last_logged_optimizer_updates_lower_bound':last['updates'],'exact_total_optimizer_updates':None,'development_best_AP50':max(v['validation_face_ap50'] for v in history['history']),'large_face_gate_ever_passed':any(v['large_face_gate_passed'] for v in history['history']),'final_model_manifest_exported':False,'final_ONNX_exported':False,'research_checkpoint_preserved':True,'further_training_scheduled':False},
 'lane':{'completed_epochs':0,'epoch0_validation_not_completed':True,'optimizer_updates':0,'final_model_exported':False,'original_source_sha256':sha256(ROOT/'scripts/train_lane_cycle2.py'),'original_plan_sha256':sha256(ROOT/'docs/reports/LANE-CYCLE2-PLAN.json')},
 'diagnosis':{'confirmed_environment_difference':'Lane training sets cudnn.benchmark=True; its bounded memory probe left the default False','causal_hypothesis':'cuDNN algorithm search/workspace during first lane forward may have contributed to combined peak','causation_proven':False,'preflight_coverage_was_insufficient':True,'reference':'https://docs.pytorch.org/docs/2.7/notes/randomness.html#cuda-convolution-benchmarking'},
 'next_policy':'One GPU training job at a time; CPU/design tasks may run in parallel','lane_retry':'New source/output/parentplan; benchmark=False; same data/loss/optimizer/scheduler/selection; full restart from v4 warmstart, never resume aborted optimizer','bitwise_training_equivalence_claimed':False,'blind_TEST_predictions':0,'user_recordings_used':False,'source_expiries_extended':False,'accuracy_target_claimed':False,'production_approved':False,'failed_files_overwritten':False}
 dest=ROOT/'docs/reports/CYCLE2-SHARED-GPU-ABORT.json'
 if dest.exists() and read(dest)!=summary:raise ValueError('abort_aggregate_already_frozen')
 if not dest.exists():atomic_json(dest,summary)
 print(json.dumps({'preserved_failure_report_sha256':sha256(dest),'completed_P2_epochs':len(history['history']),'P2_last_logged_update_lower_bound':last['updates'],'lane_completed_epochs':0,'TEST_predictions':0}))
if __name__=='__main__':report()
