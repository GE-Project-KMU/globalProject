import json
from pathlib import Path
from scripts.check_lane_retry1_contract import functions,benchmark_assignments
from scripts import train_lane_cycle2 as original,train_lane_cycle2_retry1 as retry
from agent.paths import sha256
ROOT=Path(__file__).resolve().parents[1]

def test_retry_keeps_all_learning_geometry_loss_and_export_functions():
 assert functions(ROOT/'scripts/train_lane_cycle2.py')==functions(ROOT/'scripts/train_lane_cycle2_retry1.py')

def test_actual_backend_revision_matches_plan_and_does_not_edit_parent():
 assert benchmark_assignments(ROOT/'scripts/train_lane_cycle2.py')==[True]
 assert benchmark_assignments(ROOT/'scripts/train_lane_cycle2_retry1.py')==[False]
 plan=json.loads((ROOT/'docs/reports/LANE-CYCLE2-RETRY1-PLAN.json').read_text())
 assert plan['cudnn_benchmark'] is False
 assert plan['bitwise_training_equivalence_claimed'] is False
 assert sha256(ROOT/'scripts/train_lane_cycle2.py')=='401fab5253cfa29492ca4747d9a3eee1ee57a17304cbd414731b32200f513a84'

def test_retry_is_fresh_output_from_original_v4_not_partial_checkpoint():
 assert retry.BASE==original.BASE
 assert retry.OUT!=original.OUT
 assert retry.OUT.name=='lane-cycle2-v5-retry1'
 plan=json.loads((ROOT/'docs/reports/LANE-CYCLE2-RETRY1-PLAN.json').read_text())
 assert plan['restart_mode']=='full_restart_from_v4_warmstart'
 assert plan['optimizer_state_resumed'] is False
 assert plan['requires_exclusive_GPU_training'] is True

def test_retry_data_and_learning_hyperparameters_remain_same():
 old=json.loads((ROOT/'docs/reports/LANE-CYCLE2-PLAN.json').read_text());new=json.loads((ROOT/'docs/reports/LANE-CYCLE2-RETRY1-PLAN.json').read_text())
 for key in ['development_sha256','release_sha256','fresh_manifest_sha256','seed','max_epochs','min_epochs','patience','min_delta','batch_size','training_patch_size','input_size','optimizer','scheduler','loss','augmentation','EMA','threshold_grid']:
  assert old[key]==new[key]
 assert new['validation_count']==179 and new['training_count']==1397
 assert 'old100+new79=179' in new['checkpoint_selection']
 assert '119 retained' in new['fresh_evaluation']

def test_retry_supervisor_and_plan_source_bindings():
 p=json.loads((ROOT/'docs/reports/LANE-CYCLE2-RETRY1-SUPERVISION-PLAN-V2.json').read_text())
 assert p['supervisor_source_sha256']==sha256(ROOT/'scripts/supervise_lane_cycle2_retry1.py')
 assert p['max_GPU_training_jobs']==1
 assert p['GPU_start_requires_new_root_slot'] is True
 assert p['learning_plan_sha256']==sha256(ROOT/'docs/reports/LANE-CYCLE2-RETRY1-PLAN.json')

def test_aborted_run_is_not_completed_or_independent_accuracy():
 p=json.loads((ROOT/'docs/reports/CYCLE2-SHARED-GPU-ABORT.json').read_text())
 assert p['privacy_P2']['completed_epochs']==9 and p['privacy_P2']['planned_epochs']==12
 assert p['privacy_P2']['exact_total_optimizer_updates'] is None
 assert p['privacy_P2']['last_logged_optimizer_updates_lower_bound']==7132
 assert p['lane']['completed_epochs']==0
 assert p['blind_TEST_predictions']==0
 assert p['accuracy_target_claimed'] is False

