"""Freeze a full restart after sampled memory stop; never resume optimizer state."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import sha256,atomic_json

def main():
    parent=ROOT/'data/training/pothole-small-v9';out=ROOT/'data/training/pothole-small-v9-retry1'
    names=('plan.json','release.json','preflight.json','GPU-memory-preflight.json')
    receipts=['data/training/pothole-small-v9/shared-training-supervisor.json','data/training/public-nextcycle/P2-shared-training-supervisor.json','data/training/public-nextcycle/lane-shared-training-supervisor.json']
    if (out/'retry-plan.json').exists():raise ValueError('preserve_frozen_restart_protocol')
    for name in names:
        if sha256(parent/name)!=sha256(out/name):raise ValueError('identical_learning_inputs_required')
    for path in receipts:
        data=json.loads((ROOT/path).read_text())
        if data.get('own_child_running_after_cleanup') is not False or data.get('sources_unchanged') is not True or data.get('device_memory',{}).get('over_budget_observed') is not True:raise ValueError('completed_resource_stop_evidence_required')
    value={'format':'pothole-small-v9-sequential-full-restart-v1','copied_input_sha256':{n:sha256(out/n) for n in names},'stopped_parent_receipts':{p:sha256(ROOT/p) for p in receipts},'supervisor_sha256':sha256(ROOT/'scripts/supervise_pothole_small_retry.py'),'learning_source_sha256':sha256(ROOT/'scripts/train_pothole_small.py'),'restart_from_official_initialization':True,'optimizer_state_resumed':False,'prior_partial_weights_used':False,'GPU_training_policy':'one training process at a time; CPU/code work may run concurrently','sampled_device_limit_GiB':9,'initial_device_idle_limit_GiB':2,'learning_recipe_changed':False,'TEST_used':False,'memory_failure_limits':['Probe batches did not cover full shared training peak','10.32GiB device peak observed; no own-log OOM signatures','CuDNN autotune/workspaces possible cause, not established by allocation trace']}
    atomic_json(out/'retry-plan.json',value);atomic_json(ROOT/'docs/reports/POTHOLE-SMALL-V9-RETRY1-PLAN.json',value)
    print(json.dumps({'retry_plan_sha256':sha256(out/'retry-plan.json'),'training_started':False}))
if __name__=='__main__':main()
