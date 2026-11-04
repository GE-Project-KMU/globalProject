"""Metadata-only enforceable exclusive-start revision for lane retry1."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256

def prepare():
 parent=ROOT/'docs/reports/LANE-CYCLE2-RETRY1-SUPERVISION-PLAN.json';old=json.loads(parent.read_text());source=ROOT/'scripts/supervise_lane_cycle2_retry1.py';small_plan=ROOT/'data/training/pothole-small-v9-retry1/plan.json'
 value=dict(old,format='lane-exclusive-retry1-supervision-plan-v2',parent_protocol_sha256=sha256(parent),revision_reason='pre-training enforcement: completed Smallretry1 receipt/zero exit/source consistency/child identity absent plus two initial device samples each below2GiB BEFORE training Popen; no learning math or data change',supervisor_source_sha256=sha256(source),prerequisite_Small_retry1_receipt='data/training/pothole-small-v9-retry1/shared-training-supervisor.json',prerequisite_Small_retry1_plan_sha256=sha256(small_plan),prerequisite_required_status='completed',prerequisite_required_child_exit_code=0,prerequisite_own_child_running_after_cleanup=False,prerequisite_child_identity_must_be_absent=True,initial_device_samples_required=2,initial_device_used_bytes_strictly_below=2*1024**3,training_Popen_before_gates=0,preparation_source_sha256=sha256(Path(__file__)))
 target=ROOT/'docs/reports/LANE-CYCLE2-RETRY1-SUPERVISION-PLAN-V2.json'
 if target.exists() and json.loads(target.read_text())!=value:raise ValueError('exclusive_start_protocol_already_frozen')
 if not target.exists():atomic_json(target,value)
 print(json.dumps({'supervision_plan_sha256':sha256(target),'supervisor_sha256':sha256(source),'GPU_operations':0,'training_updates':0}))
if __name__=='__main__':prepare()
