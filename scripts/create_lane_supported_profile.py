"""Create a fresh constrained export child; never overwrite legacy failed proofs."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import workspace_path,atomic_json,sha256
from agent.models import SUPPORTED_CPU_PROFILE
def main():
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();out=workspace_path(a.output)
    if out.exists():raise ValueError('fresh_output_required')
    parent=ROOT/'data/training/lane-a2d2-fp64-mean-v3';info=json.loads((parent/'lane.json').read_text())
    result=json.loads((parent/'full-validation-results.json').read_text())
    if not result.get('complete') or not result['all_raw_passed'] or result['all_post_passed']:raise ValueError('legacy_failure_receipt_required')
    out.mkdir(parents=True);shutil.copyfile(parent/'lane.onnx',out/'lane.onnx')
    provenance=json.loads(workspace_path(info['training_provenance']).read_text());provenance['models']=[(out/'lane.json').relative_to(ROOT).as_posix()];atomic_json(out/'provenance.json',provenance)
    info.update(weights=(out/'lane.onnx').relative_to(ROOT).as_posix(),training_provenance=(out/'provenance.json').relative_to(ROOT).as_posix(),required_runtime_profile=SUPPORTED_CPU_PROFILE,
        parent_manifest_sha256=sha256(parent/'lane.json'),legacy_ort120_full_validation_passed=False,
        numerical_scope='Windows AMD64 ORT1.31.1 CPU only; pending full VAL proof',profile_recipe_sha256=sha256(Path(__file__)))
    atomic_json(out/'lane.json',info);shutil.copyfile(parent/'source.json',out/'source.json')
    atomic_json(out/'artifact-status.json',dict(status='pending_strict_supported_profile_verification',production_approved=False,project_acceptance_eligible=False))
    print(sha256(out/'lane.json'))
if __name__=='__main__':main()
