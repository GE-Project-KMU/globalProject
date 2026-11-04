"""Reuse the three frozen positive VAL cases for one new family artifact."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256,workspace_path

def main(manifest):
    manifest=workspace_path(manifest,exists=True);folder=manifest.parent;checkpoint=folder/'best.pt';out=folder/'canonical-plan.json'
    if out.exists():raise ValueError('preserve_canonical_attempt')
    info=json.loads(manifest.read_text());rows=json.loads(workspace_path(info['source_release_path'],exists=True).read_text());old=ROOT/'data/training/pothole-mwpd-v6/canonical-plan.json';cases=json.loads(old.read_text())['cases'];val={r['sha256']:r for r in rows if r['split']=='validation' and r['objects']}
    if len(cases)!=3 or any(c['sha256'] not in val or val[c['sha256']]['path']!=c['image'] for c in cases):raise ValueError('unchanged_positive_VAL_cases_required')
    from agent.dataset import validate_research_release
    if not validate_research_release([val[c['sha256']] for c in cases])['valid']:raise ValueError('current_licensed_hashed_VAL_required')
    plan={'format':'pothole-family-canonical-plan-v1','manifest_sha256':sha256(manifest),'checkpoint_sha256':sha256(checkpoint),'parent_case_plan_sha256':sha256(old),'source_sha256':sha256(Path(__file__)),'cases':cases,'rtol':.001,'atol':.0001,'TEST_images':0,'reference_ORT':'1.31.1'};atomic_json(out,plan);results=[]
    for i,c in enumerate(cases):
        proof=folder/('verification-final.json' if i==0 else f'verification-mwpd-positive-{i}.json')
        command=[str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/verify_trained_models.py'),str(manifest),'--image',c['image'],'--checkpoint',str(checkpoint),'--training-python',str(ROOT/'.cache/lane-eval-311-ort1311/python.exe'),'--output',str(proof)]
        p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace');(folder/f'canonical-{i}.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8');r=json.loads(proof.read_text()) if proof.exists() else {}
        passed=p.returncode==0 and all(r.get(k) is True for k in ('artifact_contract_verified','numerical_parity_passed','postprocess_parity_passed')) and r.get('adapter_result_count',0)>0 and r.get('manifest_sha256')==plan['manifest_sha256']
        results.append({'index':i,'passed':passed,'proof_sha256':sha256(proof) if proof.exists() else None,'adapter_result_count':r.get('adapter_result_count')});print(json.dumps(results[-1]),flush=True)
        if not passed:raise ValueError('canonical_gate_failed_no_TEST')
    if sha256(manifest)!=plan['manifest_sha256'] or sha256(checkpoint)!=plan['checkpoint_sha256']:raise ValueError('canonical_artifacts_changed')
    result={'format':'pothole-family-canonical-results-v1','manifest_sha256':sha256(manifest),'checkpoint_sha256':sha256(checkpoint),'plan_sha256':sha256(out),'rtol':.001,'atol':.0001,'all_passed':True,'positive_VAL_cases':3,'results':results,'TEST_inference_performed':False,'reference_ORT':'1.31.1'};atomic_json(folder/'canonical-summary.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');main(p.parse_args().manifest)
