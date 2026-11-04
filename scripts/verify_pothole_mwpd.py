"""Three positive VAL cases in actual app, reference ORT and original Torch."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import sha256,atomic_json,workspace_path


def main():
    folder=ROOT/'data/training/pothole-mwpd-v6'
    manifest=folder/'model.json';checkpoint=folder/'best.pt'
    if (folder/'verification-final.json').exists():raise ValueError('canonical_already_attempted_preserve_evidence')
    old=json.loads((ROOT/'data/training/pothole-nano-rdd2020-v4/verification-case.json').read_text(encoding='utf-8'))
    rows=json.loads((folder/'release.json').read_text(encoding='utf-8'))
    new=sorted((r for r in rows if r['split']=='validation' and r['development_domain']=='mwpd' and r['objects']),key=lambda r:r['sample_id'])[:2]
    cases=[old['image']]+[r['path'] for r in new]
    plan={'manifest_sha256':sha256(manifest),'checkpoint_sha256':sha256(checkpoint),'cases':[{'image':image,'sha256':sha256(workspace_path(image,exists=True))} for image in cases],'selection':'existing baseline canonical plus first two sorted positive MWPD VAL; TEST never selected','script_sha256':sha256(Path(__file__))}
    atomic_json(folder/'canonical-plan.json',plan)
    results=[]
    for i,image in enumerate(cases):
        output=folder/('verification-final.json' if i==0 else f'verification-mwpd-positive-{i}.json')
        command=[str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/verify_trained_models.py'),str(manifest),'--image',str(workspace_path(image)),'--checkpoint',str(checkpoint),'--output',str(output)]
        process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (folder/f'canonical-{i}.log').write_text(process.stdout+'\n'+process.stderr,encoding='utf-8')
        report=json.loads(output.read_text(encoding='utf-8')) if output.exists() else {}
        passed=process.returncode==0 and report.get('numerical_parity_passed') is True and report.get('postprocess_parity_passed') is True and report.get('manifest_sha256')==plan['manifest_sha256']
        results.append({'case':i,'passed':passed,'report_sha256':sha256(output) if output.exists() else None,
                        'adapter_result_count':report.get('adapter_result_count'),'numerical_parity_passed':report.get('numerical_parity_passed'),'postprocess_parity_passed':report.get('postprocess_parity_passed')})
        print(json.dumps(results[-1]),flush=True)
        if not passed:raise RuntimeError('positive_VAL_numerical_gate_failed_no_TEST_inference')
    if sha256(manifest)!=plan['manifest_sha256'] or sha256(checkpoint)!=plan['checkpoint_sha256']:raise ValueError('canonical_artifact_changed')
    summary={'format':'pothole-mwpd-canonical-results-v1','manifest_sha256':plan['manifest_sha256'],'checkpoint_sha256':plan['checkpoint_sha256'],'plan_sha256':sha256(folder/'canonical-plan.json'),'positive_validation_cases':len(cases),'all_passed':all(r['passed'] for r in results),'rtol':.001,'atol':.0001,'results':results,'TEST_inference_performed':False,'GPU_support_claimed':False}
    atomic_json(folder/'canonical-summary.json',summary);atomic_json(ROOT/'docs/reports/POTHOLE-MWPD-CANONICAL.json',summary)


if __name__=='__main__':main()
