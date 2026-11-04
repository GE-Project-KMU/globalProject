"""Isolated Windows CPU evaluation overlay; original training packages are read-only.

Copies the existing official CPython embedded runtime, installs only pinned ORT
into a separate target, and puts that target before the unchanged Torch packages.
"""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256
from scripts.setup_training import environment
BASE=ROOT/'.venv-training';TARGET=ROOT/'.cache/lane-eval-311-ort1311'

def snapshot():
    paths=list(BASE.glob('*'))+[BASE/'Lib/site-packages/onnxruntime/__init__.py',BASE/'Lib/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.pyd',BASE/'Lib/site-packages/torch/__init__.py',BASE/'Lib/site-packages/numpy/__init__.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths if p.is_file()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','probe']);args=parser.parse_args()
    before=snapshot();TARGET.mkdir(parents=True,exist_ok=True);site=TARGET/'site-packages';site.mkdir(exist_ok=True)
    if args.action=='prepare':
        for p in BASE.iterdir():
            if p.is_file() and p.suffix.lower() in ('.exe','.dll','.pyd','.zip','.txt','.cat'):
                shutil.copyfile(p,TARGET/p.name)
        (TARGET/'python311._pth').write_text('python311.zip\n.\nsite-packages\n../../.venv-training/Lib/site-packages\n../..\nimport site\n',encoding='utf-8')
        subprocess.run([str(TARGET/'python.exe'),'-m','pip','install','--target',str(site),'--no-deps','onnxruntime==1.31.1','--no-warn-script-location'],cwd=ROOT,env=environment(),check=True)
    code="import json,sys,torch,numpy,onnxruntime; print(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_path=torch.__file__,numpy=numpy.__version__,numpy_path=numpy.__file__,ort=onnxruntime.__version__,ort_path=onnxruntime.__file__,providers=onnxruntime.get_available_providers())))"
    env=environment();env['CUDA_VISIBLE_DEVICES']=''
    result=subprocess.run([str(TARGET/'python.exe'),'-c',code],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
    probe=json.loads(result.stdout);after=snapshot()
    if before!=after:raise ValueError('original_training_environment_changed')
    if probe['ort']!='1.31.1' or not Path(probe['ort_path']).is_relative_to(TARGET):raise ValueError('isolated_ort_required')
    atomic_json(TARGET/'environment-proof.json',dict(probe=probe,base_before=before,base_after=after,base_unchanged=True,setup_sha256=sha256(Path(__file__))))
    print(json.dumps(probe))
if __name__=='__main__':main()
