"""Project-only training launcher and Windows bootstrap; never changes system Python."""
from pathlib import Path
import argparse,json,os,subprocess,sys,urllib.request,zipfile

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/".venv-training"
PYTHON=RUNTIME/("python.exe" if os.name=="nt" else "bin/python")
CACHE=ROOT/".cache"/"training"

def environment():
    result=os.environ.copy()
    locations={"TEMP":"tmp","TMP":"tmp","TMPDIR":"tmp","PIP_CACHE_DIR":"pip","TORCH_HOME":"torch","CUDA_CACHE_PATH":"cuda","HF_HOME":"huggingface","XDG_CACHE_HOME":"xdg","MPLCONFIGDIR":"matplotlib","PYTHONPYCACHEPREFIX":"pycache","YOLO_CONFIG_DIR":"yolo","NUMBA_CACHE_DIR":"numba","WANDB_DIR":"wandb","WANDB_CACHE_DIR":"wandb-cache","WANDB_CONFIG_DIR":"wandb-config","PYTHON_EGG_CACHE":"eggs"}
    for key,relative in locations.items():
        path=CACHE/relative;path.mkdir(parents=True,exist_ok=True);result[key]=str(path)
    result["PIP_DISABLE_PIP_VERSION_CHECK"]="1"
    result["PIP_CONFIG_FILE"]=os.devnull
    result["PIP_REQUIRE_VIRTUALENV"]="0"
    result["PYTHONNOUSERSITE"]="1"
    result["TORCH_CUDA_ARCH_LIST"]="7.5"
    return result

def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return
    part=path.with_suffix(path.suffix+".part")
    urllib.request.urlretrieve(url,part);part.replace(path)

def bootstrap():
    if os.name!="nt":raise RuntimeError("This bootstrap provisions Windows; on Linux use a project Python3.11 venv and the pinned requirements")
    RUNTIME.mkdir(exist_ok=True)
    archive=CACHE/"python-3.11.9-embed-amd64.zip"
    if not PYTHON.exists():
        download("https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip",archive)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not (RUNTIME/member.filename).resolve().is_relative_to(RUNTIME.resolve()):raise ValueError("Archive path traversal")
                z.extract(member,RUNTIME)
    (RUNTIME/"python311._pth").write_text("python311.zip\n.\nLib/site-packages\n..\nimport site\n",encoding="utf-8")
    getpip=CACHE/"get-pip.py";download("https://bootstrap.pypa.io/get-pip.py",getpip)
    subprocess.run([str(PYTHON),str(getpip),"--no-warn-script-location"],cwd=ROOT,env=environment(),check=True)

def install():
    env=environment()
    subprocess.run([str(PYTHON),"-m","pip","install","torch==2.7.1","torchvision==0.22.1","--index-url","https://download.pytorch.org/whl/cu126","--no-warn-script-location"],cwd=ROOT,env=env,check=True)
    subprocess.run([str(PYTHON),"-m","pip","install","-r",str(ROOT/"requirements-training.txt"),"--no-warn-script-location"],cwd=ROOT,env=env,check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["bootstrap","install","env","run"]);parser.add_argument("arguments",nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.action=="bootstrap":bootstrap()
    elif args.action=="install":install()
    elif args.action=="env":print(json.dumps({k:v for k,v in environment().items() if k in ("TEMP","TMP","TMPDIR","PIP_CACHE_DIR","TORCH_HOME","CUDA_CACHE_PATH","HF_HOME","XDG_CACHE_HOME","MPLCONFIGDIR","PYTHONPYCACHEPREFIX")},indent=2))
    else:raise SystemExit(subprocess.run([str(PYTHON),*args.arguments],cwd=ROOT,env=environment()).returncode)

if __name__=="__main__":main()
