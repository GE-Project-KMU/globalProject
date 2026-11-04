"""Separate project-only ORT GPU diagnostic Python; leaves app/training intact."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import setup_training
RUNTIME=ROOT/'.venv-performance'
PYTHON=RUNTIME/'python.exe'


def environment():
    previous=setup_training.CACHE
    try:
        setup_training.CACHE=ROOT/'.cache/performance'
        return setup_training.environment()
    finally:setup_training.CACHE=previous


def prepare():
    if os.name!='nt':raise ValueError('windows_diagnostic_bootstrap_only')
    setup_training.RUNTIME=RUNTIME;setup_training.PYTHON=PYTHON;setup_training.CACHE=ROOT/'.cache/performance'
    setup_training.bootstrap()
    command=[str(PYTHON),'-m','pip','install','-r',str(ROOT/'requirements-performance-win-py311.txt'),'--no-warn-script-location']
    subprocess.run(command,cwd=ROOT,env=environment(),check=True)
    subprocess.run([str(PYTHON),'-m','pip','check'],cwd=ROOT,env=environment(),check=True)
    # Package introspection only: do not initialize a CUDA session or torch device.
    subprocess.run([str(PYTHON),'-c',"import onnxruntime as o; print({'ort':o.__version__,'available_providers':o.get_available_providers(),'gpu_session_executed':False})"],cwd=ROOT,env=environment(),check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare']);p.parse_args();prepare()
