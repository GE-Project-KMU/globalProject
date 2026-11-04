"""Read-only known project research writer guard; never a global no-writer proof."""
from __future__ import annotations
import json,os,re
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
# Explicit script families, restricted to this project. Resident API services are excluded.
RESEARCH=re.compile(r"^(?:train_|improve_|refine_|supervise_|evaluate_|compare_|verify_trained_models|benchmark_|probe_|verify_gpu_runtime|export_redaction_final12|finalize_privacy_family_v9_results|finalize_privacy_family_v10_results|create_gpu_runtime_profiles|finalize_lane_|select_redaction_inference|gpu_(?:model_set|inference_worker)|prepare_(?:final_gpu|representative)|cycle2_device_memory|privacy_family_).*\.py$",re.I)

def lexical_inside(path,root):
    """String-only boundary check: never stat/resolve an external process path."""
    candidate=os.path.normcase(os.path.abspath(os.path.normpath(os.fspath(path))))
    base=os.path.normcase(os.path.abspath(os.path.normpath(os.fspath(root))))
    try:return os.path.commonpath([candidate,base])==base
    except ValueError:return False

def inspect_writers(root=ROOT,processes=None,own_pid=None):
    root=Path(root).resolve(); own_pid=os.getpid() if own_pid is None else own_pid
    rows=[];unreadable=0;identities=[]
    processes=psutil.process_iter(['pid','ppid','create_time','cmdline','cwd'],ad_value=None) if processes is None else processes
    for process in processes:
        try:
            info=process.info
            if info['pid']==own_pid:continue
            args=info.get('cmdline')
            if not args:unreadable+=1;continue
            cwd=info.get('cwd');scripts=set();identities.append(info)
            if cwd and Path(str(args[0])).stem.lower()=='nvidia-smi' and '--query-gpu=memory.used' in args and '--loop-ms=100' in args and lexical_inside(cwd,root) and Path(cwd).resolve()==root:
                scripts.add('cycle2-device-memory-observer')
            for index,arg in enumerate(args):
                # Command lines never leave this function; only safe known basenames are returned.
                if index and args[index-1]=='-m' and str(arg).startswith('scripts.'):
                    arg=str(arg).replace('.', '/')+'.py'
                candidate=Path(str(arg))
                if not RESEARCH.fullmatch(candidate.name):continue
                if not candidate.is_absolute():
                    if not cwd:continue
                    candidate=Path(cwd)/candidate
                if not lexical_inside(candidate,root):continue
                if candidate.resolve().is_relative_to(root):scripts.add(candidate.name)
            if scripts:rows.append({'pid':info['pid'],'create_time':info.get('create_time'),'scripts':sorted(scripts)})
        except (psutil.NoSuchProcess,psutil.ZombieProcess):continue
        except psutil.AccessDenied:unreadable+=1
    # Include descendants of a recognized writer, even when spawned with -c/-m.
    known={row['pid'] for row in rows}
    for _ in range(len(identities)):
        additions=[i for i in identities if i['pid'] not in known and i.get('ppid') in known and i['pid']!=own_pid]
        if not additions:break
        for info in additions:
            known.add(info['pid']);rows.append({'pid':info['pid'],'create_time':info.get('create_time'),'scripts':['known-research-writer-descendant']})
    return {'format':'cycle2-known-research-writers-v1','known_writers':sorted(rows,key=lambda x:x['pid']),
            'unreadable_processes':unreadable,'global_no_writer_proof':False,'process_control_performed':False}

def require_stopped(root=ROOT):
    report=inspect_writers(root)
    if report['known_writers']:raise ValueError('known_project_research_writer_active')
    return report

if __name__=='__main__':print(json.dumps(inspect_writers()))
