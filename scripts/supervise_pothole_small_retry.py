"""Fixed own-child capacity training with sampled device and deadline guards."""
import json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    from agent.paths import atomic_json,sha256
    from scripts.setup_training import environment
    from scripts.cycle2_device_memory import DeviceMemoryObserver,OwnedChild
    from scripts.train_pothole_small import DEPS
    OUT="data/training/pothole-small-v9-retry1"
    from scripts.supervise_cycle2_training import own_log_has_oom
    output=ROOT/OUT;source=ROOT/'scripts/train_pothole_small.py';plan=json.loads((output/'plan.json').read_text());receipt=output/'shared-training-supervisor.json'
    import psutil
    retry=json.loads((output/'retry-plan.json').read_text())
    if retry['supervisor_sha256']!=sha256(Path(__file__)):raise ValueError('frozen_retry_supervisor_required')
    for name,digest in retry['copied_input_sha256'].items():
        if sha256(output/name)!=digest or sha256(ROOT/'data/training/pothole-small-v9'/name)!=digest:raise ValueError('restart_learning_inputs_changed')
    for path,digest in retry['stopped_parent_receipts'].items():
        parent=ROOT/path
        if sha256(parent)!=digest:raise ValueError('parent_stop_evidence_changed')
        prior=json.loads(parent.read_text())
        if prior.get('own_child_running_after_cleanup') is not False:raise ValueError('prior_training_not_stopped')
        try:
            process=psutil.Process(prior['own_child_pid'])
            if process.create_time()==prior['own_child_creation_time'] and process.is_running():raise ValueError('prior_training_process_still_running')
        except psutil.NoSuchProcess:pass
    if receipt.exists() or (output/'provenance.json').exists():raise ValueError('fresh_own_child_required')
    if sha256(source)!=plan['source_sha256'] or any(sha256(ROOT/p)!=plan['dependencies'][p] for p in DEPS):raise ValueError('frozen_learning_sources_required')
    probe=json.loads((output/'GPU-memory-preflight.json').read_text())
    if probe['status']!='passed' or probe['plan_sha256']!=sha256(output/'plan.json'):raise ValueError('matching_GPU_memory_preflight_required')
    expiry=datetime.fromisoformat(plan['source_expires_at'].replace('Z','+00:00'))
    if expiry<=datetime.now(timezone.utc):raise ValueError('research_sources_expired')
    report={'format':'small-v9-sequential-restart-supervision-v1','status':'starting','plan_sha256':sha256(output/'plan.json'),'supervisor_sha256':sha256(Path(__file__)),'owns_training_child_only':True,'global_OOM_zero_claimed':False,'training_math_changed':False}
    observer=None;owned=None;log=None;start=time.monotonic()
    try:
        observer=DeviceMemoryObserver(ROOT/'.cache/cycle2-small-retry1-training-observer.log').start()
        if observer.sample_rows()[-1]['used_bytes']>2*2**30:raise ValueError('exclusive_GPU_slot_not_idle')
        logpath=ROOT/'.cache/cycle2-small-retry1-training.log';log=logpath.open('x',encoding='utf-8')
        process=subprocess.Popen([str(ROOT/'.venv-training/python.exe'),str(source),'train','--output',OUT],cwd=ROOT,env=environment(),stdout=log,stderr=log)
        try:owned=OwnedChild(process)
        except BaseException:
            if process.poll() is None:process.terminate();process.wait(timeout=10)
            raise
        report.update(status='running',own_child_pid=owned.pid,own_child_creation_time=owned.creation_time,parent_pid=owned.parent_pid);atomic_json(receipt,report)
        while process.poll() is None:
            observer.raise_if_unsafe()
            if datetime.now(timezone.utc)>=expiry:raise ValueError('source_deadline_reached')
            with logpath.open('rb') as reader:reader.seek(max(0,logpath.stat().st_size-524288));tail=reader.read().decode('utf-8',errors='replace')
            if own_log_has_oom(tail):raise ValueError('own_child_OOM_signature')
            time.sleep(.1)
        observer.raise_if_unsafe();report.update(status='completed' if process.returncode==0 else 'failed_child',child_exit_code=process.returncode)
    except BaseException as error:
        report.update(status='failed',reason=type(error).__name__+':'+str(error))
        if owned:report['own_child_stopped']=owned.stop()
    finally:
        if observer:
            observer.close();report['device_memory']=observer.summary();atomic_json(output/'shared-device-samples.json',{'samples':observer.sample_rows()})
        if log:log.close()
        report['wall_seconds']=time.monotonic()-start
        report['sources_unchanged']=sha256(source)==plan['source_sha256'] and all(sha256(ROOT/p)==plan['dependencies'][p] for p in DEPS)
        if owned:report['own_child_running_after_cleanup']=owned.process.poll() is None
        if not report['sources_unchanged'] or report.get('own_child_running_after_cleanup'):report['status']='failed'
        atomic_json(receipt,report);atomic_json(ROOT/'docs/reports/POTHOLE-SMALL-V9-RETRY1-SUPERVISION.json',{k:v for k,v in report.items() if k not in ('own_child_pid','own_child_creation_time','parent_pid')})
    print(json.dumps({'status':report['status']}))
    if report['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()

