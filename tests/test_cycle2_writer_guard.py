"""Synthetic process/control contracts; no real asset scan or process control."""
import hashlib,json
from types import SimpleNamespace
import pytest
from scripts import cycle2_research_writer_guard as g
from scripts import freeze_cycle2_retention as f
from scripts import research_retention as r

def proc(pid,args,cwd,ppid=0):
    return SimpleNamespace(info=dict(pid=pid,ppid=ppid,cmdline=args,cwd=str(cwd),create_time=123.5))

@pytest.mark.parametrize('name',['supervise_redaction_rescue_v8_retry1.py','privacy_family_v8_retry1_authorized_queue.py','gpu_inference_worker.py','gpu_model_set.py','benchmark_representative_pipeline.py','probe_gpu_pipeline_soak.py','evaluate_lane_retry1.py','compare_privacy_family_parallel.py','verify_gpu_runtime.py','create_gpu_runtime_profiles.py','probe_cycle2_gpu_memory.py','export_redaction_final12.py','finalize_privacy_family_v9_results.py'])
def test_project_writer_is_detected_without_leaking_args(tmp_path,name):
    row=proc(11,[str(tmp_path/'scripts'/name),'private-image.jpg'],tmp_path)
    report=g.inspect_writers(tmp_path,[row],99)
    assert report['known_writers']==[{'pid':11,'create_time':123.5,'scripts':[name]}]
    assert 'private-image' not in json.dumps(report)
    assert report['global_no_writer_proof'] is False and report['process_control_performed'] is False

def test_excludes_self_services_and_external_script(tmp_path):
    rows=[proc(99,[str(tmp_path/'scripts/train_x.py')],tmp_path),proc(2,['python','-m','services.media.app'],tmp_path),proc(3,[str(tmp_path.parent/'train_x.py')],tmp_path.parent)]
    assert not g.inspect_writers(tmp_path,rows,99)['known_writers']

def test_observer_relative_writer_and_descendant(tmp_path):
    rows=[proc(1,['python','scripts/train_x.py'],tmp_path),proc(2,['python','-c','private-data'],tmp_path,1),proc(3,['nvidia-smi','--query-gpu=memory.used','--loop-ms=100'],tmp_path)]
    assert {x['pid'] for x in g.inspect_writers(tmp_path,rows,99)['known_writers']}=={1,2,3}

def test_guard_blocks_without_assets(monkeypatch,tmp_path):
    monkeypatch.setattr(g,'inspect_writers',lambda root:{'known_writers':[{'pid':1}]})
    monkeypatch.setattr(f.r,'ROOT',tmp_path)
    monkeypatch.setattr(f,'protected_controls',lambda:pytest.fail('should block before control reads'))
    with pytest.raises(ValueError,match='writer_active'):f.preparation_guard()

def test_pinned_controls_fail_on_change(tmp_path,monkeypatch):
    p=tmp_path/'data/control.log';p.parent.mkdir();p.write_bytes(b'nonpersonal')
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(f,'PROTECTED_CONTROLS',{'data/control.log':hashlib.sha256(p.read_bytes()).hexdigest()})
    assert f.protected_controls()==1
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='control_changed'):f.protected_controls()

@pytest.mark.parametrize('name',['baseline-VAL508','rescue-VAL508-import','winner-freeze','strict-baseline-face','strict-winner-license_plate','fresh','regression'])
def test_exact_operator_logs_only(name):
    assert f.valid_log_source('.cache/privacy-family-v8-operator-retry1/'+name+'.log')
    assert f.valid_log_source('.cache/privacy-family-v8-operator-retry1/'+name+'-observer.log')

@pytest.mark.parametrize('name',['queue-receipt.json','spawn-control.json','private.log','strict-winner-face.jpg','../fresh.log'])
def test_operator_input_controls_and_arbitrary_files_not_movable(name):
    assert not f.valid_log_source('.cache/privacy-family-v8-operator-retry1/'+name)


def test_project_module_writer_detected(tmp_path):
    report=g.inspect_writers(tmp_path,[proc(5,['python','-m','scripts.gpu_inference_worker'],tmp_path)],99)
    assert report['known_writers'][0]['scripts']==['gpu_inference_worker.py']


def test_noncanonical_operator_separator_is_rejected():
    assert not f.valid_log_source(r'.cache\privacy-family-v8-operator-retry1\fresh.log')


@pytest.mark.parametrize('kind',['cwd','absolute-script','relative-script','parent-traversal'])
def test_external_process_paths_never_resolved(tmp_path,monkeypatch,kind):
    external=tmp_path.parent/'outside-project'
    real=type(tmp_path).resolve
    resolved=[]
    def guarded(path,*args,**kwargs):
        assert g.lexical_inside(path,tmp_path), 'outside workspace resolve attempted'
        resolved.append(str(path))
        return real(path,*args,**kwargs)
    monkeypatch.setattr(type(tmp_path),'resolve',guarded)
    if kind=='cwd':row=proc(3,['nvidia-smi','--query-gpu=memory.used','--loop-ms=100'],external)
    elif kind=='absolute-script':row=proc(3,[str(external/'train_x.py')],tmp_path)
    elif kind=='relative-script':row=proc(3,['scripts/train_x.py'],external)
    else:row=proc(3,['../outside-project/train_x.py'],tmp_path)
    assert not g.inspect_writers(tmp_path,[row],99)['known_writers']
    assert resolved==[str(tmp_path)]


def test_lexical_boundary_rejects_prefix_sibling(tmp_path):
    assert not g.lexical_inside(str(tmp_path)+'-sibling/train_x.py',tmp_path)
    assert not g.lexical_inside(tmp_path/'..'/'train_x.py',tmp_path)
    assert g.lexical_inside(tmp_path/'scripts/train_x.py',tmp_path)


@pytest.mark.parametrize('phase',['CPU-fixed12-export','baseline-VAL508','rescue-VAL508-import','fixed12-VAL508','winner-freeze','strict-baseline-face','strict-baseline-license_plate','strict-winner-face','strict-winner-license_plate','fresh','regression'])
def test_new_V9_operator_phase_logs_have_exact_allowlist(phase):
    assert f.valid_log_source('.cache/privacy-family-v9-operator/'+phase+'.log')
    assert not f.valid_log_source('.cache/privacy-family-v9-operator/'+phase+'-receipt.json')
