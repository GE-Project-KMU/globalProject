from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pytest
from agent.inference_runner import InferenceRunner,create_models,profile_options
from agent.models import validate_cpu_options


class Model:
    def __init__(self,label,barrier=None,fail=False):self.label=label;self.barrier=barrier;self.fail=fail;self.active=False
    def predict(self,value):
        assert not self.active
        self.active=True
        try:
            if self.barrier:self.barrier.wait(timeout=2)
            time.sleep(.005)
            if self.fail:raise ValueError('synthetic_failure')
            return self.label,value
        finally:self.active=False


def test_fast_workers_run_independent_models_and_keep_task_order():
    barrier=threading.Barrier(3);models={task:Model(task,barrier) for task in ('pothole','lane','redaction')}
    with InferenceRunner(models,'fast') as runner:
        result=runner.predict('synthetic')
    assert list(result)==list(models)
    assert all(value==(task,'synthetic') for task,value in result.items())
    assert runner.closed and runner.pool._shutdown
    with pytest.raises(RuntimeError,match='closed'):runner.predict(None)


def test_exception_waits_for_other_sessions_and_cleans_workers():
    models={task:Model(task,fail=task=='lane') for task in ('pothole','lane','redaction')}
    with pytest.raises(ValueError,match='synthetic_failure'):
        with InferenceRunner(models,'fast') as runner:runner.predict(None)
    assert all(not model.active for model in models.values())
    assert runner.closed and runner.pool._shutdown


def test_batches_do_not_race_mutable_model_input():
    models={task:Model(task) for task in ('pothole','lane','redaction')}
    with InferenceRunner(models,'fast') as runner,ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(runner.predict,[1,2]))
    assert results==[{task:(task,value) for task in models} for value in (1,2)]


def test_profiles_validate_and_custom_factory_keeps_single_argument_contract():
    seen=[]
    result=create_models({'a':'path-a','b':'path-b'},'fast',lambda path:seen.append(path) or Model(path))
    assert seen==['path-a','path-b'] and len(result)==2
    assert profile_options('default')=={'cpu_threads':2,'allow_spinning':True,'model_workers':1}
    with pytest.raises(ValueError):profile_options('invented')
    model=Model('shared')
    with pytest.raises(ValueError,match='independent'):InferenceRunner({'a':model,'b':model},'fast')
    for threads,spinning in [(True,True),(0,False),(33,False),(2,1)]:
        with pytest.raises(ValueError):validate_cpu_options(threads,spinning)


def test_cli_fast_profile_is_forwarded_to_real_workflow_entry(monkeypatch,capsys):
    from agent import __main__ as cli
    calls=[]
    monkeypatch.setattr(cli,'execute_config',lambda path,**kwargs:calls.append((path,kwargs)) or {'status':'completed'})
    assert cli.main(['workflow','synthetic-config.json','--cpu-profile','fast'])==0
    assert calls==[('synthetic-config.json',{'cpu_profile':'fast','inference_backend':None,'gpu_profile':None})]
    assert 'completed' in capsys.readouterr().out
