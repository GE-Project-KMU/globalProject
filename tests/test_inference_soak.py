import math
from types import SimpleNamespace
import pytest
from scripts.probe_inference_soak import Deadline,memory_summary,assert_cpu

def test_deadline_never_starts_another_frame_at_exact_boundary():
    current=[100.]
    deadline=Deadline(600.,clock=lambda:current[0])
    assert deadline.active()
    current[0]=699.999;assert deadline.active()
    current[0]=700.;assert not deadline.active() and deadline.elapsed()==600
    current[0]=701.;assert not deadline.active()

@pytest.mark.parametrize('seconds',[True,0,-1,float('nan'),float('inf')])
def test_invalid_duration_is_refused(seconds):
    with pytest.raises(ValueError):Deadline(seconds)

def test_warmup_memory_spike_is_excluded_without_hiding_later_growth():
    samples=[{'elapsed_seconds':0,'memory_bytes':99999},{'elapsed_seconds':30,'memory_bytes':100},{'elapsed_seconds':40,'memory_bytes':140},{'elapsed_seconds':600,'memory_bytes':150}]
    value=memory_summary(samples,30)
    assert value['samples']==3 and value['max_bytes']==150 and value['last_minus_first_bytes']==50
    assert value['sample_span_seconds']==570 and value['not_a_proof_of_absent_memory_leaks']

def test_missing_post_warmup_samples_is_not_success():
    with pytest.raises(ValueError,match='post_warmup'):memory_summary([{'elapsed_seconds':0,'memory_bytes':1}],30)

def test_provider_and_cpu_thread_configuration_cannot_silently_change():
    options={'cpu_intra_op_threads':2,'cpu_inter_op_threads':1,'allow_spinning':True}
    model=SimpleNamespace(backend='onnxruntime-cpu',net=SimpleNamespace(session=SimpleNamespace(get_providers=lambda:['CPUExecutionProvider']),runtime_options=options))
    assert_cpu(model)
    model.net.session.get_providers=lambda:['CUDAExecutionProvider','CPUExecutionProvider']
    with pytest.raises(ValueError,match='CPU_only'):assert_cpu(model)
    model.net.session.get_providers=lambda:['CPUExecutionProvider'];options['cpu_intra_op_threads']=8
    with pytest.raises(ValueError,match='two_thread'):assert_cpu(model)
