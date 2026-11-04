"""Soak preparation contracts only; no real model/GPU/ten-minute success claim."""
from pathlib import Path
import uuid
import pytest
from scripts import probe_gpu_pipeline_soak as soak


@pytest.mark.parametrize('duration,rate,budget,reserve',[
    (599,12,soak.BUDGET,soak.RESERVE),(True,12,soak.BUDGET,soak.RESERVE),
    (600,10,soak.BUDGET,soak.RESERVE),(600,float('nan'),soak.BUDGET,soak.RESERVE),
    (600,21,soak.BUDGET,soak.RESERVE),(600,12,100,soak.RESERVE),
    (600,12,soak.BUDGET,0),
])
def test_duration_rate_budget_never_silently_reduced(duration,rate,budget,reserve):
    with pytest.raises(ValueError):soak.settings(duration,rate,budget,reserve)


def test_budget_bound_and_approval_are_checked_without_gpu():
    value=soak.settings(600,12,soak.BUDGET,soak.RESERVE)
    assert value['maximum_frames']==7202 and value['worst_case_output_bound_bytes']<=soak.BUDGET
    with pytest.raises(ValueError,match='approval_required'):soak.run('not-read','not-created')


def test_real_synthetic_1280_decoder_sequential_frames_and_release():
    folder=soak.ROOT/'.cache/gpu-soak-tests'/uuid.uuid4().hex;folder.mkdir(parents=True)
    path=folder/'nonpersonal.mp4';soak.generated_video(path);observed=[]
    samples=soak.decoded_samples(path,duration=10,rate=1000,maximum_frames=10000,observe=lambda *args:observed.append(args))
    try:frames=[next(samples) for _ in range(3)]
    finally:samples.close()
    assert [row['frame_id'] for row in frames]==[0,1,2]
    assert all(row['frame'].shape==(720,1280,3) for row in frames)
    assert [row[0] for row in observed]==[0,1,2]
    with pytest.raises(ValueError,match='new_generated_video_required'):soak.generated_video(path)


def test_memory_summary_reports_growth_honestly_and_requires_warmup():
    with pytest.raises(ValueError):soak.memory_summary([])
    rows=[{'elapsed_seconds':t,'main_rss_bytes':m,'worker_rss_bytes':{'lane':20},'gpu':{'device_used_MiB':300}} for t,m in [(0,999),(30,100),(40,150)]]
    value=soak.memory_summary(rows)
    assert value['host_rss_bytes']['main']['last_minus_first']==50
    assert value['gpu_scope']=='whole_device_not_per_worker' and value['absence_of_memory_leaks_proven'] is False
