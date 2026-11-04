import numpy as np
import pytest
from scripts import benchmark_inference_cpu as benchmark


def test_invalid_cpu_settings_refuse_before_model_load():
    for threads,spinning in [(0,False),(9,False),(True,False),(2,0)]:
        with pytest.raises(ValueError,match='invalid_cpu_settings'):
            benchmark.ConfiguredNetwork('absent.onnx',threads,spinning)


def test_lane_equal_raw_but_changed_mask_does_not_pass():
    raw=np.array([[[[1.]],[[2.]]]],np.float32)
    result=benchmark.parity('lane',raw,np.zeros((1,1),np.uint8),raw,np.ones((1,1),np.uint8))
    assert result['raw']['passed'] and not result['passed']


def test_equal_post_cannot_hide_raw_difference():
    old=np.array([[[1.,1.,2.,2.,.5,.9]]],np.float32)
    result=benchmark.parity('pothole',old,[],old+1,[])
    assert result['post']['passed'] and not result['passed']


def test_fixed_synthetic_dimensions_and_seed_without_source_files():
    a=benchmark.synthetic_frames();b=benchmark.synthetic_frames()
    assert [x.shape for x in a]==[(720,1280,3),(720,1280,3),(1080,1920,3)]
    assert all(x.dtype==np.uint8 and np.array_equal(x,y) for x,y in zip(a,b))
