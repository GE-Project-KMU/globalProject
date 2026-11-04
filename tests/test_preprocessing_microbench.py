import numpy as np
import pytest
from scripts.benchmark_preprocessing import baseline,candidate,exact,patterns,paired

@pytest.mark.parametrize('task',['pothole','redaction','lane'])
def test_all_uint8_scalar_values_bitwise_exact(task):
    for value in range(256):
        frame=np.full((7,9,3),value,np.uint8)
        assert exact(baseline(frame,[32,32],task),candidate(frame,[32,32],task))

@pytest.mark.parametrize('task',['pothole','redaction','lane'])
@pytest.mark.parametrize('shape',[(48,64),(49,65)])
def test_patterns_and_noncontiguous_source_have_identical_bits(task,shape):
    for frame in patterns(*shape):
        for image in (frame,frame[:,::-1]):
            before=image.tobytes()
            old=baseline(image,[96,96],task);new=candidate(image,[96,96],task)
            assert exact(old,new) and new.flags.c_contiguous
            assert image.tobytes()==before

def test_lane_custom_normalization_order_and_invalid_std():
    frame=next(patterns(17,23))
    assert exact(baseline(frame,[64,64],'lane',[.12,.34,.56],[.23,.45,.67]),candidate(frame,[64,64],'lane',[.12,.34,.56],[.23,.45,.67]))
    with pytest.raises(ValueError):candidate(frame,[64,64],'lane',std=[0.,1.,1.])

def test_paired_contract_requires_more_than_one_block():
    with pytest.raises(ValueError):paired(np.zeros((4,4,3),np.uint8),[32,32],'pothole',blocks=1)

def test_baseline_matches_actual_runtime_preprocessing_without_loading_weights():
    from agent.models import OnnxModel
    for task in ('pothole','lane'):
        class Capture:
            blob=None
            def setInput(self,value):
                assert value.dtype==np.float32 and value.flags.c_contiguous
                assert np.ascontiguousarray(value) is value
                self.blob=value
            def forward(self):return np.zeros((1,2,96,96),np.float32) if task=='lane' else np.zeros((1,0,6),np.float32)
        model=object.__new__(OnnxModel);model.manifest={'task':task,'classes':['pothole'],'input_size':[96,96],'output_layout':'yolox_decoded_xywh'};model.net=Capture()
        image=next(patterns(49,65));model.predict(image)
        assert exact(model.net.blob,baseline(image,[96,96],task))

@pytest.mark.parametrize('task,size',[('pothole',[640,640]),('redaction',[1920,1920]),('lane',[1024,1024])])
def test_applied_runtime_all_patterns_and_uint8_values_before_any_network_forward(task,size):
    from agent.models import OnnxModel
    class InputCaptured(Exception):pass
    class Capture:
        blob=None
        def setInput(self,value):
            assert value.dtype==np.float32 and value.flags.c_contiguous
            assert np.ascontiguousarray(value) is value
            self.blob=value
            raise InputCaptured
        def forward(self):pytest.fail('network inference is forbidden')
    model=object.__new__(OnnxModel);model.manifest={'task':task,'input_size':size};model.net=Capture()
    def check(frame,target):
        model.manifest['input_size']=target
        original=frame.tobytes()
        with pytest.raises(InputCaptured):model.predict(frame)
        assert exact(model.net.blob,baseline(frame,target,task))
        assert frame.tobytes()==original
        model.net.blob=None
    for value in range(256):check(np.full((7,9,3),value,np.uint8),[32,32])
    for shape in ((48,64),(49,65),(720,1280),(1080,1920)):
        for frame in patterns(*shape):check(frame,size)
