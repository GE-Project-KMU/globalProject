"""Data-free optional tiled architecture contract; no model accuracy claim."""
import pytest
from agent.redaction_tiled_network import inference_passes

def test_tiled_plan_covers_each_quadrant_once_and_preserves_global_branches():
    passes=inference_passes()
    assert [p['head'] for p in passes]==['face']*5+['plate']
    assert passes[0]=={'head':'face','crop':None,'resize':960,'coordinate_scale':2.,'offset':(0.,0.)}
    assert passes[-1]=={'head':'plate','crop':None,'resize':640,'coordinate_scale':3.,'offset':(0.,0.)}
    quadrants=passes[1:5]
    assert {p['crop'] for p in quadrants}=={(0,0,960,960),(960,0,1920,960),(0,960,960,1920),(960,960,1920,1920)}
    for p in quadrants:
        x0,y0,x1,y1=p['crop']
        assert (x1-x0,y1-y0)==(960,960) and p['offset']==(x0,y0)
        assert p['resize'] is None and p['coordinate_scale']==1.

@pytest.mark.parametrize('patch',[{'shared_input_size':960},{'shared_input_size':2048},{'face_input_size':640},{'plate_input_size':960},{'shared_input_size':1920.0},{'face_input_size':True}])
def test_no_unfrozen_geometry_fallback(patch):
    with pytest.raises(ValueError,match='unsupported_frozen_tiled_geometry'):inference_passes(**patch)
