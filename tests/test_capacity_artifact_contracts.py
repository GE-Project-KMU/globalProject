"""New candidate dispatch and development-only provenance stay explicit."""
import json
import numpy as np
import pytest
from scripts.verify_trained_models import validate_architecture,check_model_output
from scripts.train_pothole_small import parameter_groups

def test_small_exact_geometry_and_output():
    info={'task':'pothole','architecture':'YOLOX-S','input_size':[640,640],'classes':['pothole']}
    validate_architecture(info);check_model_output(info,np.zeros((1,8400,6),np.float32))
    with pytest.raises(ValueError):validate_architecture(info|{'input_size':[960,960]})
    with pytest.raises(ValueError):check_model_output(info,np.zeros((1,100,6),np.float32))

def test_rescue_exact_geometry_and_ordered_row_contract():
    info={'task':'redaction','architecture':'frozen legacy YOLOX-Nano specialists plus four tile P2 rescue','input_size':[1920,1920],'shared_input_size':1920,'face_input_size':960,'plate_input_size':640,'face_tile_grid':2,'classes':['face','license_plate']}
    validate_architecture(info);check_model_output(info,np.zeros((1,333300,7),np.float32))
    with pytest.raises(ValueError):validate_architecture(info|{'face_input_size':640})
    with pytest.raises(ValueError):check_model_output(info,np.zeros((1,102900,7),np.float32))

def test_grouping_exactly_once_without_bn_or_bias_decay():
    class P:
        def __init__(self,ndim):self.ndim=ndim
    items=[('backbone.conv.weight',P(4)),('backbone.bn.weight',P(1)),('backbone.conv.bias',P(1)),('head.conv.weight',P(4)),('head.bn.bias',P(1))]
    class Model:
        def named_parameters(self):return iter(items)
    groups=parameter_groups(Model());flat=[p for g in groups for p in g['params']]
    assert len(flat)==len(items)==len({id(p) for p in flat})
    for name,p in items:
        group=next(g for g in groups if any(v is p for v in g['params']))
        assert group['base_lr']==(.002 if name.startswith('head.') else .0002)
        assert group['weight_decay']==(.0005 if p.ndim>1 else 0.)
