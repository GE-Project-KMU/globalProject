import numpy as np
import pytest
from scripts.verify_trained_models import validate_architecture,check_model_output

P2='YOLOX-Nano P2 face640 global plus four tiles and frozen plate640'

def p2():return {'task':'redaction','architecture':P2,'input_size':[1920,1920],'shared_input_size':1920,'face_input_size':640,'plate_input_size':640,'face_tile_grid':2,'classes':['face','license_plate']}

def test_capacity_candidate_has_explicit_pothole_only_geometry():
    info={'task':'pothole','architecture':'YOLOX-Tiny','input_size':[640,640],'classes':['pothole']}
    validate_architecture(info);check_model_output(info,np.zeros((1,8400,6),np.float32))
    with pytest.raises(ValueError):validate_architecture({'task':'redaction','architecture':'YOLOX-Tiny','input_size':[640,640]})
    with pytest.raises(ValueError):validate_architecture({'task':'pothole','architecture':'YOLOX-Tiny','input_size':[416,416]})
    with pytest.raises(ValueError):validate_architecture({'task':'pothole','architecture':'YOLOX-Tiny','input_size':[640.,640]})
    with pytest.raises(ValueError):check_model_output(info,np.zeros((1,3549,6),np.float32))

def test_P2_has_new_exact_contract_without_admitting_legacy_shape():
    info=p2();validate_architecture(info)
    check_model_output(info,np.zeros((1,176400,7),np.float32))
    with pytest.raises(ValueError,match='P2_output_shape'):check_model_output(info,np.zeros((1,102900,7),np.float32))

@pytest.mark.parametrize('key,value',[('shared_input_size',960),('face_input_size',960),('plate_input_size',320),('face_tile_grid',True),('input_size',[960,960])])
def test_P2_mismatched_geometry_refused(key,value):
    info=p2();info[key]=value
    with pytest.raises(ValueError):validate_architecture(info)
