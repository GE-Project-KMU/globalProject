"""Synthetic split isolation and reconstructible architecture policy checks."""
import importlib,sys
import pytest
from scripts.public_refinement_data import novel_components,audit_development
from scripts.verify_trained_models import validate_architecture

def image(name,visual):return {'source_hash':name,'visual':f'{visual:016x}'}
def test_connected_near_duplicate_chain_to_history_is_excluded():
    old=[image('old',0)];new=[image('bridge',1),image('chain',3),image('independent',255)]
    kept=novel_components(old,new,'source_hash','visual',1)
    assert [r['source_hash'] for r in kept]==['independent']
def test_exact_hash_exclusion_even_if_perceptual_hash_differs():
    old=[image('same',0)];new=[image('same',2**64-1)]
    assert novel_components(old,new,'source_hash','visual',0)==[]
def test_test_rows_cannot_enter_development():
    with pytest.raises(ValueError,match='test_in_development'):audit_development([{'split':'test'}],[])
@pytest.mark.parametrize('field,value',[('source_id','old'),('source_hash','oldhash')])
def test_historical_test_identity_or_hash_never_fit(field,value):
    history=[{'split':'test','source_id':'old','source_hash':'oldhash'}];current={'split':'train','source_id':'new','source_hash':'newhash','path':'data/public/a.png'};current[field]=value
    with pytest.raises(ValueError,match='historical_test'):audit_development([current],history)
def test_user_recordings_are_forbidden_even_if_in_project():
    current={'split':'train','source_id':'new','source_hash':'newhash','path':'데이터/private.svo2'}
    with pytest.raises(ValueError,match='user_recording'):audit_development([current],[])
def test_new_architecture_is_explicit_not_arbitrary():
    validate_architecture({'task':'lane','architecture':'deeplabv3_mobilenet_v3_large_stride4_residual_refinement','aspp_dilations':[6,12,18]})
    with pytest.raises(ValueError,match='unsupported_lane'):validate_architecture({'task':'lane','architecture':'unregistered'})
