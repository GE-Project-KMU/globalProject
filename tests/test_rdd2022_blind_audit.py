from scripts.audit_rdd2022_blind import near_components

def row(value,flip=None,positive=True):
    return {'perceptual_dhash':f'{value:016x}','_flip_dhash':value if flip is None else flip,'objects':[{}] if positive else []}

def test_near_chain_and_mixed_annotations_detected_without_model():
    pairs,components,mixed=near_components([row(0),row(7),row(63,positive=False),row(2**64-1)])
    assert pairs==[[0,1],[1,2]] and components==[[0,1,2]] and mixed==1

def test_horizontal_flip_near_match_is_not_missed():
    pairs,components,mixed=near_components([row(0,2**64-1),row(2**64-1)])
    assert pairs==[[0,1]] and components==[[0,1]] and mixed==0

def test_unrelated_images_are_independent_under_declared_hash_rule():
    assert near_components([row(0),row(2**64-1),row(int('aa'*8,16))])==([],[],0)
