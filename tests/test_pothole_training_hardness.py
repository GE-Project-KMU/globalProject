import copy,math
import pytest
from scripts.mine_pothole_training_hardness import hardness,membership,assert_train_separation

@pytest.mark.parametrize('gt,fn,fp,expected',[(1,0,0,1),(1,1,0,1.5),(3,3,0,2),(1,1,10,3),(0,0,0,1),(0,0,1,1+2/3),(0,0,3,3),(0,0,100,3)])
def test_frozen_positive_and_negative_formula(gt,fn,fp,expected):
    assert hardness(gt,fn,fp)==pytest.approx(expected)

@pytest.mark.parametrize('args',[(-1,0,0),(1,2,0),(1,-1,0),(1,0,-1),(1.,0,0),(True,0,0)])
def test_invalid_matching_counts_are_not_coerced(args):
    with pytest.raises(ValueError):hardness(*args)

def row(value,split='train'):
    return {'sample_id':value,'sha256':value*64,'source_hash':value*64,'leakage_group_id':'group-'+value,'split':split,'hardness_stratum':'RDD_positive'}

def test_membership_digest_is_order_independent_but_source_identity_and_stratum_bound():
    rows=[row('a'),row('b')];digest=membership(rows)
    assert membership(list(reversed(rows)))==digest
    changed=copy.deepcopy(rows);changed[0]['hardness_stratum']='MWPD_positive'
    assert membership(changed)!=digest
    changed=copy.deepcopy(rows);changed[0]['sha256']='c'*64
    assert membership(changed)!=digest

def test_VAL_TEST_rows_never_enter_training_membership_even_with_same_image_metadata():
    for split in ['validation','test']:
        with pytest.raises(ValueError,match='TRAIN_only'):membership([row('a',split)])
        with pytest.raises(ValueError,match='TRAIN_only'):assert_train_separation([row('a',split)],[])

@pytest.mark.parametrize('field',['sample_id','sha256','source_hash','leakage_group_id'])
def test_historical_VAL_or_TEST_identity_overlap_is_rejected(field):
    train=row('a');heldout=row('b','test');heldout[field]=train[field]
    with pytest.raises(ValueError,match='identity_overlap'):assert_train_separation([train],[heldout])

def test_separate_training_and_protected_identities_are_valid():
    assert_train_separation([row('a')],[row('b','validation'),row('c','test')])
