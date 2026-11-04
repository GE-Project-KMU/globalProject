import random
from collections import Counter
import pytest
from scripts.train_pothole_localization import QUOTAS,epoch_indices,validate_hardness

def items():return [{'stratum':key,'weight':value} for key in QUOTAS for value in (1.,3.)]

def test_exact_stratum_counts_not_altered_by_hardness_or_drop_last():
    rows=items();indices=epoch_indices(rows,20261108,0)
    assert len(indices)==4096 and len(indices)%16==0
    assert Counter(rows[i]['stratum'] for i in indices)==QUOTAS
    assert sum(i%2==1 for i in indices)>2800  # Actual weighted replacement emphasizes hard images.

def test_sampling_is_reproducible_but_independent_of_global_augmentation_random():
    rows=items();a=epoch_indices(rows,42,0);random.seed(1337);before=random.getstate();b=epoch_indices(rows,42,0)
    assert a==b and random.getstate()==before
    assert epoch_indices(rows,42,1)!=a

def test_missing_class_stratum_refused():
    with pytest.raises(ValueError):epoch_indices([{'stratum':'MWPD_positive','weight':1.}],42,0)

@pytest.fixture
def full_weights():
    rows=[];weights=[]
    for domain,count,positive in [('original',2676,True),('original',2676,False),('mwpd',2188,True)]:
        for i in range(count):
            sid=f'{domain}-{positive}-{i}';rows.append({'sample_id':sid,'sha256':'a'*64,'split':'train','development_domain':domain,'objects':[{}] if positive else []})
            weights.append({'sample_id':sid,'sha256':'a'*64,'stratum':('RDD' if domain=='original' else 'MWPD')+('_positive' if positive else '_negative'),'weight':2.})
    return rows,{'samples':weights}

def test_all_and_only_training_samples_are_bound(full_weights):
    rows,data=full_weights;assert len(validate_hardness(rows,data))==7540
    rows[0]['split']='test'
    with pytest.raises(ValueError,match='membership'):validate_hardness(rows,data)

@pytest.mark.parametrize('field,value',[('sample_id','foreign-test'),('sha256','b'*64),('stratum','MWPD_positive'),('weight',float('nan')),('weight',3.001),('weight',True)])
def test_tampered_identity_source_stratum_or_weight_refused(full_weights,field,value):
    rows,data=full_weights;data['samples'][0][field]=value
    with pytest.raises(ValueError):validate_hardness(rows,data)
