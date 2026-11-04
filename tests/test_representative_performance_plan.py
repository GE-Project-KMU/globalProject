"""Synthetic case metadata selection only; no model/image/GPU or score access."""
import pytest
from scripts.prepare_representative_performance import select_representative,GROUPS


def cases():
    plan={'models':{}};rows={};canonical={}
    for task in ('pothole','lane','redaction'):
        labels=['face','license_plate'] if task=='redaction' else [task]
        pool=[]
        for label in labels:
            for index in range(20):
                pool.append({'sample_id':str(index),'source_id':str(20-index).zfill(3),'sha256':label+'-'+str(index),'path':'SYNTHETIC/'+label+'/'+str(index),
                    'split':'validation','expires_at':'2026-11-10T00:00:00Z','evaluation_class':label})
        rows[task]=pool;plan['models'][task]={'cases':pool}
        canonical[task]={label+'-19' for label in labels}
    return plan,rows,canonical


def test_canonical_sorted_group_balance_roundrobin_is_deterministic():
    plan,rows,canonical=cases();selected=select_representative(plan,rows,canonical)
    assert len(selected)==32 and [row['group'] for row in selected]==list(GROUPS)*8
    assert all(row['sample_id']=='19' for row in selected[:4])
    assert selected==select_representative(plan,{key:list(reversed(value)) for key,value in rows.items()},canonical)


def test_test_rows_outside_frozen160_never_selected():
    plan,rows,canonical=cases();rows['pothole'].append(rows['pothole'][0]|{'sha256':'outside','source_id':'000','split':'test'})
    assert all(row['sha256']!='outside' for row in select_representative(plan,rows,canonical))


def test_missing_canonical_or_group_population_refused():
    plan,rows,canonical=cases();canonical['lane'].add('missing')
    with pytest.raises(ValueError,match='canonical_omitted'):select_representative(plan,rows,canonical)
    plan,rows,canonical=cases();rows['lane']=rows['lane'][:7]
    with pytest.raises(ValueError,match='eight_frozen'):select_representative(plan,rows,canonical)
