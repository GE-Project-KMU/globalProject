import hashlib,json
from datetime import datetime,timedelta,timezone
import pytest
from scripts.prepare_rdd2022_us_offline import OfflineRange,component_representatives

def folder(tmp_path):
    (tmp_path/'source.json').write_text(json.dumps({'expires_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()}),encoding='utf-8')
    (tmp_path/'range-cache').mkdir()
    return tmp_path

def test_offline_reader_requires_exact_proven_cache_and_has_no_network_client(tmp_path):
    path=folder(tmp_path);cache=path/'range-cache/0-3.bin';cache.write_bytes(b'0123')
    cache.with_suffix('.sha256').write_text(hashlib.sha256(b'0123').hexdigest(),encoding='utf-8')
    with OfflineRange(folder=path,block=4) as reader:
        assert reader.fetch(0,3)==b'0123' and not hasattr(reader,'client')
        assert reader.network_bytes==reader.range_responses==0
        with pytest.raises(ValueError,match='offline_range_missing'):reader.fetch(4,7)
        cache.write_bytes(b'bad!')
        with pytest.raises(ValueError,match='integrity_failure'):reader.fetch(0,3)

def test_original_expiry_blocks_even_cached_bytes(tmp_path):
    path=folder(tmp_path)
    (path/'source.json').write_text(json.dumps({'expires_at':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()}),encoding='utf-8')
    with OfflineRange(folder=path) as reader:
        with pytest.raises(ValueError,match='source_expired'):reader.read(1)

def sample(identifier,value,positive=True):
    return {'sample_id':identifier,'perceptual_dhash':f'{value:016x}','_flip_dhash':value,'objects':[{}] if positive else []}

def test_single_component_keeps_smallest_uuid_and_mixed_component_is_omitted():
    rows=[sample('b',0),sample('a',7),sample('c',63),sample('d',2**64-1),sample('e',2**64-2,False)]
    kept,audit=component_representatives(rows)
    assert [r['sample_id'] for r in kept]==['a']
    assert audit['mixed_components']==1
    assert audit['removed']=={'same_label_component_redundancy':2,'mixed_label_component':2}
