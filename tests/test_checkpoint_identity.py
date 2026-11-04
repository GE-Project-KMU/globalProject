from scripts.verify_checkpoint_identity import compare_records,inference_contract


def record(raw=b'\x00\x00',shape=(1,),dtype='float16'):
    return {'raw':raw,'shape':shape,'dtype':dtype}


def test_exact_raw_identity_including_nan_payload_bits():
    left={'weight':record(b'\x01\x7e'),'count':record(b'\x01',dtype='uint8')}
    assert compare_records(left,dict(left))['all_state_tensors_exact']
    changed={**left,'weight':record(b'\x02\x7e')}
    result=compare_records(left,changed)
    assert result['raw_byte_mismatches']==1 and not result['all_state_tensors_exact']


def test_key_shape_and_dtype_mismatches_cannot_be_hidden_by_equal_bytes():
    assert not compare_records({'a':record()},{'b':record()})['state_keys_exact']
    result=compare_records({'a':record()},{'a':record(shape=(2,),dtype='uint8')})
    assert result['shape_mismatches']==result['dtype_mismatches']==1
    assert not result['all_state_tensors_exact']


def test_metadata_paths_do_not_change_contract_but_confidence_does():
    old={'task':'pothole','classes':['pothole'],'input_size':[640,640],'confidence':.1,'weights':'a'}
    assert inference_contract(old)==inference_contract({**old,'weights':'b','best_epoch':0})
    assert inference_contract(old)!=inference_contract({**old,'confidence':.35})
