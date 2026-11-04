import math
import pytest
from scripts.prepare_mwpd import source_group,select_pairs,parse_yolo,historical_identity,isolate_components


def entry(group,token,suffix='jpg'):
    return {'filename':group+'.rf.'+token*32+'.'+suffix}


def metadata(split,groups):
    images=[entry(g,str(i%10)) for i,g in enumerate(groups)]
    return [{'split':split,'kind':'images','files':images},
            {'split':split,'kind':'labels','files':[{'filename':x['filename'][:-3]+'txt'} for x in images]}]


def test_augmented_filename_groups_never_cross_split():
    rows=metadata('test',['a_jpg'])+metadata('valid',['a_jpg','b_jpg'])+metadata('train',['a_jpg','b_jpg','c_jpg','c_jpg'])
    selected,removed=select_pairs(rows)
    assert [(r['split'],r['group']) for r in selected]==[('test','a_jpg'),('validation','b_jpg'),('train','c_jpg'),('train','c_jpg')]
    assert removed['publisher_cross_split_filename_group']==3


def test_missing_label_is_not_silently_negative():
    rows=metadata('test',['a_jpg'])+metadata('valid',['b_jpg'])+metadata('train',['c_jpg'])
    rows[1]['files']=[]
    with pytest.raises(ValueError,match='pair_missing'):select_pairs(rows)


@pytest.mark.parametrize('name',['../escape.jpg','x.rf.'+'a'*32+'.png','x.rf.bad.jpg'])
def test_unexpected_filename_is_rejected(name):
    with pytest.raises(ValueError):source_group(name)


def test_publisher_box_conversion_and_clipping():
    result=parse_yolo('0 .95 .95 .2 .2\n',100,50)
    assert result[0]['class']=='pothole' and result[0]['certainty']=='confirmed'
    assert result[0]['bbox_xywh_px']==pytest.approx([85,42.5,15,7.5])
    assert parse_yolo('',100,50)==[]


@pytest.mark.parametrize('line',['1 .5 .5 .2 .2','0 nan .5 .2 .2','0 2 .5 .2 .2','0 .5 .5 0 .2','0 .5 .5 .2'])
def test_invalid_class_or_geometry_cannot_enter_training(line):
    with pytest.raises(ValueError):parse_yolo(line,100,50)


def candidate(name,visual,split):
    return {'sha256':name,'perceptual_dhash':f'{visual:016x}','_flip_dhash':visual,'leakage_group_id':name,'split':split}


def test_old_validation_overlap_propagates_through_new_training_chain():
    rows=[candidate('bridge',1,'train'),candidate('chain',15,'train'),candidate('independent',65535,'train')]
    kept,removed=isolate_components(rows,[('historical',0,'validation')])
    assert [r['sha256'] for r in kept]==['independent']
    assert removed['historical_validation_or_test_overlap']==2


def test_new_holdout_is_protected_through_indirect_near_duplicate_chain():
    rows=[candidate('test',0,'test'),candidate('bridge',1,'train'),candidate('chain',15,'train')]
    kept,_=isolate_components(rows,[])
    assert [r['sha256'] for r in kept]==['test']
    assert '_flip_dhash' not in kept[0]


def test_missing_historical_visual_identity_cannot_silently_skip_exact_hash():
    with pytest.raises(ValueError,match='historical_source_identity'):
        historical_identity([{'sha256':'a'*64,'split':'validation'}])
