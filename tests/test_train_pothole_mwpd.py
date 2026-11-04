import pytest
from scripts.train_pothole_mwpd import choose

def row(epoch,old,new):return {'epoch':epoch,'old_validation_ap50':old,'new_validation_ap50':new}

def test_new_domain_cannot_hide_old_validation_regression():
    history=[row(0,.68,.2),row(1,.60,.9),row(2,.66,.4)]
    assert choose(history)['epoch']==2

def test_initial_baseline_and_earliest_tie_are_preserved():
    assert choose([row(0,.68,.2),row(1,.68,.2)])['epoch']==0
    assert choose([row(0,.68,.2),row(1,.68,.19)])['epoch']==0

@pytest.mark.parametrize('history',[[],[row(1,.5,.5)],[row(0,float('nan'),.5)],[row(0,.5,.5),row(3,.5,.6)]])
def test_invalid_validation_history_cannot_select_a_checkpoint(history):
    with pytest.raises(ValueError):choose(history)
