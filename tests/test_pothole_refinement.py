"""Validation-only candidate selection must retain a better baseline."""
import pytest
from scripts.refine_pothole_validation import choose

def test_initial_baseline_retained_when_refinement_regresses_or_ties():
    history=[{'epoch':0,'validation_ap50':.7},{'epoch':1,'validation_ap50':.6},{'epoch':2,'validation_ap50':.7}]
    assert choose(history)['epoch']==0

def test_only_validation_improvement_can_replace_baseline():
    history=[{'epoch':0,'validation_ap50':.7,'test_ap50':1.},{'epoch':1,'validation_ap50':.72,'test_ap50':0.}]
    assert choose(history)['epoch']==1

@pytest.mark.parametrize('history',[
    [],[{'epoch':1,'validation_ap50':.8}],
    [{'epoch':0,'validation_ap50':float('nan')}],
    [{'epoch':0,'validation_ap50':float('inf')}],
    [{'epoch':0,'validation_ap50':True}],
    [{'epoch':0,'validation_ap50':1.1}],
    [{'epoch':0,'validation_ap50':.5},{'epoch':2,'validation_ap50':.8}],
    [{'epoch':False,'validation_ap50':.8}],
])
def test_invalid_or_unordered_validation_cannot_select_checkpoint(history):
    with pytest.raises(ValueError):choose(history)
