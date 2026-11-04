import math
import pytest
from scripts.train_pothole_tiny import choose,learning_rate

def row(epoch,old,new):return {'epoch':epoch,'old_validation_ap50':old,'new_validation_ap50':new}

def test_baseline_retained_when_capacity_candidate_fails_development_gate():
    h=[row(0,.66,.59),row(1,.61,.9),row(2,.65,.58)]
    assert choose(h)['epoch']==0

def test_two_domain_mean_selects_improvement_and_earliest_tie():
    assert choose([row(0,.66,.59),row(1,.7,.75),row(2,.7,.75)])['epoch']==1

@pytest.mark.parametrize('history',[[],[row(1,.5,.5)],[row(0,.5,.5),row(2,.6,.6)],[row(0,.5,.5),row(1,float('nan'),.6)],[row(0,.5,.5),row(1,.6,1.1)]])
def test_invalid_selection_history_refused(history):
    with pytest.raises(ValueError):choose(history)

def test_warmup_and_cosine_use_attempts_with_finite_positive_learning_rates():
    rates=[learning_rate(s,1000,100,.0025) for s in range(1000)]
    assert all(math.isfinite(v) and 0<v<=.0025 for v in rates)
    assert rates[99]==.0025 and rates[100]==.0025
    assert rates[0]<rates[50]<rates[99]
    assert rates[100]>rates[500]>rates[999]
    assert rates[999]==pytest.approx(.00025,abs=1e-8)

@pytest.mark.parametrize('args',[(-1,100,10,.1),(100,100,10,.1),(0,100,100,.1),(0,100,0,.1)])
def test_scheduler_refuses_invalid_steps(args):
    with pytest.raises(ValueError):learning_rate(*args)
