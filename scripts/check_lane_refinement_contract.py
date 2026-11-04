"""CPU synthetic structural/loss contracts; not model accuracy evidence."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from agent.lane_network import make_lane_network,freeze_batch_norm_statistics
from agent.lane_refinement_network import make_lane_refinement_network,load_legacy_warmstart
from scripts.refine_lane_train import paint_loss
from agent.paths import atomic_json,sha256
torch.set_num_threads(2)

class Contracts(unittest.TestCase):
    def test_zero_residual_preserves_baseline(self):
        torch.manual_seed(7);base=make_lane_network(pretrained=False,aspp_dilations=[6,12,18]).eval();candidate=make_lane_refinement_network().eval();load_legacy_warmstart(candidate,base.state_dict())
        for size in [(128,160),(160,128)]:
            x=torch.randn(1,3,*size)
            with torch.inference_mode():a=base(x)['out'];b=candidate(x)['out']
            self.assertTrue(torch.equal(a,b));self.assertEqual(tuple(b.shape),(1,2,*size))
        self.assertEqual(sum(p.numel() for p in candidate.parameters())-sum(p.numel() for p in base.parameters()),20114)
    def test_warmstart_rejects_missing_state(self):
        model=make_lane_refinement_network()
        with self.assertRaises(RuntimeError):load_legacy_warmstart(model,{})
    def test_refinement_receives_actual_gradient(self):
        model=make_lane_refinement_network().train();freeze_batch_norm_statistics(model);x=torch.randn(2,3,128,128);y=torch.zeros(2,128,128,dtype=torch.long);y[:,:,60:63]=1
        loss=paint_loss(model(x)['out'],y);loss.backward();g=model.refinement[-1].weight.grad
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0.)
    def test_all_ignore_is_finite_zero_with_zero_gradient(self):
        logits=torch.randn(1,2,12,12,requires_grad=True);labels=torch.full((1,12,12),255,dtype=torch.long);loss=paint_loss(logits,labels);loss.backward()
        self.assertEqual(float(loss),0.);self.assertEqual(float(logits.grad.abs().sum()),0.)
    def test_perfect_thin_paint_scores_better_than_miss(self):
        labels=torch.zeros(1,20,20,dtype=torch.long);labels[:,:,9:11]=1
        perfect=torch.stack([torch.where(labels==0,torch.tensor(12.),torch.tensor(-12.)),torch.where(labels==1,torch.tensor(12.),torch.tensor(-12.))],1)
        missed=torch.full_like(perfect,-12.);missed[:,0]=12.
        self.assertLess(float(paint_loss(perfect,labels)),.001);self.assertGreater(float(paint_loss(missed,labels)),.5)
    def test_ignored_region_cannot_change_loss(self):
        labels=torch.zeros(1,20,20,dtype=torch.long);labels[:,:,9:11]=1;labels[:,:5,:]=255
        a=torch.randn(1,2,20,20);b=a.clone();b[:,:,:5,:]=100*torch.randn_like(b[:,:,:5,:])
        self.assertTrue(torch.equal(paint_loss(a,labels),paint_loss(b,labels)))
    def test_no_GPU_initialized(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    report={'synthetic_contract_tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'accuracy_evidence':False,
        'GPU_initialized':torch.cuda.is_initialized(),'factory_sha256':sha256(ROOT/'agent/lane_refinement_network.py'),'trainer_sha256':sha256(ROOT/'scripts/refine_lane_train.py')}
    atomic_json(ROOT/'docs/reports/LANE-REFINEMENT-CONTRACT.json',report);print(json.dumps(report));sys.exit(0 if result.wasSuccessful() else 2)
