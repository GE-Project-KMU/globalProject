"""CPU synthetic exact-view/label/frozen-state contracts; no accuracy evidence."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.refine_privacy_train import view_sample,clip_boxes,state_digest
from agent.paths import atomic_json,sha256
torch.set_num_threads(2)
class Contracts(unittest.TestCase):
    def image(self):
        image=np.zeros((1920,1920,3),np.uint8)
        for index,value in enumerate([10,40,80,160]):image[(index//2)*960:(index//2+1)*960,(index%2)*960:(index%2+1)*960]=value
        return image
    def test_actual_global_scale_and_values(self):
        pixels,labels,count=view_sample(self.image(),[[100,200,40,60]],view='global')
        self.assertEqual(tuple(pixels.shape),(3,960,960));self.assertEqual(count,1)
        self.assertTrue(torch.equal(labels[0],torch.tensor([0,60,115,20,30],dtype=torch.float32)))
        self.assertEqual(float(pixels[0,0,0]),10.);self.assertEqual(float(pixels[0,-1,-1]),160.)
    def test_exact_quadrants_and_clipped_gt(self):
        for index,value in enumerate([10,40,80,160]):
            pixels,labels,count=view_sample(self.image(),[[950,950,40,40]],view='tile',tile=index)
            self.assertTrue(torch.all(pixels==value));self.assertEqual(count,1)
            expected=[[0,955,955,10,10],[0,15,955,30,10],[0,955,15,10,30],[0,15,15,30,30]][index]
            self.assertTrue(torch.equal(labels[0],torch.tensor(expected,dtype=torch.float32)))
    def test_negative_tile_has_zero_targets(self):
        pixels,labels,count=view_sample(self.image(),[[10,10,20,20]],view='tile',tile=3)
        self.assertEqual(count,0);self.assertEqual(int(torch.count_nonzero(labels)),0)
    def test_reject_too_many_targets_without_silent_truncation(self):
        with self.assertRaises(ValueError):view_sample(np.zeros((2,2,3),np.uint8),np.ones((513,4),np.float32))
    def test_positive_intersections_only(self):
        clipped=clip_boxes([[0,0,10,10],[50,50,10,10],[9,9,4,4]],10,10,20,20)
        self.assertTrue(np.array_equal(clipped,np.array([[0,0,3,3]],np.float32)))
    def test_frozen_plate_digest_catches_value_dtype_and_shape(self):
        a={'layer':torch.tensor([1.,2.])};same={'layer':a['layer'].clone()}
        self.assertEqual(state_digest(a),state_digest(same))
        self.assertNotEqual(state_digest(a),state_digest({'layer':torch.tensor([1.,3.])}))
        self.assertNotEqual(state_digest(a),state_digest({'layer':a['layer'].double()}))
        self.assertNotEqual(state_digest(a),state_digest({'layer':a['layer'].view(1,2)}))
    def test_invalid_view_is_rejected(self):
        with self.assertRaises(ValueError):view_sample(np.zeros((2,2,3),np.uint8),[],view='tile',tile=4)
    def test_cpu_only(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    report={'synthetic_contract_tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'accuracy_evidence':False,
        'GPU_initialized':torch.cuda.is_initialized(),'training_code_sha256':sha256(ROOT/'scripts/refine_privacy_train.py')}
    atomic_json(ROOT/'docs/reports/PRIVACY-REFINEMENT-CONTRACT.json',report);print(json.dumps(report));sys.exit(0 if result.wasSuccessful() else 2)
