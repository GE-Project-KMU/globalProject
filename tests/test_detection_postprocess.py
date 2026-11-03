"""Frozen old-loop differential oracle. Synthetic only; stdlib runnable on1.x/2.x."""
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import cv2
from agent.models import OnnxModel,detection_candidates

def old_scalar_candidates(predictions,scale,threshold):
    boxes,scores,labels=[],[],[]
    for row in predictions:
        label=int(np.argmax(row[5:]));confidence=float(row[4]*row[5+label])
        if confidence<threshold:continue
        cx,cy,bw,bh=row[:4]/scale
        if bw<=0 or bh<=0:continue
        boxes.append([float(cx-bw/2),float(cy-bh/2),float(bw),float(bh)])
        scores.append(confidence);labels.append(label)
    return boxes,scores,labels

def old_scalar_result(predictions,scale,threshold,width,height):
    boxes,scores,labels=old_scalar_candidates(predictions,scale,threshold);results=[]
    for label in set(labels):
        candidates=[i for i,value in enumerate(labels) if value==label]
        selected=cv2.dnn.NMSBoxes([boxes[i] for i in candidates],[scores[i] for i in candidates],threshold,.5)
        for local in np.asarray(selected).reshape(-1):
            index=candidates[int(local)];x,y,bw,bh=boxes[index]
            left,top=max(0.,x),max(0.,y);right,bottom=min(width,x+bw),min(height,y+bh)
            if right<=left or bottom<=top:continue
            results.append({'bbox_xywh_px':[left,top,right-left,bottom-top],'confidence':scores[index],'class':['face','license_plate'][label]})
    return results

def app_result(raw,threshold=.35,width=1937,height=1081,size=1920):
    model=OnnxModel.__new__(OnnxModel)
    model.manifest={'task':'redaction','classes':['face','license_plate'],'input_size':[size,size],'output_layout':'yolox_decoded_xywh','confidence':threshold,'nms_iou':.5}
    class FakeNet:
        def setInput(self,value):pass
        def forward(self):return raw[None]
    model.net=FakeNet()
    return model.predict(np.zeros((height,width,3),np.uint8))

def edge_rows(dtype):
    return np.array([
        [10,10,8,8,.35,1,0],
        [12,12,8,8,np.nextafter(dtype(.35),dtype(np.inf)),1,0],
        [14,14,8,8,np.nextafter(dtype(.35),dtype(-np.inf)),1,0],
        [-5,20,30,10,.9,.7,.7],
        [1925,1085,30,30,.9,.1,.9],
        [-200,10,5,5,.9,.9,.1],
        [20,20,0,10,.9,.9,.1],
        [20,20,-1,10,.9,.9,.1],
        [20,20,10,-0.,.9,.9,.1],
        [100.12345,200.54321,89.98765,32.12345,.9,.9,.1],
        [100.12345,200.54321,89.98765,32.12345,.9,.9,.1],
        [40,40,8,8,.5,1,0],
    ],dtype)

class DetectionPostprocessTests(unittest.TestCase):
    def test_exact_candidate_differential_float16_32_64(self):
        rng=np.random.default_rng(20261104)
        for dtype in (np.float16,np.float32,np.float64):
            random=rng.uniform(0,1,(1000,7)).astype(dtype);random[:,:4]*=dtype(100)
            edges=edge_rows(dtype);empty=edges.copy();empty[:,4]=0
            for rows in (random,edges,empty,np.empty((0,7),dtype)):
                for scale in (1.,1.5,1920/1937,1920/257,1920/4096):
                    for threshold in (.35,.5,float(dtype(.35))):
                        with self.subTest(dtype=dtype,shape=rows.shape,scale=scale,threshold=threshold):
                            self.assertEqual(old_scalar_candidates(rows,scale,threshold),detection_candidates(rows,scale,threshold))

    def test_actual_app_matches_old_loop_after_nms_and_clipping(self):
        for dtype in (np.float16,np.float32,np.float64):
            rows=edge_rows(dtype)
            for threshold in (.35,.5,float(dtype(.35))):
                with self.subTest(dtype=dtype,threshold=threshold):
                    expected=old_scalar_result(rows,1920/1937,threshold,1937,1081)
                    self.assertEqual(expected,app_result(rows,threshold))

    def test_large_tiled_raw_has_exactly200_candidates_and_same_app_output(self):
        rng=np.random.default_rng(20261104);rows=np.zeros((102900,7),np.float32)
        rows[:,:2]=rng.uniform(-200,2120,(len(rows),2));rows[:,2:4]=rng.uniform(1,300,(len(rows),2))
        rows[:,4]=1e-6;rows[:,5:]=rng.uniform(.01,.9,(len(rows),2))
        selected=np.sort(rng.choice(len(rows),200,replace=False));rows[selected,4]=.95
        rows[selected,5:]=.01;rows[selected,5+(np.arange(200)%2)]=.9
        expected=old_scalar_candidates(rows,1920/1937,.35)
        self.assertEqual(len(expected[0]),200)
        self.assertEqual(expected,detection_candidates(rows,1920/1937,.35))
        self.assertEqual(old_scalar_result(rows,1920/1937,.35,1937,1081),app_result(rows))

    def test_classwise_nms_precedes_source_clipping(self):
        rows=np.array([[-100,50,400,100,.95,1,0],[50,50,100,100,.9,1,0],[50,50,100,100,.85,0,1]],np.float32)
        actual=app_result(rows,width=128,height=128,size=128)
        self.assertEqual(actual,old_scalar_result(rows,1.,.35,128,128))
        self.assertEqual(len(actual),3)
        self.assertEqual([v['class'] for v in actual].count('face'),2)
        self.assertTrue(all(v['bbox_xywh_px']==[0.,0.,100.,100.] for v in actual))

    def test_python_threshold_tie_and_first_class_argmax_are_preserved(self):
        low=np.float32(.35);high=np.nextafter(low,np.float32(np.inf))
        rows=np.array([[10,10,8,8,low,1,1],[20,20,8,8,high,1,1]],np.float32)
        boxes,scores,labels=detection_candidates(rows,1.,.35)
        self.assertEqual(len(boxes),1);self.assertEqual(scores,[float(high)]);self.assertEqual(labels,[0])

    def test_empty_and_nonpositive_candidates_remain_empty(self):
        for dtype in (np.float16,np.float32,np.float64):
            for rows in (np.empty((0,7),dtype),np.array([[10,10,0,8,.9,1,0],[10,10,8,-1,.9,1,0],[10,10,8,8,0,1,0]],dtype)):
                self.assertEqual(detection_candidates(rows,1.,.35),([],[],[]))
                self.assertEqual(app_result(rows,width=16,height=16,size=16),[])

if __name__=='__main__':unittest.main()
