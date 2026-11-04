"""Actual CPU-only architecture/loss contracts on synthetic tensors, not accuracy."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,sha256

def run():
 import numpy as np,torch,onnxruntime as ort
 from torch import nn
 from agent.redaction_p2_network import build_redaction_p2_network,assemble_p2_network,load_p2_warmstart
 from scripts.refine_privacy_train import state_digest
 from scripts.train_lane_cycle2 import paint_loss
 torch.set_num_threads(2);torch.manual_seed(20261106)
 cases=[]
 def check(name,ok):
  if not bool(ok):raise ValueError('contract_failed:'+name)
  cases.append(name)
 class Boxes(nn.Module):
  def forward(self,x):return torch.tensor([[[10.,20.,30.,40.,.8,.9]]],dtype=x.dtype).expand(x.shape[0],-1,-1)
 mock=assemble_p2_network(Boxes(),Boxes()).eval();coords=mock(torch.zeros(1,3,1920,1920))
 check('five_face_one_plate_geometry_and_class_columns',torch.equal(coords[0,:,:4],torch.tensor([[30.,60.,90.,120.],[15.,30.,45.,60.],[975.,30.,45.,60.],[15.,990.,45.,60.],[975.,990.,45.,60.],[30.,60.,90.,120.]])) and torch.count_nonzero(coords[0,:5,6])==0 and coords[0,5,5]==0)
 from scripts.train_privacy_p2_cycle2 import view_sample
 pattern=np.random.default_rng(20261106).integers(0,256,(1920,1920,3),dtype=np.uint8)
 boxes=np.asarray([[950.,950.,40.,40.]],np.float32)
 global_pixels,global_labels,count=view_sample(pattern,boxes,view='global')
 expected=torch.nn.functional.interpolate(torch.from_numpy(pattern.transpose(2,0,1).copy().astype(np.float32))[None],size=(640,640),mode='bilinear',align_corners=False)[0]
 check('training_global_single_bilinear_pixels_exact_and_GT_coordinates',torch.equal(global_pixels,expected) and count==1 and torch.allclose(global_labels[0,1:],torch.tensor([970/3,970/3,40/3,40/3])))
 tile_pixels,tile_labels,count=view_sample(pattern,boxes,view='tile',tile=3)
 expected=torch.nn.functional.interpolate(torch.from_numpy(pattern[960:,960:].transpose(2,0,1).copy().astype(np.float32))[None],size=(640,640),mode='bilinear',align_corners=False)[0]
 check('training_tile_single_bilinear_pixels_exact_and_intersecting_GT_kept',torch.equal(tile_pixels,expected) and count==1 and torch.equal(tile_labels[0,1:],torch.tensor([10.,10.,20.,20.])))
 rejected=False
 try:view_sample(pattern,np.tile(boxes,(513,1)),view='global')
 except ValueError:rejected=True
 check('over512_GT_rejected_before_canvas_no_silent_truncation',rejected)
 bundle=build_redaction_p2_network();state=torch.load(ROOT/'data/training/redaction-specialists-v4-efficient/best.pt',map_location='cpu',weights_only=True)['model'];audit=load_p2_warmstart(bundle,state)
 oldplate={k.removeprefix('plate.'):v for k,v in state.items() if k.startswith('plate.')}
 check('plate_all_parameters_and_buffers_strict_byte_digest',state_digest(bundle.plate.state_dict())==state_digest(oldplate))
 check('new_P2_stem_and_neck_are_explicitly_new_not_silent_partial_load',audit['initialized_tensor_count']>0 and audit['assigned_tensor_count']>0 and audit['full_model_state_identical_claim'] is False)
 bundle.face.train();images=torch.rand(2,3,128,128)*255;labels=torch.zeros(2,512,5);labels[0,0]=torch.tensor([0,60,60,4,4]);labels[1,0]=torch.tensor([0,64,64,64,64]);loss=bundle.face(images,labels)['total_loss'];loss.backward()
 check('actual_P2_tiny_and_large_GT_loss_backward_finite',torch.isfinite(loss) and any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad!=0).any()) for p in bundle.face.parameters()))
 # Reload the pinned baseline transplantation after synthetic backward changes BN buffers.
 bundle=build_redaction_p2_network();audit=load_p2_warmstart(bundle,state);bundle.eval();pixels=torch.rand(1,3,1920,1920)*255
 with torch.no_grad():raw=bundle(pixels).numpy()
 check('actual_shared1920_P2_640_output_shape_finite_bounded_memory',raw.shape==(1,176400,7) and np.isfinite(raw).all() and raw.nbytes<64*1024**2)
 out=ROOT/'data/training/public-nextcycle/contracts';out.mkdir(parents=True,exist_ok=True)
 onnx=out/'p2-synthetic-contract.onnx';torch.onnx.export(bundle,pixels,str(onnx),opset_version=17,input_names=['images'],output_names=['detections'],dynamo=False)
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
 session=ort.InferenceSession(str(onnx),sess_options=opts,providers=['CPUExecutionProvider']);converted=session.run(None,{'images':pixels.numpy()})[0]
 check('P2_actual_ONNX_export_Torch_ORT_default_raw_tolerance',np.isfinite(converted).all() and np.allclose(raw,converted,rtol=.001,atol=.0001))
 logits=torch.randn(2,2,8,8,requires_grad=True);labels=torch.full((2,8,8),255,dtype=torch.long);ignored=paint_loss(logits,labels);ignored.backward()
 check('lane_all_ignore_zero_loss_and_zero_gradient',float(ignored)==0 and torch.count_nonzero(logits.grad)==0)
 logits=torch.randn(2,2,8,8,requires_grad=True);labels=torch.zeros((2,8,8),dtype=torch.long);labels[:,2:4,2:4]=1;labels[:,7]=255;loss=paint_loss(logits,labels);loss.backward()
 check('lane_hardCE_Dice_boundary_finite_and_ignored_gradient_zero',torch.isfinite(loss) and torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad[:,:,7])==0)
 report={'format':'public-cycle2-CPU-contract-v1','synthetic_only_not_accuracy':True,'GPU_operations':0,'cases':cases,'passed':True,'torch':torch.__version__,'ORT':ort.__version__,'P2_output_shape':list(raw.shape),'P2_output_bytes':raw.nbytes,'P2_transferred_tensor_count':audit['assigned_tensor_count'],'P2_initialized_tensor_count':audit['initialized_tensor_count'],'raw_rtol':.001,'raw_atol':.0001,'factory_sha256':sha256(ROOT/'agent/redaction_p2_network.py'),'contract_source_sha256':sha256(Path(__file__)),'lane_trainer_sha256':sha256(ROOT/'scripts/train_lane_cycle2.py'),'privacy_trainer_sha256':sha256(ROOT/'scripts/train_privacy_p2_cycle2.py')}
 atomic_json(out/'aggregate.json',report);atomic_json(ROOT/'docs/reports/CYCLE2-CPU-CONTRACT.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':run()
