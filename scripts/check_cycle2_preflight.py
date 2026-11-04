"""CPU-only real-development geometry/loss smoke; never reads TEST pixels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agent.paths import atomic_json,workspace_path,sha256

def run(kind):
 import cv2,numpy as np,torch
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(20261106)
 if kind=='privacy':
  from agent.redaction_p2_network import build_redaction_p2_network,load_p2_warmstart
  from scripts.train_privacy_p2_cycle2 import DEV,BASE,view_sample
  rows=[r for r in json.loads(DEV.read_text()) if r['split']=='train' and r['evaluation_class']=='face'];chosen=sorted(rows,key=lambda r:(-len(r['objects']),r['source_id']))[:4]
  images=[];labels=[];counts=[]
  for i,r in enumerate(chosen):
   image=cv2.imread(str(workspace_path(r['path'],exists=True)));boxes=np.asarray([o['bbox_xywh_px'] for o in r['objects']],np.float32);x,y,count=view_sample(image,boxes,view='global' if i%2==0 else 'tile',tile=i%4);images.append(x);labels.append(y);counts.append(count)
  bundle=build_redaction_p2_network();transfer=load_p2_warmstart(bundle,torch.load(BASE,map_location='cpu',weights_only=True)['model']);bundle.face.train();loss=bundle.face(torch.stack(images),torch.stack(labels))['total_loss'];loss.backward()
  if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in bundle.face.parameters()):raise ValueError('real_development_loss_gradient_nonfinite')
  report={'format':'cycle2-real-development-CPU-preflight-v1','task':kind,'actual_development_images':4,'max_original_GT_count':max(len(r['objects']) for r in chosen),'retained_view_GT_counts':counts,'tensor_shape':list(torch.stack(images).shape),'loss':float(loss.detach()),'passed':True,'GPU_operations':0,'TEST_images_loaded':0,'user_recordings_used':False,'source_expiries_extended':False,'expiry':min(r['expires_at'] for r in rows),'code_sha256':sha256(Path(__file__)),'trainer_sha256':sha256(ROOT/('scripts/train_lane_cycle2.py' if kind=='lane' else 'scripts/train_privacy_p2_cycle2.py')),'factory_sha256':sha256(ROOT/'agent/redaction_p2_network.py'),'limits':['Geometry/loss/backward smoke, not accuracy or generalization']}
 else:
  from scripts.train_lane_cycle2 import DEV,BASE,paint_loss,full_context_sample
  from agent.lane_refinement_network import make_lane_refinement_network
  from agent.lane_network import freeze_batch_norm_statistics
  rows=[r for r in json.loads(DEV.read_text()) if r['split']=='train'];chosen=max(rows,key=lambda r:r.get('positive_pixels',0)/max(1,r.get('valid_pixels',1)));x,y=full_context_sample(chosen);net=make_lane_refinement_network();net.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=True),strict=True);net.train();freeze_batch_norm_statistics(net);out=net(torch.from_numpy(x[None]))['out'];loss=paint_loss(out,torch.from_numpy(y[None]));loss.backward()
  if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('real_development_loss_gradient_nonfinite')
  report={'format':'cycle2-real-development-CPU-preflight-v1','task':kind,'actual_development_images':1,'tensor_shape':list(x[None].shape),'unique_GT_values':np.unique(y).tolist(),'valid_GT_pixels':int((y!=255).sum()),'positive_GT_pixels':int((y==1).sum()),'loss':float(loss.detach()),'passed':True,'GPU_operations':0,'TEST_images_loaded':0,'user_recordings_used':False,'source_expiries_extended':False,'expiry':min(r['expires_at'] for r in rows),'code_sha256':sha256(Path(__file__)),'trainer_sha256':sha256(ROOT/('scripts/train_lane_cycle2.py' if kind=='lane' else 'scripts/train_privacy_p2_cycle2.py')),'limits':['Geometry/loss/backward smoke, not accuracy or generalization']}
 out=ROOT/f'data/training/public-nextcycle/{kind}-CPU-preflight.json';atomic_json(out,report);atomic_json(ROOT/f'docs/reports/{kind.upper()}-CYCLE2-PREFLIGHT.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['lane','privacy']);a=p.parse_args();run(a.kind)
