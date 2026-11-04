"""Torch-lazy research P2 face detector and frozen legacy plate specialist.

New architecture/geometry, not a precision optimization of prior weights.
Shared1920 -> global640 + four crop960->640 faces; one shared1920->640 plate.
No downloads, runtime selection, tolerance changes, or data access in factory.
"""
from pathlib import Path
import sys
ARCHITECTURE='YOLOX-Nano P2 face640 global plus four tiles and frozen plate640'

def make_p2_face(*,yolox_root=None):
 import torch
 from torch import nn
 root=Path(__file__).resolve().parents[1];source=(Path(yolox_root) if yolox_root else root/'.cache/vendor/YOLOX').resolve()
 if not source.is_relative_to(root) or not (source/'yolox/models/yolo_head.py').is_file():raise ValueError('workspace_official_yolox_source_required')
 sys.path.insert(0,str(source))
 from yolox.models import YOLOX,YOLOPAFPN,YOLOXHead
 neck=YOLOPAFPN(.33,.25,in_features=('dark2','dark3','dark4'),in_channels=[128,256,512],depthwise=True)
 neck.backbone.out_features=('dark2','dark3','dark4');neck.backbone.dark5=nn.Identity()
 head=YOLOXHead(1,.25,strides=[4,8,16],in_channels=[128,256,512],depthwise=True)
 face=YOLOX(neck,head)
 for module in face.modules():
  if isinstance(module,nn.BatchNorm2d):module.eps=1e-3;module.momentum=.03
 face.head.initialize_biases(.01)
 return face

def assemble_p2_network(face,plate):
 import torch
 from torch import nn
 from torch.nn import functional as F
 class P2Specialists(nn.Module):
  def __init__(self):super().__init__();self.face=face;self.plate=plate
  def forward(self,images):
   if images.ndim!=4 or images.shape[1]!=3 or images.shape[-2:]!=(1920,1920):raise ValueError('shared1920_required')
   parts=[]
   for index in range(5):
    if index==0:pixels=images;scale=3.;left=top=0.
    else:
     tile=index-1;left=(tile%2)*960;top=(tile//2)*960;pixels=images[:,:,top:top+960,left:left+960];scale=1.5
    raw=self.face(F.interpolate(pixels,size=(640,640),mode='bilinear',align_corners=False));zero=torch.zeros_like(raw[...,5:6])
    parts.append(torch.cat((raw[...,0:1]*scale+left,raw[...,1:2]*scale+top,raw[...,2:4]*scale,raw[...,4:6],zero),-1))
   raw=self.plate(F.interpolate(images,size=(640,640),mode='bilinear',align_corners=False));zero=torch.zeros_like(raw[...,5:6])
   parts.append(torch.cat((raw[...,:4]*3.,raw[...,4:5],zero,raw[...,5:6]),-1))
   return torch.cat(parts,1)
 return P2Specialists()

def build_redaction_p2_network(*,yolox_root=None):
 from .redaction_network import build_redaction_network
 legacy=build_redaction_network(yolox_root=yolox_root)
 return assemble_p2_network(make_p2_face(yolox_root=yolox_root),legacy.plate)

def load_p2_warmstart(bundle,legacy_state):
 """Explicit active-backbone and corresponding-head transplantation from v4.

 P2 neck and first stem are new; no approximate/silent strict=False state load.
 Every assigned tensor must match shape. Plate state is copied strictly.
 """
 import torch
 legacy_face={k.removeprefix('face.'):v for k,v in legacy_state.items() if k.startswith('face.')}
 plate={k.removeprefix('plate.'):v for k,v in legacy_state.items() if k.startswith('plate.')}
 bundle.plate.load_state_dict(plate,strict=True)
 state=bundle.face.state_dict();copied={};initial=[]
 for name,value in state.items():
  origin=None
  if name.startswith('backbone.backbone.'):origin=name
  elif name.startswith('head.'):
   fields=name.split('.')
   if len(fields)>3 and fields[2] in ('0','1','2'):
    index=int(fields[2]);oldindex=0 if index==0 else index-1
    # P2's new32-channel stem has no corresponding old64-channel stem.
    if not (fields[1]=='stems' and index==0):origin='.'.join(fields[:2]+[str(oldindex)]+fields[3:])
  if origin is not None:
   if origin not in legacy_face or legacy_face[origin].shape!=value.shape:raise ValueError('explicit_transfer_shape_or_key_mismatch:'+name)
   state[name]=legacy_face[origin].clone();copied[name]=origin
  else:initial.append(name)
 bundle.face.load_state_dict(state,strict=True)
 for name,origin in copied.items():
  if not torch.equal(bundle.face.state_dict()[name],legacy_face[origin]):raise ValueError('warmstart_byte_verification_failed')
 return {'assigned_tensor_count':len(copied),'initialized_tensor_count':len(initial),'assignment':copied,'new_tensor_names':initial,'plate_strict_loaded':True,'full_model_state_identical_claim':False}
