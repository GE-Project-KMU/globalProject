"""Optional research-only global+quadrant face inference; Torch stays lazy.

Uses exactly the canonical .face/.plate state keys, not separately learned tile
weights. No download, source data access, default selection or threshold tuning.
"""
from pathlib import Path

ARCHITECTURE='two YOLOX-Nano specialists global plus four tiles'

def inference_passes(*,shared_input_size=1920,face_input_size=960,plate_input_size=640):
    if any(type(x) is not int for x in (shared_input_size,face_input_size,plate_input_size)) or (shared_input_size,face_input_size,plate_input_size)!=(1920,960,640):
        raise ValueError('unsupported_frozen_tiled_geometry')
    return (
        {'head':'face','crop':None,'resize':960,'coordinate_scale':2.,'offset':(0.,0.)},
        {'head':'face','crop':(0,0,960,960),'resize':None,'coordinate_scale':1.,'offset':(0.,0.)},
        {'head':'face','crop':(960,0,1920,960),'resize':None,'coordinate_scale':1.,'offset':(960.,0.)},
        {'head':'face','crop':(0,960,960,1920),'resize':None,'coordinate_scale':1.,'offset':(0.,960.)},
        {'head':'face','crop':(960,960,1920,1920),'resize':None,'coordinate_scale':1.,'offset':(960.,960.)},
        {'head':'plate','crop':None,'resize':640,'coordinate_scale':3.,'offset':(0.,0.)},
    )

def assemble_tiled_network(face,plate,*,shared_input_size=1920,face_input_size=960,plate_input_size=640):
    import torch
    from torch import nn
    from torch.nn import functional as F
    passes=inference_passes(shared_input_size=shared_input_size,face_input_size=face_input_size,plate_input_size=plate_input_size)
    class TiledSpecialists(nn.Module):
        def __init__(self):
            super().__init__();self.face=face;self.plate=plate
        def forward(self,images):
            if images.ndim!=4 or images.shape[1]!=3 or images.shape[-2:]!=(1920,1920):raise ValueError('shared_1920_bgr_tensor_required')
            outputs=[]
            for spec in passes:
                pixels=images
                if spec['crop'] is not None:
                    x0,y0,x1,y1=spec['crop'];pixels=pixels[:,:,y0:y1,x0:x1]
                if spec['resize'] is not None:pixels=F.interpolate(pixels,size=(spec['resize'],spec['resize']),mode='bilinear',align_corners=False)
                raw=self.face(pixels) if spec['head']=='face' else self.plate(pixels)
                cx=raw[...,0:1]*spec['coordinate_scale']+spec['offset'][0]
                cy=raw[...,1:2]*spec['coordinate_scale']+spec['offset'][1]
                wh=raw[...,2:4]*spec['coordinate_scale'];objectness=raw[...,4:5]
                probability=raw[...,5:6];zero=torch.zeros_like(probability)
                classes=(probability,zero) if spec['head']=='face' else (zero,probability)
                outputs.append(torch.cat((cx,cy,wh,objectness,*classes),dim=-1))
            return torch.cat(outputs,dim=1)
    return TiledSpecialists()

def build_redaction_tiled_network(*,yolox_root=None,shared_input_size=1920,face_input_size=960,plate_input_size=640):
    inference_passes(shared_input_size=shared_input_size,face_input_size=face_input_size,plate_input_size=plate_input_size)
    root=Path(__file__).resolve().parents[1]
    source=(Path(yolox_root) if yolox_root is not None else root/'.cache/vendor/YOLOX').resolve()
    if not source.is_relative_to(root):raise ValueError('yolox_source_outside_workspace')
    from .redaction_network import build_redaction_network
    canonical=build_redaction_network(yolox_root=source,face_input_size=face_input_size,plate_input_size=plate_input_size)
    return assemble_tiled_network(canonical.face,canonical.plate,shared_input_size=shared_input_size,face_input_size=face_input_size,plate_input_size=plate_input_size)
