"""Research prototype: frozen legacy specialists plus four lightweight P2 heads.

Torch imports stay lazy. Original .face/.plate tensors and the ordered102900
legacy proposal prefix are preserved. Rescue heads are shared across four tiles;
the added230400 proposals are NOT a precision-equivalent trained architecture.
"""
from pathlib import Path
import math

ARCHITECTURE='frozen legacy YOLOX-Nano specialists plus four tile P2 rescue'
LEGACY_PROPOSALS=102900
RESCUE_PROPOSALS=230400
RESCUE_SCORE_SCALE=.5
RESCUE_INITIAL_PROBABILITY=1e-4
MAX_LABELS=512

def build_redaction_rescue_network(*,yolox_root=None,shared_input_size=1920,face_input_size=960,plate_input_size=640):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from .redaction_tiled_network import inference_passes
    from .redaction_network import build_redaction_network
    inference_passes(shared_input_size=shared_input_size,face_input_size=face_input_size,plate_input_size=plate_input_size)
    root=Path(__file__).resolve().parents[1]
    source=(Path(yolox_root) if yolox_root else root/'.cache/vendor/YOLOX').resolve()
    if not source.is_relative_to(root):raise ValueError('workspace_yolox_source_required')
    canonical=build_redaction_network(yolox_root=source)
    from yolox.models import YOLOXHead
    from yolox.models.network_blocks import BaseConv

    class FrozenLegacyP2Rescue(nn.Module):
        def __init__(self):
            super().__init__();self.face=canonical.face;self.plate=canonical.plate
            # CSPDarknet already computes dark2 in exactly this execution order.
            # Only the returned dictionary gains a reference, never an op/module.
            self.face.backbone.backbone.out_features=('dark2','dark3','dark4','dark5')
            for legacy in (self.face,self.plate):
                legacy.requires_grad_(False);legacy.eval()
            self.rescue_projection=BaseConv(96,32,1,1,act='silu')
            self.rescue_head=YOLOXHead(1,width=.125,strides=[4],in_channels=[256],depthwise=True)
            for group in (self.rescue_projection,self.rescue_head):
                for module in group.modules():
                    if isinstance(module,nn.BatchNorm2d):module.eps=1e-3;module.momentum=.03
            with torch.no_grad():
                for conv in [*self.rescue_head.obj_preds,*self.rescue_head.cls_preds]:
                    conv.weight.zero_();conv.bias.fill_(math.log(RESCUE_INITIAL_PROBABILITY/(1-RESCUE_INITIAL_PROBABILITY)))

        def train(self,mode=True):
            super().train(mode)
            # Legacy BN buffers and parameters remain frozen even when the new
            # projection/head use their native training-mode loss.
            self.face.eval();self.plate.eval()
            return self

        def legacy_features(self,pixels):
            """Official YOLOPAFPN.forward order, with the existing dark2 return."""
            neck=self.face.backbone
            with torch.no_grad():
                out=neck.backbone(pixels)
                x2,x1,x0=[out[name] for name in neck.in_features]
                fpn_out0=neck.lateral_conv0(x0)
                f_out0=neck.upsample(fpn_out0)
                f_out0=torch.cat([f_out0,x1],1)
                f_out0=neck.C3_p4(f_out0)
                fpn_out1=neck.reduce_conv1(f_out0)
                f_out1=neck.upsample(fpn_out1)
                f_out1=torch.cat([f_out1,x2],1)
                pan_out2=neck.C3_p3(f_out1)
                p_out1=neck.bu_conv2(pan_out2)
                p_out1=torch.cat([p_out1,fpn_out1],1)
                pan_out1=neck.C3_n3(p_out1)
                p_out0=neck.bu_conv1(pan_out1)
                p_out0=torch.cat([p_out0,fpn_out0],1)
                pan_out0=neck.C3_n4(p_out0)
            return (pan_out2,pan_out1,pan_out0),out['dark2']

        def rescue_features(self,features,dark2):
            context=F.interpolate(features[0].detach(),size=dark2.shape[-2:],mode='bilinear',align_corners=False)
            return self.rescue_projection(torch.cat((dark2.detach(),context),1))

        def forward(self,images):
            if self.training:raise ValueError('rescue_inference_requires_eval_use_rescue_loss_for_training')
            if images.ndim!=4 or images.shape[1]!=3 or images.shape[-2:]!=(1920,1920):raise ValueError('shared_1920_bgr_tensor_required')
            legacy=[];rescue=[]
            def legacy_coordinates(raw,scale,left,top,head):
                # Same operations and concatenation order as the frozen tiled
                # factory, including multiplication by1 and zero offsets.
                cx=raw[...,0:1]*scale+left;cy=raw[...,1:2]*scale+top
                wh=raw[...,2:4]*scale;objectness=raw[...,4:5]
                probability=raw[...,5:6];zero=torch.zeros_like(probability)
                classes=(probability,zero) if head=='face' else (zero,probability)
                return torch.cat((cx,cy,wh,objectness,*classes),-1)
            with torch.no_grad():raw=self.face(F.interpolate(images,size=(960,960),mode='bilinear',align_corners=False))
            legacy.append(legacy_coordinates(raw,2.,0.,0.,'face'))
            for tile in range(4):
                left=(tile%2)*960;top=(tile//2)*960;pixels=images[:,:,top:top+960,left:left+960]
                features,dark2=self.legacy_features(pixels)
                with torch.no_grad():raw=self.face.head(features)
                legacy.append(legacy_coordinates(raw,1.,left,top,'face'))
                extra=self.rescue_head((self.rescue_features(features,dark2),))
                zero=torch.zeros_like(extra[...,5:6])
                rescue.append(torch.cat((extra[...,0:1]+left,extra[...,1:2]+top,extra[...,2:4],extra[...,4:5]*RESCUE_SCORE_SCALE,extra[...,5:6],zero),-1))
            with torch.no_grad():raw=self.plate(F.interpolate(images,size=(640,640),mode='bilinear',align_corners=False))
            legacy.append(legacy_coordinates(raw,3.,0.,0.,'plate'))
            return torch.cat((*legacy,*rescue),1)

        def rescue_loss(self,tile_images,labels):
            if not self.training:raise ValueError('rescue_loss_requires_train_mode')
            if tile_images.ndim!=4 or tile_images.shape[1]!=3 or tile_images.shape[-2:]!=(960,960):raise ValueError('native_960_tile_training_required')
            if labels.ndim!=3 or labels.shape[0]!=tile_images.shape[0] or labels.shape[2]!=5 or not 1<=labels.shape[1]<=MAX_LABELS:raise ValueError('native_labels_B_N_5_max512_required')
            if not torch.isfinite(labels).all():raise ValueError('finite_native_labels_required')
            nonempty=labels.sum(-1)>0
            if torch.any(labels[...,0][nonempty]!=0):raise ValueError('face_single_class_native_labels_required')
            features,dark2=self.legacy_features(tile_images)
            # Native head receives ALL labels unchanged. No truncation or
            # inference score scaling enters native logits/assignment/loss.
            return self.rescue_head((self.rescue_features(features,dark2),),labels,tile_images)

    return FrozenLegacyP2Rescue()

def load_frozen_legacy(bundle,state):
    """Exact transplant only; no approximate/silent strict=False loading."""
    expected={name for name in bundle.state_dict() if name.startswith(('face.','plate.'))}
    if set(state)!=expected:raise ValueError('complete_legacy_face_plate_state_required')
    for name in ('face','plate'):
        prefix=name+'.';sub={key.removeprefix(prefix):value for key,value in state.items() if key.startswith(prefix)}
        getattr(bundle,name).load_state_dict(sub,strict=True)
        getattr(bundle,name).requires_grad_(False);getattr(bundle,name).eval()
    return {'legacy_tensor_count':len(expected),'all_legacy_parameters_frozen':True,'legacy_BN_eval':True,'approximate_transplant':False}
