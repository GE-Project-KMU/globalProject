"""Lazy Torch factory: stride4 detail residual on the unchanged lane backbone.

Not an official DeepLabv3+ checkpoint; this is a small research decoder inspired
by its low-level feature fusion. Runtime ONNX inference never imports Torch.
"""
ARCHITECTURE='deeplabv3_mobilenet_v3_large_stride4_residual_refinement'

def make_lane_refinement_network(*,pretrained=False,aspp_dilations=(6,12,18)):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from .lane_network import make_lane_network
    class LaneRefinement(nn.Module):
        def __init__(self):
            super().__init__();self.base=make_lane_network(pretrained=pretrained,aspp_dilations=aspp_dilations)
            self.low_projection=nn.Sequential(nn.Conv2d(24,16,1,bias=False),nn.ReLU())
            self.context_projection=nn.Sequential(nn.Conv2d(256,64,1,bias=False),nn.ReLU())
            self.refinement=nn.Sequential(nn.Conv2d(80,80,3,padding=1,groups=80,bias=False),nn.ReLU(),nn.Conv2d(80,32,1,bias=False),nn.ReLU(),nn.Conv2d(32,2,1))
            nn.init.zeros_(self.refinement[-1].weight);nn.init.zeros_(self.refinement[-1].bias)
        def forward(self,pixels):
            size=pixels.shape[-2:];value=pixels;low=None
            for key,layer in self.base.backbone.items():
                value=layer(value)
                if key=='3':low=value
            if low is None:raise ValueError('stride4_low_feature_missing')
            context=value
            for layer in list(self.base.classifier.children())[:-1]:context=layer(context)
            coarse=self.base.classifier[-1](context)
            detail=self.low_projection(low);high=F.interpolate(self.context_projection(context),size=detail.shape[-2:],mode='bilinear',align_corners=False)
            residual=self.refinement(torch.cat([detail,high],dim=1))
            return {'out':F.interpolate(coarse,size=size,mode='bilinear',align_corners=False)+F.interpolate(residual,size=size,mode='bilinear',align_corners=False)}
    return LaneRefinement()

def load_legacy_warmstart(model,state):
    """Load every legacy learned byte; no silent missing/shape-tolerant load."""
    model.base.load_state_dict(state,strict=True)
    import torch
    for key,value in state.items():
        if not torch.equal(value,model.base.state_dict()[key]):raise ValueError('legacy_warmstart_state_changed')
    if torch.count_nonzero(model.refinement[-1].weight) or torch.count_nonzero(model.refinement[-1].bias):raise ValueError('zero_residual_initialization_required')
