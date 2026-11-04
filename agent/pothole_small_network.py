"""Torch-lazy official YOLOX-S capacity candidate; never an untrained substitute."""
from pathlib import Path
ARCHITECTURE='YOLOX-S'
INITIAL_SHA='f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30'

def make_pothole_small(*,pretrained=None,yolox_root=None):
    import sys,torch
    from .paths import ROOT,workspace_path,sha256
    source=workspace_path(yolox_root or ROOT/'.cache/vendor/YOLOX',exists=True);sys.path.insert(0,str(source))
    from yolox.exp import get_exp
    exp=get_exp(exp_file=str(source/'exps/default/yolox_s.py'));exp.num_classes=1;model=exp.get_model();model.head.decode_in_inference=True
    audit={'transferred_tensors':0,'reinitialized_keys':[]}
    if pretrained is not None:
        path=workspace_path(pretrained,exists=True)
        if sha256(path)!=INITIAL_SHA:raise ValueError('pinned_official_COCO_S_initialization_required')
        state=torch.load(path,map_location='cpu',weights_only=True)['model'];own=model.state_dict();compatible={k:v for k,v in state.items() if k in own and own[k].shape==v.shape};missing=sorted(set(own)-set(compatible))
        if missing!=sorted(f'head.cls_preds.{i}.{v}' for i in range(3) for v in ('weight','bias')):raise ValueError('only_COCO_classifier_reset_allowed')
        model.load_state_dict(compatible,strict=False);audit={'transferred_tensors':len(compatible),'reinitialized_keys':missing}
    return model,audit
