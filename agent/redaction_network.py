"""Torch-lazy research factory: two genuine YOLOX-Nano specialists, one contract.

This is NOT one Nano detector: inference executes a native-960 face detector and
a native-640 plate detector. The latter consumes bilinear resized shared input;
its decoded coordinates are returned in the shared input coordinate system.
No runtime download, deployment approval, confidence tuning or fallback occurs.
"""
from pathlib import Path
import sys


def build_redaction_network(*,yolox_root=None,face_input_size=960,plate_input_size=640):
    import torch
    from torch import nn
    from torch.nn import functional as F
    source=Path(yolox_root) if yolox_root else Path(__file__).resolve().parents[1]/'.cache/vendor/YOLOX'
    if not (source/'exps/default/yolox_nano.py').is_file():raise ValueError('official_yolox_source_missing')
    if (face_input_size,plate_input_size)!=(960,640):raise ValueError('unsupported_frozen_specialist_geometry')
    sys.path.insert(0,str(source))
    from yolox.exp import get_exp
    def nano():
        exp=get_exp(exp_file=str(source/'exps/default/yolox_nano.py'));exp.num_classes=1
        return exp.get_model()
    class RedactionSpecialists(nn.Module):
        def __init__(self):
            super().__init__();self.face=nano();self.plate=nano()
            self.face_input_size=face_input_size;self.plate_input_size=plate_input_size
        def plate_input(self,images):
            return F.interpolate(images,size=(plate_input_size,plate_input_size),mode='bilinear',align_corners=False)
        def forward(self,images):
            face=self.face(images);plate=self.plate(self.plate_input(images))
            zero_face=torch.zeros_like(face[...,5:6]);zero_plate=torch.zeros_like(plate[...,5:6])
            face=torch.cat((face[...,:6],zero_face),dim=-1)
            ratio=face_input_size/plate_input_size
            plate=torch.cat((plate[...,:4]*ratio,plate[...,4:5],zero_plate,plate[...,5:6]),dim=-1)
            return torch.cat((face,plate),dim=1)
    return RedactionSpecialists()
