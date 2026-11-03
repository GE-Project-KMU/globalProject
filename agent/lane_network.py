"""Explicit Torch checkpoint factory. The application ONNX path never imports Torch."""


def make_lane_network(*, pretrained=False, aspp_dilations=(12, 24, 36)):
    import torch
    import torchvision
    rates=list(aspp_dilations)
    if len(rates)!=3 or any(type(r) is not int or not 1<=r<=64 for r in rates):
        raise ValueError("invalid_lane_aspp_dilations")
    if pretrained:
        model=torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
        model.classifier[-1]=torch.nn.Conv2d(model.classifier[-1].in_channels,2,1)
        model.aux_classifier=None
    else:
        model=torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None,weights_backbone=None,num_classes=2,aux_loss=False)
    # Keep identical parameter keys/shapes and explicitly encode the research
    # dilation variant. The legacy default remains exactly torchvision's.
    for block,rate in zip(model.classifier[0].convs[1:4],rates):
        block[0].dilation=(rate,rate);block[0].padding=(rate,rate)
    return model


def freeze_batch_norm_statistics(model):
    import torch
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):
            layer.eval()
