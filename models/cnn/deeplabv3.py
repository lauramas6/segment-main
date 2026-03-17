import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def get_deeplabv3_model(CFG):
    # Load pretrained DeepLabV3 model
    # model = deeplabv3_resnet50(pretrained=True, progress=True, weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    # new api
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, progress=True)

    # Modify the classifier for your number of classes
    model.classifier[4] = nn.Conv2d(256, CFG.num_classes, kernel_size=1)

    # Modify input channels if needed
    if CFG.in_channels != 3:
        old_conv = model.backbone.conv1
        model.backbone.conv1 = nn.Conv2d(
            in_channels=CFG.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

    # Freeze encoder if specified
    if hasattr(CFG, "freeze_encoder") and CFG.freeze_encoder:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model
