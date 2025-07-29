# models/mask2former.py

from transformers import Mask2FormerForSemanticSegmentation
import torch.nn as nn

def get_mask2former_model(CFG):
    model = Mask2FormerForSemanticSegmentation.from_pretrained(
        CFG.model_name,
        num_labels=CFG.num_classes,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
        use_safetensors=True
    )

    # Custom input channels (e.g., RGB + thermal)
    if CFG.in_channels != 3:
        # Locate the first conv layer to replace
        # Some Mask2Former backbones like Swin use patch_embed
        # So we safely attempt to modify the patch embedding projection
        try:
            model.model.encoder.patch_embed.proj = nn.Conv2d(
                in_channels=CFG.in_channels,
                out_channels=model.model.encoder.patch_embed.proj.out_channels,
                kernel_size=model.model.encoder.patch_embed.proj.kernel_size,
                stride=model.model.encoder.patch_embed.proj.stride,
                padding=model.model.encoder.patch_embed.proj.padding,
                bias=model.model.encoder.patch_embed.proj.bias is not None
            )
        except AttributeError:
            raise NotImplementedError("Custom input channel handling needs to be adapted for this Mask2Former backbone")

    # Freeze encoder if requested
    if CFG.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    return model
