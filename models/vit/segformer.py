from transformers import SegformerForSemanticSegmentation
import torch.nn as nn

def get_segformer_model(CFG):
    model = SegformerForSemanticSegmentation.from_pretrained(
        CFG.model_name,
        num_labels=CFG.num_classes,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
        use_safetensors=True
    )

    # If we're using non-standard input channels (e.g., 4-channel with thermal)
    if CFG.in_channels != 3:
        # Overwrite the first embedding layer's projection conv2d
        model.segformer.encoder.patch_embeddings[0].projection = nn.Conv2d(
            in_channels=CFG.in_channels,
            out_channels=model.segformer.encoder.patch_embeddings[0].projection.out_channels,
            kernel_size=7,
            stride=4,
            padding=3
        )

    # Optional: Freeze encoder if specified
    if CFG.freeze_encoder:
        for param in model.segformer.encoder.parameters():
            param.requires_grad = False

    return model
