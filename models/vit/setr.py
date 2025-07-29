# models/setr.py

from transformers import SetrModel, SetrForSemanticSegmentation
import torch.nn as nn

def get_setr_model(CFG):
    model = SetrForSemanticSegmentation.from_pretrained(
        CFG.model_name,
        num_labels=CFG.num_classes,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
        use_safetensors=True
    )

    # Handle non-standard input channels (e.g., thermal or extra modality)
    if CFG.in_channels != 3:
        # Replace the first conv2d projection layer
        model.setr.encoder.patch_embeddings[0].projection = nn.Conv2d(
            in_channels=CFG.in_channels,
            out_channels=model.setr.encoder.patch_embeddings[0].projection.out_channels,
            kernel_size=7,
            stride=4,
            padding=3
        )

    # Optionally freeze the encoder
    if CFG.freeze_encoder:
        for param in model.setr.encoder.parameters():
            param.requires_grad = False

    return model
