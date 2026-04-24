import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class SETRNaive(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(hidden_size, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class SETRPup(nn.Module):
    """Progressive UPsampling decoder — best mIoU in the paper."""
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.stages = nn.Sequential(
            nn.Conv2d(hidden_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.stages(x)


class SETRModel(nn.Module):
    """SETR: SEgmentation TRansformer (CVPR 2021).

    Encoder: ViT-Large (patch16, pretrained at 384px).
    Decoder: PUP (default) or Naive.
    Position embeddings are interpolated automatically
    when image_size differs from the backbone's native 384px.
    """
    PATCH_SIZE = 16

    def __init__(
        self,
        num_classes=2,
        backbone="google/vit-large-patch16-384",
        decoder="pup",
        image_size=(512, 512),
    ):
        super().__init__()
        self.h_patches = image_size[0] // self.PATCH_SIZE
        self.w_patches = image_size[1] // self.PATCH_SIZE

        self.vit = ViTModel.from_pretrained(
            backbone, add_pooling_layer=False
        )
        hidden_size = self.vit.config.hidden_size  # 1024 for ViT-Large

        if image_size != (384, 384):
            self._resize_pos_embed(image_size)

        if decoder.lower() == "pup":
            self.decode_head = SETRPup(hidden_size, num_classes)
        elif decoder.lower() == "naive":
            self.decode_head = SETRNaive(hidden_size, num_classes)
        else:
            raise ValueError(f"Unknown decoder: {decoder!r}. Use 'pup' or 'naive'.")

    def _resize_pos_embed(self, image_size):
        """Interpolate position embeddings to match a non-384 image size."""
        native = self.vit.config.image_size // self.PATCH_SIZE  # 24 for 384px
        target_h = image_size[0] // self.PATCH_SIZE
        target_w = image_size[1] // self.PATCH_SIZE
        if native == target_h and native == target_w:
            return
        pos = self.vit.embeddings.position_embeddings  # (1, 1+N, D)
        cls_tok, patches = pos[:, :1], pos[:, 1:]
        D = patches.shape[-1]
        patches = patches.reshape(1, native, native, D).permute(0, 3, 1, 2)
        patches = torch.nn.functional.interpolate(
            patches, size=(target_h, target_w),
            mode="bilinear", align_corners=False
        )
        patches = patches.permute(0, 2, 3, 1).reshape(1, target_h * target_w, D)
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.cat([cls_tok, patches], dim=1)
        )

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        tokens = self.vit(pixel_values, return_dict=True).last_hidden_state
        tokens = tokens[:, 1:, :]  # drop [CLS] token
        x = tokens.reshape(B, self.h_patches, self.w_patches, -1)
        x = x.permute(0, 3, 1, 2)  # (B, D, H_p, W_p)
        return self.decode_head(x)


def get_setr_model(cfg):
    backbone = cfg.model_name or "google/vit-large-patch16-384"
    image_size = getattr(cfg, "image_size", (512, 512))
    decoder = getattr(cfg, "setr_decoder", "pup")
    return SETRModel(
        num_classes=cfg.num_classes,
        backbone=backbone,
        decoder=decoder,
        image_size=image_size,
    )