"""DINOv3-ViTB16 for Semantic Segmentation
Self-supervised Vision Transformer by Meta AI using the ViTB16 variant.
Optimized implementation for semantic segmentation with frozen backbone features.

Official Model: facebook/dinov3-vitb16-pretrain-lvd1689m
Paper: https://arxiv.org/abs/2508.10104
Reference: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, DINOv3ViTModel
import torch.nn.functional as F
import math


class DINOv3SegmentationHead(nn.Module):
    """Dense segmentation decoder for DINOv3 patch features"""
    
    def __init__(self, in_channels, num_classes, hidden_dim=256):
        super().__init__()
        
        # Input normalization
        self.norm = nn.LayerNorm(in_channels)
        
        # Progressive decoder with residual-like structure
        self.decoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                # Use relu for kaiming_normal as it's compatible with GELU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: (B, N, C) where N is num_patches, C is feature dimension
        Returns:
            logits: (B, N, num_classes)
        """
        # Normalize patch features
        x = self.norm(features)
        
        # Decode through MLP layers
        x = self.decoder(x)
        
        # Classify each patch independently
        logits = self.classifier(x)
        
        return logits


class NewDINOv3ForSegmentation(nn.Module):
    """DINOv3-ViTB16 model adapted for semantic segmentation
    
    Uses frozen pretrained DINOv3-ViTB16 features with a learnable segmentation head.
    The backbone provides excellent dense features for pixel-level prediction tasks.
    """
    
    def __init__(self, 
                 num_classes=2, 
                 image_size=518, 
                 model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
                 freeze_backbone=True,
                 hidden_dim=256):
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        # Normalize image_size
        if isinstance(image_size, (tuple, list)):
            self.image_size = image_size[0]
        else:
            self.image_size = image_size
        
        print(f"[INFO] Loading DINOv3 backbone: {model_name}")
        
        # Load pretrained DINOv3-ViTB16 model
        try:
            self.backbone = DINOv3ViTModel.from_pretrained(
                model_name,
                trust_remote_code=False
            )
            print(f"[INFO] ✓ Successfully loaded {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_name}: {e}")
        
        # Freeze backbone if specified (DINOv3 features are already excellent)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[INFO] Backbone frozen - using pretrained features only")
        else:
            print("[INFO] Backbone trainable - fine-tuning enabled")
        
        # Get feature dimension from model config
        self.feature_dim = self.backbone.config.hidden_size  # 768 for ViTB16
        self.patch_size = self.backbone.config.patch_size   # 16 for ViTB16
        
        print(f"[INFO] Feature dimension: {self.feature_dim}, Patch size: {self.patch_size}")
        
        # Segmentation head
        self.seg_head = DINOv3SegmentationHead(
            in_channels=self.feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )
        
        # Track spatial dimensions after first forward pass
        self.num_patches = None
        self.spatial_h = None
        self.spatial_w = None
    
    def forward(self, pixel_values):
        """
        Forward pass for semantic segmentation
        
        Args:
            pixel_values: Tensor of shape (B, 3, H, W) with values in [0, 1] or [-1, 1]
        
        Returns:
            logits: Tensor of shape (B, num_classes, H, W)
        """
        batch_size = pixel_values.shape[0]
        input_h, input_w = pixel_values.shape[-2:]
        input_size = (input_h, input_w)
        
        # Extract features from frozen/trainable backbone
        if self.freeze_backbone:
            with torch.no_grad():
                backbone_output = self.backbone(pixel_values, output_hidden_states=False)
        else:
            backbone_output = self.backbone(pixel_values, output_hidden_states=False)
        
        # Get patch embeddings (includes [CLS] token as first position)
        # Shape: (B, 1 + num_patches + num_registers, feature_dim)
        # For DINOv3: 1 [CLS] + 4 registers + (H/16 * W/16) patches
        last_hidden = backbone_output.last_hidden_state
        
        # Remove [CLS] token and register tokens (keep only patch tokens)
        # DINOv3 has 4 register tokens after [CLS]
        patch_features = last_hidden[:, 5:, :]  # Skip [CLS] (1) + registers (4)
        
        # Determine spatial grid size from patch count
        if self.num_patches is None:
            self.num_patches = patch_features.shape[1]
            
            # Calculate spatial dimensions
            sqrt_patches = int(math.sqrt(self.num_patches))
            self.spatial_h = sqrt_patches
            self.spatial_w = sqrt_patches
            
            print(f"[INFO] Detected patches: {self.num_patches} ({self.spatial_h}x{self.spatial_w})")
        
        # Apply segmentation head to all patch features
        seg_logits = self.seg_head(patch_features)  # (B, num_patches, num_classes)
        
        # Reshape to spatial grid
        seg_logits = seg_logits.reshape(
            batch_size,
            self.spatial_h,
            self.spatial_w,
            self.num_classes
        )
        
        # Permute to (B, num_classes, H, W) for standard PyTorch format
        seg_logits = seg_logits.permute(0, 3, 1, 2)
        
        # Upsample to original input size
        if seg_logits.shape[-2:] != input_size:
            seg_logits = F.interpolate(
                seg_logits,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
        
        return seg_logits


def get_new_dinov3_model(cfg):
    """Factory function for creating DINOv3-ViTB16 segmentation model
    
    Available DINOv3 variants from facebook:
    - dinov3-vitb16-pretrain-lvd1689m   (base, ~86M params, recommended for speed)
    - dinov3-vitl16-pretrain-lvd1689m   (large, ~300M params, better quality)
    - dinov3-vitg16-pretrain-lvd1689m   (giant, ~1.3B params, best quality)
    
    Args:
        cfg: Configuration object with attributes:
            - num_classes: Number of segmentation classes
            - image_size: Input image size (tuple or int)
            - model_name: (optional) HuggingFace model identifier
    
    Returns:
        NewDINOv3ForSegmentation: Model ready for training/inference
    """
    
    # Use model name from config if provided, otherwise use ViTB16 (default)
    model_name = getattr(cfg, 'model_name', None) or "facebook/dinov3-vitb16-pretrain-lvd1689m"
    
    model = NewDINOv3ForSegmentation(
        num_classes=cfg.num_classes,
        image_size=cfg.image_size,
        model_name=model_name,
        freeze_backbone=True,
        hidden_dim=256
    )
    
    return model
