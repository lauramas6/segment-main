"""Mask2Former for Semantic Segmentation
Universal image segmentation framework supporting panoptic, instance and semantic segmentation.
Masked-attention Mask Transformer provides state-of-the-art performance.

Reference: https://huggingface.co/docs/transformers/en/model_doc/mask2former
Paper: Masked-attention Mask Transformer for Universal Image Segmentation
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch.nn.functional as F


class Mask2FormerForSegmentation(nn.Module):
    """Mask2Former for semantic segmentation"""
    
    def __init__(
        self,
        num_classes=2,
        image_size=518,
        model_name="facebook/mask2former-swin-small-ade-semantic"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Handle both tuple and int image_size
        if isinstance(image_size, (tuple, list)):
            self.image_size = image_size[0]  # Use first dimension (assume square)
        else:
            self.image_size = image_size
        
        # Load pre-trained Mask2Former model
        print(f"[INFO] Loading {model_name}...")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        print(f"[INFO] Successfully loaded {model_name}")
        
        # Load image processor for correct preprocessing
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Get number of classes from pretrained model
        self.num_pretrained_classes = self.model.config.num_labels
        
        print(f"[INFO] Pretrained model num_classes: {self.num_pretrained_classes}")
        print(f"[INFO] Target num_classes: {num_classes}")
        
        # If target classes differ from pretrained, we need to adapt
        if num_classes != self.num_pretrained_classes:
            self._adapt_classifier_head(num_classes)
    
    def _adapt_classifier_head(self, num_classes):
        """Adapt the classifier head for different number of classes"""
        # Get the current class head dimension
        old_num_classes = self.model.config.num_labels
        hidden_dim = self.model.config.hidden_dim
        
        print(f"[INFO] Adapting classifier from {old_num_classes} to {num_classes} classes")
        
        # Create new classifier head
        new_class_predictor = nn.Linear(hidden_dim, num_classes)
        
        # Initialize with small weights
        nn.init.normal_(new_class_predictor.weight, std=0.01)
        nn.init.constant_(new_class_predictor.bias, 0)
        
        # Replace the old classifier
        self.model.class_predictor = new_class_predictor
        
        # Update config
        self.model.config.num_labels = num_classes
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (B, 3, H, W)
        Returns:
            logits: (B, num_classes, H, W)
        """
        batch_size = pixel_values.shape[0]
        input_size = pixel_values.shape[-2:]  # Get actual input size (H, W)
        
        # Forward pass through Mask2Former
        outputs = self.model(pixel_values, return_dict=True)
        
        # Extract class logits: (batch_size, num_queries, num_classes+1)
        class_queries_logits = outputs.class_queries_logits
        
        # Extract mask logits: (batch_size, num_queries, height, width)
        masks_queries_logits = outputs.masks_queries_logits
        
        # Get class predictions (exclude no-object class)
        class_logits = class_queries_logits[:, :, :-1]  # (B, num_queries, num_classes)
        
        # Get predicted classes per query
        predicted_classes = class_logits.argmax(dim=-1)  # (B, num_queries)
        
        # Compute class-specific mask predictions
        # (B, num_queries, height, width)
        num_queries = masks_queries_logits.shape[1]
        class_masks = []
        
        for i in range(self.num_classes):
            # Get mask confidence for this class
            class_mask = torch.zeros(
                batch_size, 1, *masks_queries_logits.shape[-2:],
                device=masks_queries_logits.device,
                dtype=masks_queries_logits.dtype
            )
            
            # Queries predicted as this class
            class_query_mask = (predicted_classes == i)  # (B, num_queries)
            
            for b in range(batch_size):
                if class_query_mask[b].any():
                    # Take max mask among queries predicted as this class
                    relevant_masks = masks_queries_logits[b][class_query_mask[b]]
                    class_mask[b] = relevant_masks.max(dim=0, keepdim=True)[0]
            
            class_masks.append(class_mask)
        
        # Stack class masks: (B, num_classes, H, W)
        seg_logits = torch.cat(class_masks, dim=1)
        
        # Upsample to original input size if needed
        if seg_logits.shape[-2:] != input_size:
            seg_logits = F.interpolate(
                seg_logits,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
        
        return seg_logits


def get_mask2former_model(cfg):
    """Factory function to create Mask2Former segmentation model
    
    Available official Mask2Former models (examples):
    - facebook/mask2former-swin-tiny-coco-instance
    - facebook/mask2former-swin-small-coco-instance
    - facebook/mask2former-swin-small-ade-semantic
    - facebook/mask2former-swin-small-cityscapes-panoptic
    
    For semantic segmentation, use models with 'semantic' or 'instance' in the name.
    
    Reference: https://huggingface.co/models?filter=mask2former
    """
    model = Mask2FormerForSegmentation(
        num_classes=cfg.num_classes,
        image_size=cfg.image_size,
        model_name="facebook/mask2former-swin-large-ade-semantic"
    )
    return model