import torch
import torch.nn.functional as F

def get_logits(output, num_classes=None):
    """
    Universal segmentation output extractor.
    Converts diverse model outputs into logits of shape [B, num_classes, H, W].
    Supports SAM, SegFormer, FRRN, DeepLab, and plain tensor/dict models.
    """
    # --- SAM (Segment Anything)
    if "Sam" in output.__class__.__name__:
        for key in ["low_res_masks", "pred_masks", "masks"]:
            if hasattr(output, key):
                masks = getattr(output, key)

                # If list/tuple, take the first element
                if isinstance(masks, (list, tuple)):
                    masks = masks[0]

                # Convert to tensor if needed
                if not isinstance(masks, torch.Tensor):
                    masks = torch.as_tensor(masks, dtype=torch.float32)

                # Handle different SAM output shapes
                # Target: [B, C, H, W]
                if masks.dim() == 2:      # [H, W]
                    masks = masks.unsqueeze(0).unsqueeze(0)
                elif masks.dim() == 3:    # [C, H, W]
                    masks = masks.unsqueeze(0)
                elif masks.dim() == 4:    # [B, C, H, W] → fine
                    pass
                elif masks.dim() == 5:    # [B, 1, C, H, W] or [B, num_masks, 1, H, W]
                    # squeeze singleton dims intelligently
                    if masks.shape[1] == 1:
                        masks = masks.squeeze(1)  # [B, C, H, W]
                    elif masks.shape[2] == 1:
                        masks = masks.squeeze(2)  # [B, num_masks, H, W]

                # Ensure final shape
                if masks.dim() != 4:
                    raise ValueError(f"SAM mask tensor has unexpected shape: {masks.shape}")

                # Interpolate
                masks = F.interpolate(masks, size=(512, 512), mode="bilinear", align_corners=False)
                return masks

        print(f"[DEBUG] SAM output attributes: {dir(output)}")
        raise ValueError("Could not locate SAM mask tensor in model output.")

    # --- SegFormer, DeepLab, etc.
    if hasattr(output, "logits"):
        return output.logits

    # --- FRRN / torchvision-like dicts
    if isinstance(output, dict):
        if "out" in output:
            return output["out"]
        elif "logits" in output:
            return output["logits"]

    # --- plain tensor
    if isinstance(output, torch.Tensor):
        if output.dim() == 3:  # [C, H, W]
            output = output.unsqueeze(0)  # [1, C, H, W]
        return output

    raise ValueError(f"Unknown model output type: {type(output)}")