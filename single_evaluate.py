#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from config import CFG
from models.factory import get_model
from models.model_zoo import MODEL_ZOO
from utils.helpers import get_logits
from utils.visualization import save_mask, save_overlay, load_palette_from_csv
from utils.metrics import iou_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.flir_extractor import FlirImageExtractor


# ------------------ CLI ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Single image evaluation with temperature extraction")
    parser.add_argument("--image_path", type=str, required=True, help="Path to FLIR image")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--label_csv", type=str, required=True)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------ CONFIG ------------------
    CFG.architecture = args.architecture
    CFG.model_name = args.model_name
    CFG.dataset_root = args.data_root
    CFG.label_csv = args.label_csv
    CFG.in_channels = args.in_channels
    CFG.num_classes = args.num_classes
    CFG.weights = args.weights

    model_cfg = MODEL_ZOO.get(CFG.architecture, {})
    CFG.image_size = model_cfg.get("image_size", CFG.image_size)

    if not os.path.exists(CFG.weights):
        raise FileNotFoundError(f"Checkpoint not found: {CFG.weights}")

    # ------------------ LOAD MODEL ------------------
    ckpt = torch.load(CFG.weights, map_location=CFG.device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    model = get_model().to(CFG.device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ------------------ LOAD IMAGE ------------------
    orig_img = Image.open(args.image_path).convert("RGB")
    orig_size = orig_img.size  # (W, H)

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(orig_img).unsqueeze(0).to(CFG.device)

    # ------------------ PREDICT MASK ------------------
    with torch.no_grad():
        logits = get_logits(model(image_tensor))
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # Resize back to original size
    pred_mask = cv2.resize(preds.astype(np.uint8), orig_size, interpolation=cv2.INTER_NEAREST)

    # ------------------ LOAD GROUND TRUTH ------------------
    basename = os.path.splitext(os.path.basename(args.image_path))[0]
    gt_path = os.path.join(CFG.dataset_root, "test", "masks", basename + ".png")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth mask not found: {gt_path}")
    gt_mask = np.array(Image.open(gt_path))

    # ------------------ METRICS ------------------
    acc = accuracy_score(gt_mask.flatten(), pred_mask.flatten())
    prec = precision_score(gt_mask.flatten(), pred_mask.flatten(), average="macro", zero_division=0)
    rec = recall_score(gt_mask.flatten(), pred_mask.flatten(), average="macro", zero_division=0)
    f1 = f1_score(gt_mask.flatten(), pred_mask.flatten(), average="macro", zero_division=0)
    iou = iou_score(torch.tensor(pred_mask)[None, ...], torch.tensor(gt_mask)[None, ...], CFG.num_classes)

    print("\n[Evaluation Results]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")

    # ------------------ SAVE VISUALS ------------------
    palette = load_palette_from_csv(os.path.join(CFG.dataset_root, CFG.label_csv))
    out_dir = os.path.join("outputs", "single_eval")
    os.makedirs(out_dir, exist_ok=True)
    save_mask(pred_mask, os.path.join(out_dir, f"{basename}_mask.png"), palette)
    save_overlay(image_tensor[0], torch.tensor(pred_mask), os.path.join(out_dir, f"{basename}_overlay.png"), palette)

    # ------------------ TEMPERATURE EXTRACTION ------------------
    fie = FlirImageExtractor()
    fie.extracted_metadata = fie.extract_metadata(args.image_path)
    fie.updated_metadata = fie.extracted_metadata
    fie.process_image(args.image_path)

    thermal_np = fie.get_thermal_np()

    # Align mask to thermal size if needed
    if pred_mask.shape != thermal_np.shape:
        pred_mask_resized = cv2.resize(pred_mask, (thermal_np.shape[1], thermal_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        pred_mask_resized = pred_mask

    sunlit_temps = thermal_np[pred_mask_resized == 1]  # class 1 = sunlit leaf
    if sunlit_temps.size > 0:
        print("\n[Temperature Extraction]")
        print(f"Average sunlit temp: {np.mean(sunlit_temps):.2f} °C")
        print(f"Min sunlit temp:     {np.min(sunlit_temps):.2f} °C")
        print(f"Max sunlit temp:     {np.max(sunlit_temps):.2f} °C")
    else:
        print("[WARN] No sunlit pixels found.")

if __name__ == "__main__":
    main()
