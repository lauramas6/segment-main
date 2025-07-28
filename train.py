import argparse
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from config import CFG
from models.factory import get_model
from utils.dataloader import get_loaders
from utils.metrics import dice_coef, iou_score  # Stub in metrics.py
import numpy as np
from tqdm import tqdm

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Optional: makes TF stop printing stuff

# ------------------ CLI ARGUMENTS ------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default=CFG.architecture)
    parser.add_argument("--model_name", type=str, default=CFG.model_name)
    parser.add_argument("--data_root", type=str, default=CFG.dataset_root)
    parser.add_argument("--label_csv", type=str, default=CFG.label_csv)
    return parser.parse_args()

args = parse_args()
CFG.architecture = args.architecture
CFG.model_name = args.model_name
CFG.dataset_root = args.data_root
CFG.label_csv = args.label_csv

# ------------------ SETUP ------------------

torch.manual_seed(CFG.seed)
os.makedirs(CFG.output_dir, exist_ok=True)

device = CFG.device
model = get_model().to(device)

train_loader, val_loader = get_loaders(CFG.dataset_root, CFG.label_csv)

optimizer = AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
ce_loss = nn.CrossEntropyLoss(ignore_index=CFG.ignore_index)

def loss_fn(pred, target):
    ce = ce_loss(pred, target)
    if CFG.use_dice_loss:
        dice = dice_coef(pred, target, num_classes=CFG.num_classes)
        return ce + CFG.dice_weight * (1 - dice)
    return ce

# ------------------ TRAINING ------------------

best_val_loss = float("inf")

for epoch in range(CFG.epochs):
    model.train()
    train_loss = []

    for images, masks in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG.epochs}] Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)

    # ------------------ VALIDATION ------------------
    if (epoch + 1) % CFG.val_every == 0:
        model.eval()
        val_loss = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images).logits
                loss = loss_fn(outputs, masks)
                val_loss.append(loss.item())

        avg_val_loss = np.mean(val_loss)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CFG.output_dir, f"{CFG.project_name}_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Checkpoint] Saved best model to {checkpoint_path}")
