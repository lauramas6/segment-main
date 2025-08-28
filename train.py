import argparse
import os
import subprocess
import torch
import torch.nn as nn
from torch.optim import AdamW
from config import CFG
from models.factory import get_model
from utils.dataloader import get_loaders
from utils.metrics import dice_coef, iou_score
import numpy as np
from tqdm import tqdm
from utils.helpers import get_logits
from utils.cli import parse_args
import sys
import warnings
warnings.filterwarnings("ignore", message=".*NCCL.*")

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------ CLI ARGUMENTS ------------------

## Parse command line arguments and override CFG defaults
args = parse_args()

# Core
CFG.architecture = args.architecture
CFG.model_name = args.model_name
CFG.dataset_root = args.data_root
CFG.label_csv = args.label_csv

# Model-related
CFG.in_channels = args.in_channels
CFG.num_classes = args.num_classes
CFG.freeze_encoder = args.freeze_encoder   # bool flag
CFG.use_dice_loss = args.use_dice_loss     # bool flag
CFG.dice_weight = args.dice_weight

# Training
CFG.epochs = args.epochs
CFG.batch_size = args.batch_size
CFG.learning_rate = args.learning_rate
CFG.weight_decay = args.weight_decay
CFG.val_every = args.val_every
CFG.patience = args.patience

# Evaluation / Logging
CFG.save_best_only = args.save_best_only   # bool flag
CFG.num_eval_samples = args.num_eval_samples
CFG.show_sample_predictions = args.show_sample_predictions   # bool flag
CFG.weights = args.weights


# ------------------ FORMAT DATASET CHECK ------------------

## Each model (CNN/ViT etc.) should have a configureation saved the the model_zoo.py
## Whatever your dataset root is, this patch of code will create a subfolder in it for the specific model architecture you are running
## (In case of special formatting needs for individual architectures)
## !!! This can eat up a lot of memory as it will copy all of your images/masks into the subfolder after reformatting !!!

formatted_dataset_path = os.path.join(CFG.dataset_root, CFG.architecture)
if not os.path.exists(formatted_dataset_path):
    print(f"[INFO] Formatted dataset not found at {formatted_dataset_path}.")
    print("[INFO] Running format_dataset.py...")
    subprocess.run([
    sys.executable, "utils/format_dataset.py",
    "--data_root", CFG.dataset_root,
    "--architecture", CFG.architecture
    ], check=True)
else:
    print(f"[INFO] Found formatted dataset at {formatted_dataset_path}, skipping formatting.")

# ------------------ SETUP ------------------
torch.manual_seed(CFG.seed)
os.makedirs(CFG.output_dir, exist_ok=True)

device = CFG.device
model = get_model()

# Enable multi-GPU if available
if torch.cuda.device_count() > 1:
    print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

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

## For tracking best val loss and patience counter for early stopping 
best_val_loss = float("inf")
no_improve_counter = 0

## 
dataset_name = os.path.basename(os.path.normpath(CFG.dataset_root))
ckpt_dir = os.path.join(CFG.output_dir, dataset_name, CFG.architecture, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

for epoch in range(CFG.epochs):
    model.train()
    train_loss = []

    for images, masks in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG.epochs}] Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = get_logits(model(images))
        outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)

    if (epoch + 1) % CFG.val_every == 0:
        model.eval()
        val_loss = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images, masks = images.to(device), masks.to(device)
                outputs = get_logits(model(images))
                outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss_fn(outputs, masks)
                val_loss.append(loss.item())

        avg_val_loss = np.mean(val_loss)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_counter = 0
            checkpoint_path = os.path.join(ckpt_dir, f"{dataset_name}_{CFG.architecture}_best.pt")
            torch.save(
                model.state_dict(),
                checkpoint_path)
            print(f"[Checkpoint] Saved best model to {checkpoint_path}")
        else:
            no_improve_counter += 1
            print(f"[Early Stop] No improvement for {no_improve_counter}/{CFG.patience} rounds")

        if no_improve_counter >= CFG.patience:
            print("[Early Stop] Validation loss did not improve, stopping training.")
            break
