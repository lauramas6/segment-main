from types import SimpleNamespace
from models.model_zoo import MODEL_ZOO
import torch
from pathlib import Path

arch = "segformer"
defaults = MODEL_ZOO[arch]

CFG = SimpleNamespace(
    # General
    project_name="segmentation-pipeline",
    seed=42,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    # Model
    architecture=arch,
    model_name=defaults["default_model"],
    num_classes=defaults["num_classes"],
    in_channels=defaults["in_channels"],   # will override to 4 if USE_THERMAL later
    trust_remote_code=defaults["trust_remote_code"],
    ignore_index=255,
    pretrained=True,
    freeze_encoder=False,

    # Input
    image_size=defaults["image_size"],

    # Training
    epochs=50,
    batch_size=32,
    learning_rate=5e-5,
    weight_decay=1e-4,
    val_every=1,
    patience=5,

    # Loss
    use_dice_loss=False,
    dice_weight=0.5,

    # Data paths
    dataset_root="tomato",
    label_csv="class_dict.csv",

    # Logging / Outputs
    output_dir="./results/",
    save_best_only=True,
    log_dir="./logs/",

    # Evaluation
    show_sample_predictions=True,
    num_eval_samples=20,

    # Thermal defaults
    USE_THERMAL=False,
    THERM_UPSAMPLE="bilinear",
    THERM_MEAN=0.0,
    THERM_STD=1.0,
    MODALITY_DROPOUT_P=0.0,
)
