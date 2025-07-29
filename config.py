from types import SimpleNamespace
from models.model_zoo import MODEL_ZOO
import torch

arch = "segformer"
defaults = MODEL_ZOO[arch]

CFG = SimpleNamespace(
    # General
    project_name = "segmentation-pipeline",
    seed = 42,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    # Model
    architecture = arch,
    model_name = defaults["default_model"],
    num_classes = defaults["num_classes"],
    in_channels = defaults["in_channels"],
    trust_remote_code = defaults["trust_remote_code"],
    ignore_index = 255,
    pretrained = True,
    freeze_encoder = False,

    # Input
    image_size = (512, 512),

    # Training
    epochs = 20,
    batch_size = 8,
    learning_rate = 5e-5,
    weight_decay = 1e-4,
    val_every = 1,

    # Loss
    use_dice_loss = False,
    dice_weight = 0.5,

    # Data paths
    dataset_root = "data",
    label_csv = "class_dict.csv",

    # Logging / Outputs
    output_dir = "./results/",
    save_best_only = True,
    log_dir = "./logs/",

    # Evaluation
    show_sample_predictions = True,
    num_eval_samples = 5
)
