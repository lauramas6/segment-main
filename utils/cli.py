import argparse
from config import CFG

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--multi_plant", action="store_true", default=CFG.multi_plant)

    # Existing core args
    parser.add_argument("--architecture", type=str, default=CFG.architecture)
    parser.add_argument("--model_name", type=str, default=CFG.model_name)
    parser.add_argument("--data_root", type=str, default=CFG.dataset_root)
    parser.add_argument("--label_csv", type=str, default=CFG.label_csv)

    # Model-related
    parser.add_argument("--in_channels", type=int, default=CFG.in_channels)
    parser.add_argument("--num_classes", type=int, default=CFG.num_classes)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--use_dice_loss", action="store_true")
    parser.add_argument("--dice_weight", type=float, default=CFG.dice_weight)

    # Training
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--learning_rate", type=float, default=CFG.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--val_every", type=int, default=CFG.val_every)
    parser.add_argument("--patience", type=int, default=CFG.patience)

    # Evaluation / Logging
    parser.add_argument("--save_best_only", action="store_true")
    parser.add_argument("--num_eval_samples", type=int, default=CFG.num_eval_samples)
    parser.add_argument("--show_sample_predictions", action="store_true", default=True)

    # Weights (defer to evaluate/train if None)
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (.pt). If not provided, auto-built dynamically in train/evaluate."
    )


    return parser.parse_args()
