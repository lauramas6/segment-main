# 🧠 Semantic Segmentation Training Suite

A modular framework for training and evaluating semantic segmentation models using various Convolutional Neural Network and Vision Transformer architectures.

---

### Dataset Requirements (placed within repo root):  
your_dataset/  
├── train/  
├── train_labels/  
├── val/  
├── val_labels/  
├── test/  
├── test_labels/  
└── class_dict.csv  

## 🚀 Getting Started

### Train a Model

```bash
python3 train.py --data_root tomato/segformer --architecture segformer
```

### Evaluate a trained Model

```bash
python3 evaluate.py --data_root tomato/segformer --architecture segformer --weights weights/your_model.pt
```
### Current Compatible Architectures:
Vision Transformers
  - Segformer
  - Mask2Former
  - SETR

Convolutional Neural Networks
  - ...
  - ...

### 🧾 Training / Evaluation Command-Line Arguments

| Argument              | Description                                                                 | Example                                     |
|-----------------------|-----------------------------------------------------------------------------|---------------------------------------------|
| `--architecture`      | Model architecture to use                                                   | `segformer`, `mask2former`, `setr`, `fcn`   |
| `--model_name`        | HF or local model path/name                                                 | `nvidia/segformer-b3-finetuned-ade-512-512` |
| `--data_root`         | Path to dataset root folder                                                 | `tomato/segformer`                          |
| `--label_csv`         | Relative path to class_dict CSV inside dataset                              | `class_dict.csv`                            |
| `--weights`           | (Eval) Path to model weights `.pt` file                                     | `weights/best_model.pt`                     |
| `--in_channels`       | Number of input channels (e.g. RGB=3, RGB+Thermal=4)                         | `3`                                         |
| `--num_classes`       | Number of classes for segmentation output                                   | `2`                                         |
| `--freeze_encoder`    | Whether to freeze the encoder backbone during training                      | `--freeze_encoder`                          |
| `--use_dice_loss`     | Use Dice Loss in addition to CrossEntropy                                   | `--use_dice_loss`                           |
| `--dice_weight`       | Weight for combining Dice loss with CE loss                                 | `0.5`                                       |
| `--epochs`            | Number of training epochs                                                    | `50`                                        |
| `--batch_size`        | Batch size for training/evaluation                                           | `16`                                        |
| `--learning_rate`     | Learning rate for optimizer                                                  | `5e-5`                                      |
| `--weight_decay`      | Weight decay (L2 regularization)                                             | `1e-4`                                      |
| `--val_every`         | Evaluate on validation set every N epochs                                   | `1`                                         |
| `--patience`          | Early stopping patience in epochs                                           | `5`                                         |
| `--save_best_only`    | Only save the best-performing model (lowest val loss)                       | `--save_best_only`                          |
| `--show_sample_predictions` | Save sample visual predictions during evaluation                     | `--show_sample_predictions`                 |
| `--num_eval_samples`  | Number of samples to visualize during evaluation                            | `5`                                         |

If needed, use the dataset formatting utility to prepare your dataset directory structure and ensure compatibility with the model architecture:

```bash
python3 utils/format_dataset.py --data_root tomato --architecture setr
```

