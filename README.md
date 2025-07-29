# 🧠 Semantic Segmentation Training Suite

A modular framework for training and evaluating semantic segmentation models using transformer-based architectures.

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

| Argument         | Description                                  | Example                                 |
|------------------|----------------------------------------------|-----------------------------------------|
| `--architecture` | Model architecture to use                    | `segformer`, `mask2former`, `setr`      |
| `--data_root`    | Path to dataset root folder                  | `tomato/segformer`                      |
| `--weights`      | (Evaluation only) Path to model weights `.pt` file | `weights/best_model.pt`           |

If needed, use the dataset formatting utility to prepare your dataset directory structure and ensure compatibility with the model architecture:

```bash
python3 utils/format_dataset.py --data_root tomato --architecture setr
```

