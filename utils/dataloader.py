import os
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataset import SegmentationDataset
from config import CFG

def get_transforms():
    """Basic image transform. Expand here if needed."""
    return T.Compose([
        T.ToTensor()
    ])

def get_loaders(data_root="data", label_csv="class_dict.csv", include_test_only=False):
    """
    Returns train, val, and optionally test DataLoaders.

    Args:
        data_root (str): root directory of dataset (with train/val/test subfolders)
        label_csv (str): name of label CSV (inside data_root)
        include_test (bool): whether to return test_loader

    Returns:
        loaders: tuple of train_loader, val_loader[, test_loader]
    """
    transform = get_transforms()


    # take train/val from all plants other than dataset_root and test from dataset_root 
    if CFG.multi_plant:
        
        train_plants = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]
        # print(f"Log: All plants found: {train_plants}")

        train_plants.remove(data_root)
        # print(f"Log: Test plant: {data_root}")
        # print(f"Log: Train/Val plants: {train_plants}")


        train_roots = [os.path.join("data", p) for p in train_plants]
        test_root = os.path.join("data", data_root)

        # print(f"Log: Train roots: {train_roots}")
        # print(f"Log: Test root: {test_root}")

        # for evaluate_multiclass.py
        if include_test_only:
            test_set = SegmentationDataset(test_root, split="test", label_csv=label_csv, transform=transform)
            test_loader = DataLoader(
                test_set,
                batch_size=CFG.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            return test_loader


        train_set = SegmentationDataset(train_roots, split="train", label_csv=label_csv, transform=transform)
        val_set   = SegmentationDataset(train_roots, split="val",   label_csv=label_csv, transform=transform)

    # normal case: train/val/test from dataset_root
    else:
        if include_test_only:
            test_set = SegmentationDataset(data_root, split="test", label_csv=label_csv, transform=transform)
            test_loader = DataLoader(
                test_set,
                batch_size=CFG.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            return test_loader

        train_set = SegmentationDataset(data_root, split="train", label_csv=label_csv, transform=transform)
        val_set   = SegmentationDataset(data_root, split="val",   label_csv=label_csv, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_set,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader
