import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
from config import CFG
from utils.labels import load_class_map

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split="train", label_csv="class_dict.csv", transform=None):
        self.transform = transform
        self.split = split
        self.samples = []

        # allow root_dir to be either a string or a list of folders
        if isinstance(root_dir, str):
            root_dirs = [root_dir]
        else:
            root_dirs = root_dir

        # load class map from the first dataset
        label_csv_path = os.path.join(root_dirs[0], label_csv)
        self.class_map, self.class_names = load_class_map(label_csv_path)

        for one_root in root_dirs:
            image_dir = os.path.join(one_root, split)
            mask_dir = os.path.join(one_root, f"{split}_labels")

            image_filenames = sorted(os.listdir(image_dir))
            mask_filenames = sorted(os.listdir(mask_dir))

            assert len(image_filenames) == len(mask_filenames), \
                f"{split} set mismatch in {one_root}: {len(image_filenames)} images, {len(mask_filenames)} masks"

            for img_name, mask_name in zip(image_filenames, mask_filenames):
                self.samples.append((
                    os.path.join(image_dir, img_name),
                    os.path.join(mask_dir, mask_name),
                ))

        print(f"Log: Loaded {len(self.samples)} samples for {split} from {root_dirs}")
        # print(f"Log:   Samples: {self.samples[:1]} ... {self.samples[-1:]}")

        # self.image_dir = os.path.join(root_dir, split)
        # self.mask_dir  = os.path.join(root_dir, f"{split}_labels")
        # self.image_filenames = sorted(os.listdir(self.image_dir))
        # self.mask_filenames  = sorted(os.listdir(self.mask_dir))

        # # Load class map from CSV
        # label_csv_path = os.path.join(root_dir, label_csv)
        # self.class_map, self.class_names = load_class_map(label_csv_path)

        # assert len(self.image_filenames) == len(self.mask_filenames), \
        #     f"{split} set mismatch: {len(self.image_filenames)} images, {len(self.mask_filenames)} masks"

    def _convert_mask(self, mask):
        """
        Convert RGB mask to 2D class index mask using self.class_map.
        Any unknown colors are mapped to background (0).
        """
        mask = np.array(mask)
        h, w, _ = mask.shape
        label_mask = np.zeros((h, w), dtype=np.int64)

        # Fill in known colors
        for color, class_id in self.class_map.items():
            matches = np.all(mask == color, axis=-1)
            label_mask[matches] = class_id

        # Clamp out-of-range values just in case
        label_mask[label_mask < 0] = 0
        label_mask[label_mask >= CFG.num_classes] = 0

        return torch.tensor(label_mask, dtype=torch.long)

    def __len__(self):

        return len(self.samples)
        # return len(self.image_filenames)

    def __getitem__(self, idx):
        
        image_path, mask_path = self.samples[idx]
        # image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = image.resize(CFG.image_size)

        # Load and convert mask
        # mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = Image.open(mask_path).convert("RGB")  # RGB mask w/ color codes
        mask = mask.resize(CFG.image_size, resample=Image.NEAREST)
        mask = self._convert_mask(mask)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        return image, mask
