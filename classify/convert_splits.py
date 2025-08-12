import os
import shutil

# Go one level up from the classify/ folder
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Classes
classes = ["tomato", "pistachio", "almond", "grape", "orange"]

# Splits
splits = ["train", "val", "test"]

# Output classification dataset folder inside classify/
target_root = os.path.join(repo_root, "classify", "cls_data")

for split in splits:
    for cls in classes:
        src_dir = os.path.join(repo_root, cls, split)
        dst_dir = os.path.join(target_root, split, cls)

        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            print(f"⚠️ Skipping {src_dir} (does not exist)")
            continue

        for fname in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)

            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)

print(f"✅ Classification dataset created at: {target_root}")
