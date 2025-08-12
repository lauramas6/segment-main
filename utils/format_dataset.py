import argparse
from pathlib import Path
from PIL import Image
import shutil
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.model_zoo import MODEL_ZOO

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def resize_and_copy_image(src_path: Path, dst_path: Path, size):
    try:
        img = Image.open(src_path)
        interp = Image.NEAREST if 'label' in str(src_path).lower() else Image.BILINEAR
        img = img.resize(size, interp)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
    except Exception as e:
        print(f"[!] Failed to resize {src_path}: {e}")

def main(data_root, arch):
    data_root = Path(data_root).resolve()
    arch = arch.lower()

    if arch not in MODEL_ZOO:
        raise ValueError(f"[!] Unknown architecture: {arch}")

    image_size = MODEL_ZOO[arch].get("image_size", (512, 512))
    output_dir = data_root / arch

    if output_dir.exists():
        print(f"[!] Output directory already exists: {output_dir}")
    else:
        print(f"[+] Creating formatted dataset at: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Copy and resize each image/mask directory
    for subfolder in ["train", "train_labels", "val", "val_labels", "test", "test_labels"]:
        src_folder = data_root / subfolder
        dst_folder = output_dir / subfolder

        if not src_folder.exists():
            print(f"[!] Skipping missing folder: {src_folder}")
            continue

        for src_file in src_folder.glob("*"):
            if src_file.suffix.lower() in VALID_IMAGE_EXTS:
                dst_file = dst_folder / src_file.name
                resize_and_copy_image(src_file, dst_file, image_size)

    # Copy class_dict.csv as-is
    class_csv = data_root / "class_dict.csv"
    if class_csv.exists():
        shutil.copy(class_csv, output_dir / "class_dict.csv")

    print(f"[✓] All data resized to {image_size} and saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to the dataset folder (e.g., data/tomato)")
    parser.add_argument("--architecture", type=str, required=True, help="Architecture name (e.g., segformer, setr, mask2former)")
    args = parser.parse_args()

    main(args.data_root, args.architecture)
