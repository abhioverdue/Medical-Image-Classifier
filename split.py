import os
import shutil
import random
from collections import defaultdict

def split_data(
    src_dir="data/all",        # where current folders are
    dest_dir="data",           # where you want train/val/test
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    random.seed(seed)

    # Avoid overwriting
    if os.path.exists(os.path.join(dest_dir, "train")):
        print(f"⚠️ '{dest_dir}/train' already exists. Remove or rename it first.")
        return

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

    summary = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    # Go through each class folder
    for cls in os.listdir(src_dir):
        cls_path = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"⚠️ No images found in {cls_path}, skipping.")
            continue

        random.shuffle(images)
        n_total = len(images)
        n_train = max(1, int(train_ratio * n_total))
        n_val = max(1, int(val_ratio * n_total))
        n_test = n_total - n_train - n_val
        if n_test == 0 and n_total >= 3:  # ensure test has at least 1
            n_test = 1
            n_train -= 1

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }

        # Copy into split folders
        for split, files in splits.items():
            split_cls_dir = os.path.join(dest_dir, split, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(cls_path, f), os.path.join(split_cls_dir, f))
            summary[cls][split] = len(files)

        print(f"✅ {cls}: {n_total} → train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Print summary table
    print("\n📊 Final Split Summary:")
    print(f"{'Class':<15}{'Train':<10}{'Val':<10}{'Test':<10}{'Total':<10}")
    print("-"*55)
    for cls, counts in summary.items():
        total = counts['train'] + counts['val'] + counts['test']
        print(f"{cls:<15}{counts['train']:<10}{counts['val']:<10}{counts['test']:<10}{total:<10}")

if __name__ == "__main__":
    split_data()
