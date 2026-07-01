import os
import shutil
import random

base_dir = "dataset/test"
val_dir = "dataset/Train"

classes = ["Healthy", "Rust", "Phoma", "Miner"]

split_ratio = 0.2  # 20% validation

for cls in classes:
    os.makedirs(f"{val_dir}/{cls}", exist_ok=True)

    images = os.listdir(f"{base_dir}/{cls}")
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)

    val_images = images[:split_idx]

    for img in val_images:
        src = f"{base_dir}/{cls}/{img}"
        dst = f"{val_dir}/{cls}/{img}"
        shutil.move(src, dst)

print("Validation split created!")