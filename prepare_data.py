import os
import random
import shutil
import kagglehub

# Download dataset (~430 MB)
base_path = kagglehub.dataset_download("ciplab/real-and-fake-face-detection")

# Find training_fake and training_real folders
source_dirs = {}
for root, dirs, _ in os.walk(base_path):
    if 'training_fake' in dirs and 'training_real' in dirs:
        source_dirs['fake'] = os.path.join(root, 'training_fake')
        source_dirs['real'] = os.path.join(root, 'training_real')
        break

# Shuffle and move into 70% train / 15% val / 15% test
random.seed(42)
for label, src_dir in source_dirs.items():
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)

    n = len(files)
    splits = {
        'train': files[:int(0.7 * n)],
        'val': files[int(0.7 * n):int(0.85 * n)],
        'test': files[int(0.85 * n):]
    }

    for split_name, split_files in splits.items():
        dest_dir = os.path.join('data', split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        for f in split_files:
            shutil.move(os.path.join(src_dir, f), os.path.join(dest_dir, f))

print("Dataset ready in data/train, data/val, and data/test.")