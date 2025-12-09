import os
import shutil
import random

for folder in ['train/images', 'train/labels', 'valid/images', 'valid/labels']:
    os.makedirs(f"dataset_final/{folder}", exist_ok=True)

images = [f for f in os.listdir("dataset/images") if f.endswith(".png")]
random.shuffle(images)

# 80/20 split
split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def move_files(file_list, type_split):
    for img in file_list:
        label = img.replace(".png", ".txt")
        shutil.copy(f"dataset/images/{img}", f"dataset_final/{type_split}/images/{img}")
        if os.path.exists(f"dataset/labels/{label}"):
            shutil.copy(f"dataset/labels/{label}", f"dataset_final/{type_split}/labels/{label}")

move_files(train_imgs, "train")
move_files(val_imgs, "valid")
print("Dataset organizado em 'dataset_final'!")