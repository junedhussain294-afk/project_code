import os
import zipfile
import requests
from PIL import Image

# 1. Download BUSI dataset (hosted on Kaggle or other sources)
# For demo, let's assume you already have BUSI dataset zip file locally.
# If not, you can manually download from Kaggle: "Breast Ultrasound Images (BUSI)"

dataset_path = "C:\Users\juned\Downloads\archive.zip\jpeg"  # update with your folder path

# 2. Analyze dataset
classes = os.listdir(dataset_path)
print("Classes found:", classes)

num_images = 0
image_sizes = []
formats = set()

for cls in classes:
    cls_path = os.path.join(dataset_path, cls)
    for file in os.listdir(cls_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            num_images += 1
            img = Image.open(os.path.join(cls_path, file))
            image_sizes.append(img.size)  # (width, height)
            formats.add(img.format)

print("\nTotal number of images:", num_images)
print("Classes:", classes)
print("Unique image sizes:", set(image_sizes))
print("File formats:", formats)
