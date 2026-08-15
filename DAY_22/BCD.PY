from PIL import Image
import matplotlib.pyplot as plt
import os

# Paths to sample images (update with your actual file names)
normal_path = "/workspaces/project_code/DAY_22/normal_sample.jpg"
benign_path = "/workspaces/project_code/DAY_22/benign_sample.jpg"
malignant_path = "/workspaces/project_code/DAY_22/malignant_sample.jpg"

# Load images
normal_img = Image.open(normal_path)
benign_img = Image.open(benign_path)
malignant_img = Image.open(malignant_path)

# Display side by side
fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].imshow(normal_img); axes[0].set_title("Normal Tissue"); axes[0].axis("off")
axes[1].imshow(benign_img); axes[1].set_title("Benign Tumor"); axes[1].axis("off")
axes[2].imshow(malignant_img); axes[2].set_title("Malignant Tumor"); axes[2].axis("off")
plt.show()

# Explore image properties
print("Normal Image Size (Resolution):", normal_img.size)   # (width, height)
print("Benign Image Size (Resolution):", benign_img.size)
print("Malignant Image Size (Resolution):", malignant_img.size)

print("Normal Image Mode (Color Channels):", normal_img.mode)   # e.g., 'RGB' or 'L'
print("Benign Image Mode:", benign_img.mode)
print("Malignant Image Mode:", malignant_img.mode)

print("Normal Image Format:", normal_img.format)   # e.g., JPEG, PNG
print("Benign Image Format:", benign_img.format)
print("Malignant Image Format:", malignant_img.format)
