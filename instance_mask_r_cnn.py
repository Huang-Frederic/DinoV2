import torch
from torchvision import models, transforms
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Mask R-CNN model
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.maskrcnn_resnet50_fpn(weights=weights)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

image_folder = "images"
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(
    image_folder) if fname.lower().endswith(('jpg', 'jpeg', 'png'))]

num_images = len(image_paths)
fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

for idx, image_path in enumerate(image_paths):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)

    # Process predictions
    masks = prediction[0]['masks']
    scores = prediction[0]['scores']
    threshold = 0.5

    # Create an empty segmentation mask
    segmentation = np.zeros((image.height, image.width), dtype=np.uint8)

    # Combine masks with scores above the threshold
    for mask, score in zip(masks, scores):
        if score > threshold:
            binary_mask = mask[0].cpu().numpy() > 0.5
            segmentation[binary_mask] = 1  # Mark object regions with 1

    # Segmentation mask
    mask_image = np.zeros(
        (segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)

    # Apply colors
    mask_image[segmentation == 1] = [255, 0, 0]
    mask_image[segmentation == 0] = [128, 0, 128]

    # Step 6: Visualize results
    axes[idx, 0].imshow(image)
    axes[idx, 0].axis("off")

    axes[idx, 1].imshow(mask_image)
    axes[idx, 1].axis("off")

plt.tight_layout()
plt.show()
