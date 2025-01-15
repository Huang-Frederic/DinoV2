import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# DeepLabV3 model pre-trained on COCO
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = models.segmentation.deeplabv3_resnet101(weights=weights)
model.eval()

transform = T.Compose([
    T.Resize(520),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
])

image_folder = "images"
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(
    image_folder) if fname.lower().endswith(('jpg', 'jpeg', 'png'))]

num_images = len(image_paths)
fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

for idx, image_path in enumerate(image_paths):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    # Extract the predicted class for each pixel (argmax across classes)
    output_predictions = output['out'][0].argmax(0).cpu().numpy()

    segmentation = np.where(output_predictions == 0, 0, 1)

    image_resized = image.resize((520, 520), Image.Resampling.LANCZOS)
    image_array = np.array(image_resized)

    mask_image = np.zeros(
        (segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)

    mask_image[segmentation == 0] = [128, 128, 128]
    mask_image[segmentation == 1] = [0, 255, 255]

    axes[idx, 0].imshow(image)
    axes[idx, 0].set_title(f"Original Image: {os.path.basename(image_path)}")
    axes[idx, 0].axis("off")

    axes[idx, 1].imshow(mask_image)
    axes[idx, 1].axis("off")

plt.tight_layout()
plt.show()
