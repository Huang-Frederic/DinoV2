import timm
import os
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

model_name = "vit_large_patch14_dinov2"
model = timm.create_model(model_name, pretrained=True, features_only=True)
model.eval()

image_folder = "images"
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(
    image_folder) if fname.lower().endswith(('jpg', 'jpeg', 'png'))]

transform = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

original_images = []
feature_maps = []
superposed_images = []

for image_path in image_paths:
    print(f"Traitement de l'image : {image_path}")

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(input_tensor)

    feature_map = features[-1][0]
    feature_map = feature_map[0].cpu().numpy()
    feature_map = (feature_map - feature_map.min()) / \
        (feature_map.max() - feature_map.min())

    image_resized = image.resize((feature_map.shape[1], feature_map.shape[0]))
    image_np = np.array(image_resized)

    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)

    image_np = image_np / 518.0

    superposed = 0.9 * image_np + 0.1 * plt.cm.jet(feature_map)[..., :3]

    original_images.append(image)
    feature_maps.append(feature_map)
    superposed_images.append(superposed)

fig, axes = plt.subplots(
    len(image_paths), 3, figsize=(15, len(image_paths) * 5))

if len(image_paths) == 1:
    axes = np.expand_dims(axes, axis=0)

for i in range(len(image_paths)):
    axes[i, 0].imshow(original_images[i])
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(feature_maps[i], cmap="jet")
    axes[i, 1].set_title("Feature Map")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(superposed_images[i])
    axes[i, 2].set_title("Superposition")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
