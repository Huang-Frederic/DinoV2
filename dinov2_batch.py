import torch
import timm
from safetensors.torch import load_file
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

model_path = "models/vit_large_patch14_dinov2.safetensors"
model_weights = load_file(model_path)

model = timm.create_model("vit_large_patch14_dinov2",
                          pretrained=False, features_only=True)
model.load_state_dict(model_weights, strict=False)
model.eval()

transform = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

image_path = "images/Jet.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)
input_tensor.shape


with torch.no_grad():
    features = model(input_tensor)

feature_map = features[-1][0]
print("Shape of selected feature map:", feature_map.shape)

feature_map = feature_map[0].cpu().numpy()
feature_map = (feature_map - feature_map.min()) / \
    (feature_map.max() - feature_map.min())


image_resized = image.resize((feature_map.shape[1], feature_map.shape[0]))
image_np = np.array(image_resized)

if len(image_np.shape) == 2:
    image_np = np.stack([image_np] * 3, axis=-1)

image_np = image_np / 518.0

superposed = 0.8 * image_np + 0.2 * plt.cm.jet(feature_map)[..., :3]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(feature_map, cmap="jet")
axes[1].set_title("Feature Map")
axes[1].axis("off")

axes[2].imshow(superposed)
axes[2].set_title("Superposition")
axes[2].axis("off")

plt.show()
