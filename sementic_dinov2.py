import os
import timm
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# DinoV2 model
model_name = "vit_large_patch14_dinov2"
model = timm.create_model(model_name, pretrained=True, features_only=True)
model.eval()

transform = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

image_folder = "images"
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(
    image_folder) if fname.lower().endswith(('jpg', 'jpeg', 'png'))]

num_images = len(image_paths)
fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 6))

for idx, image_path in enumerate(image_paths):
    print(f"Processing image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(input_tensor)

    feature_map = features[-1][0]  # Shape: (C, H, W)
    print(f"Feature map shape for {image_path}: {feature_map.shape}")

    # Flatten the features for clustering (C, H, W) -> (H*W, C)
    flattened_features = feature_map.permute(
        1, 2, 0).reshape(-1, feature_map.shape[0])

    # Apply k-means clustering
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
    # Cluster assignment for each patch
    clusters = kmeans.fit_predict(flattened_features.cpu().numpy())

    cluster_map = clusters.reshape(feature_map.shape[1], feature_map.shape[2])

    cluster_colors = plt.colormaps["tab10"]
    segmented_image = cluster_colors(
        cluster_map / num_clusters)[..., :3]  # Remove alpha channel

    axes[idx, 0].imshow(image)
    axes[idx, 0].axis("off")

    axes[idx, 1].imshow(segmented_image)
    axes[idx, 1].axis("off")

plt.tight_layout()
plt.show()
