import os
import timm
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.measure import label
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

# Step 1: Load the DinoV2 model
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
fig, axes = plt.subplots(num_images, 3, figsize=(15, num_images * 5))

for idx, image_path in enumerate(image_paths):

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(input_tensor)

    feature_map = features[-1][0]  # Shape: (C, H, W)

    # Step 4: Flatten features (C, H, W) -> (H*W, C)
    flattened_features = feature_map.permute(
        1, 2, 0).reshape(-1, feature_map.shape[0])

    sum_of_squared_distances = []
    silhouette_scores = []
    cluster_range = range(2, 10)

    for num_clusters in cluster_range:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
        kmeans.fit(flattened_features.cpu().numpy())

        sum_of_squared_distances.append(kmeans.inertia_)

        cluster_labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(
            flattened_features.cpu().numpy(), cluster_labels))

    # Choose the optimal number of clusters
    optimal_clusters = cluster_range[silhouette_scores.index(
        max(silhouette_scores))]
    print(f"Optimal number of clusters for {image_path}: {optimal_clusters}")

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, n_init="auto")
    clusters = kmeans.fit_predict(flattened_features.cpu().numpy())

    cluster_map = clusters.reshape(feature_map.shape[1], feature_map.shape[2])
    final_colors = plt.colormaps["tab10"]

    normalized_cluster_map = cluster_map / np.max(cluster_map)
    segmented_image = final_colors(normalized_cluster_map)[..., :3]

    # Calculate the cluster proportions
    cluster_areas = np.bincount(cluster_map.ravel())
    total_pixels = cluster_map.size

    # Define a threshold for background clusters (covering >30% of the image)
    background_threshold = 0.3 * total_pixels

    # Identify background clusters
    background_clusters = [i for i, area in enumerate(
        cluster_areas) if area > background_threshold]

    filtered_cluster_map = np.copy(cluster_map)
    for background in background_clusters:
        filtered_cluster_map[filtered_cluster_map ==
                             background] = -1  # Set background to -1

    # Filter small clusters
    binary_mask = filtered_cluster_map != -1
    filtered_mask = remove_small_objects(binary_mask, min_size=50)
    filtered_mask = binary_fill_holes(filtered_mask)

    final_cluster_map = label(filtered_mask)

    # Ensure no division by zero
    max_value = np.max(final_cluster_map)
    if max_value > 0:
        normalized_final_cluster_map = final_cluster_map / \
            max_value
        segmented_filtered_image = final_colors(
            normalized_final_cluster_map)[..., :3]
    else:
        segmented_filtered_image = np.zeros_like(
            final_cluster_map, dtype=np.float32)

    # Visualization
    axes[idx, 0].imshow(image)
    axes[idx, 0].axis("off")

    axes[idx, 1].imshow(segmented_image)
    axes[idx, 1].axis("off")

    axes[idx, 2].imshow(segmented_filtered_image)
    axes[idx, 2].axis("off")

plt.tight_layout()
plt.show()
