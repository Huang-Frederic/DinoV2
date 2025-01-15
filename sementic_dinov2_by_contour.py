import os
import timm
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.measure import label


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
fig, axes = plt.subplots(num_images, 3, figsize=(15, num_images * 5))

for idx, image_path in enumerate(image_paths):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(input_tensor)

    feature_map = features[-1][0]  # Shape: (C, H, W)

    # Flatten features (C, H, W) -> (H*W, C)
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

    optimal_clusters = cluster_range[silhouette_scores.index(
        max(silhouette_scores))]
    print(f"Optimal number of clusters for {image_path}: {optimal_clusters}")

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, n_init="auto")
    clusters = kmeans.fit_predict(flattened_features.cpu().numpy())

    cluster_map = clusters.reshape(feature_map.shape[1], feature_map.shape[2])

    cluster_colors = plt.cm.get_cmap("tab10", optimal_clusters)
    segmented_image = cluster_colors(cluster_map / optimal_clusters)[..., :3]

    # Parameters for edge detection
    edge_detection_method = "canny"  # Choose between "canny" or "sobel"

    # Preprocess cluster_map for contour analysis
    filtered_cluster_map = np.copy(cluster_map)
    unique_clusters = np.unique(cluster_map)

    # Mask to store significant clusters
    significant_clusters_mask = np.zeros_like(cluster_map, dtype=bool)

    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip background
            continue

        # Create a binary mask for the current cluster
        cluster_mask = (cluster_map == cluster_id)

        # Smooth the mask to remove noise
        smoothed_mask = gaussian(cluster_mask.astype(float), sigma=2)

        # Apply edge detection
        if edge_detection_method == "canny":
            edges = canny(smoothed_mask, sigma=1.5)
        else:
            raise ValueError("Unsupported edge detection method!")

        # Compute the number of edge pixels and cluster size
        edge_count = np.sum(edges)
        cluster_size = np.sum(cluster_mask)

        # Determine if the cluster is significant based on edge count
        if edge_count > 50 and cluster_size > 200:  # Adjust thresholds as needed
            significant_clusters_mask |= cluster_mask
        else:
            filtered_cluster_map[cluster_map ==
                                 cluster_id] = -1  # Mark as background

    # Relabel the significant clusters
    final_cluster_map = label(significant_clusters_mask)

    # Map the filtered clusters to an image
    final_colors = plt.cm.get_cmap("tab10", np.max(final_cluster_map) + 1)
    segmented_filtered_image = final_colors(
        final_cluster_map / np.max(final_cluster_map))[..., :3]

    # Visualization for each image
    axes[idx, 0].imshow(image)
    axes[idx, 0].axis("off")

    axes[idx, 1].imshow(segmented_image)
    axes[idx, 1].axis("off")

    axes[idx, 2].imshow(segmented_filtered_image)
    axes[idx, 2].axis("off")

plt.tight_layout()
plt.show()
