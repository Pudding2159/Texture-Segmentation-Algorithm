import cv2
from skimage import filters, feature
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.segmentation import slic
from skimage.filters import sobel
from skimage.color import label2rgb
import os
os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.cluster import KMeans


def Segmentation_ALG1(IMAGE_FILE,size):
    from sklearn.metrics import silhouette_score
    Region_Size = size
    #Определение оптимального количества кластеров
    def calculate_optimal_clusters(X, max_clusters=10):
        wcss = []
        silhouette_scores = []
        for i in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, n_init='auto').fit(X)
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))

        return np.argmax(silhouette_scores) + 3  # +2 потому что индексация начинается с 2 кластеров


    def Mask_create(image,clustering,Region_Size):
        markers = np.zeros_like(image, dtype=int)
        cluster_centers = np.round(clustering.cluster_centers_).astype(int)
        for i, (x, y) in enumerate(cluster_centers, start=1):
            markers[y:y+Region_Size, x:x+Region_Size] = i
        image = (image * 255).astype(np.uint8)
        markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)

        boundary_mask = markers == -1

        # Display results
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(image, cmap='gray')
        ax.imshow(boundary_mask, alpha=1, cmap='jet')  # Display the boundary mask
        plt.savefig('./boundary_mask.png', bbox_inches='tight', pad_inches=0)
        plt.show()


    processed_image = cv2.equalizeHist(cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE))
    processed_image = (filters.gaussian(processed_image, sigma=0.5) * 255).astype(np.uint8)
    processed_image = cv2.resize(processed_image, (
    processed_image.shape[1] // Region_Size * Region_Size, processed_image.shape[0] // Region_Size * Region_Size))

    plt.imshow(processed_image, cmap='gray')
    plt.show()

    # Calculating texture features for each imageFragments
    fragmentCoords = np.array([(x, y) for y in range(0, processed_image.shape[0], Region_Size) for x in
                             range(0, processed_image.shape[1], Region_Size)])
    imageFragments = np.array([processed_image[y:y + Region_Size, x:x + Region_Size] for (x, y) in fragmentCoords])

   #GLCM computation
    texture_features = []

    for Fragm in imageFragments:
        glcm = feature.graycomatrix(Fragm, distances=[3, 1], angles=[np.deg2rad(angle) for angle in [0, 45, 90, 160]], levels=256,
                                    normed=True)
        patch_features = {
            'dissimilarity': feature.graycoprops(glcm, 'dissimilarity').mean(),
            'correlation': feature.graycoprops(glcm, 'correlation').mean()
        }
        texture_features.append(patch_features)

    # Convert dictionary lists to NumPy arrays for each characteristic
    dissimilarity_ = np.array([feature['dissimilarity'] for feature in texture_features])
    correlation_ = np.abs(np.array([feature['correlation'] for feature in texture_features]))

    # Creating a condition for filtering
    dissimilarity_Filter = 40
    correlation_Filter = 0.9
    condition = np.logical_and(dissimilarity_ < dissimilarity_Filter, correlation_ > correlation_Filter)
    filteredPatches = fragmentCoords[condition]

    optimal_clusters = calculate_optimal_clusters(filteredPatches)
    print("Počet definovaných klastrov   ",optimal_clusters)
    CLUSTERS = optimal_clusters

    # KMeans Clustering
    clustering = KMeans(n_clusters=CLUSTERS, n_init='auto')
    clustering.fit(filteredPatches)
    Mask_create(processed_image,clustering,Region_Size)

    grayscale_palette = [
        "#B0B5B3",
        "#949997",
        "#818583",
        "#6A6E6C",
        "#545756",
        "#434544",
    ]

    # Visualization of the clusters
    for cluster_index in range(CLUSTERS):
        # Determining the mask for the current cluster
        is_current_cluster = clustering.labels_ == cluster_index
        current_cluster_patches = filteredPatches[is_current_cluster]

        # Selecting a color for the cluster from the palette
        cluster_color = grayscale_palette[cluster_index % len(grayscale_palette)]

        # Plotting the cluster
        plt.scatter(current_cluster_patches[:, 0], current_cluster_patches[:, 1], marker='o', color=cluster_color)

    # Preparing data for watershed segmentation
    marker_image = np.zeros_like(processed_image, dtype=int)
    rounded_cluster_centers = np.round(clustering.cluster_centers_).astype(int)

    # Creating markers for watershed
    for marker_index, (center_x, center_y) in enumerate(rounded_cluster_centers, start=1):
        marker_image[center_y:center_y + Region_Size, center_x:center_x + Region_Size] = marker_index

    # Applying watershed to the image
    processed_image_uint8 = (processed_image * 255).astype(np.uint8)
    colored_image = cv2.cvtColor(processed_image_uint8, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(colored_image, marker_image)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(processed_image, cmap='gray')
    ax.imshow(markers, alpha=1, cmap='gray')
    plt.savefig('./Segmented_image.png', bbox_inches='tight', pad_inches=0)
    plt.show()


    def detect_texture_edges(segments, processed_image):
        edge_segments = np.zeros_like(processed_image)
        for segment_value in np.unique(segments):
            mask = segments == segment_value
            edges = sobel(mask)
            edge_intensity = np.mean(processed_image[edges > 0])
            edge_segments[edges > 0] = edge_intensity
        return edge_segments

    processed_image = imread(IMAGE_FILE, as_gray=True)
    segments = slic(processed_image, n_segments=500, compactness=1, channel_axis=None)
    texture_edges = detect_texture_edges(segments, processed_image)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(label2rgb(segments, processed_image, kind='avg'), cmap='gray')
    plt.imshow(texture_edges, cmap='hot', alpha=0.5)  # Boundary overlap
    plt.title('Texture Edges Highlighted')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    plt.show()




