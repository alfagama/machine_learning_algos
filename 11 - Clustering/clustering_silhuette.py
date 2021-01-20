# =============================================================================
# import stuff!
# =============================================================================
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score

# =============================================================================
# Read dataset
# =============================================================================
X, y = datasets.load_iris(return_X_y=True)  # split into X & Y

# =============================================================================
# Silhutee calculation & plot
# =============================================================================
for n_clusters in range(2, 11):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    # ax1 ############
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=11)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("K-Means for n_clusters =", n_clusters,
          ". The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("K-Means")
    ax1.set_xlabel("The silhouette coefficient values")
    # ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax2 ############
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer2 = SpectralClustering(n_clusters=n_clusters, random_state=11)
    cluster_labels2 = clusterer2.fit_predict(X)
    silhouette_avg2 = silhouette_score(X, cluster_labels2)
    print("Spectral Clustering for n_clusters =", n_clusters,
          ". The average silhouette_score is :", silhouette_avg2)
    sample_silhouette_values2 = silhouette_samples(X, cluster_labels2)
    y_lower2 = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values2 = \
            sample_silhouette_values2[cluster_labels2 == i]
        ith_cluster_silhouette_values2.sort()
        size_cluster_i2 = ith_cluster_silhouette_values2.shape[0]
        y_upper2 = y_lower2 + size_cluster_i2
        color2 = cm.nipy_spectral(float(i) / n_clusters)
        ax2.fill_betweenx(np.arange(y_lower2, y_upper2),
                          0, ith_cluster_silhouette_values2,
                          facecolor=color2, edgecolor=color2, alpha=0.7)
        ax2.text(-0.05, y_lower2 + 0.5 * size_cluster_i2, str(i))
        y_lower2 = y_upper2 + 10
    ax2.set_title("Spectral Clustering")
    ax2.set_xlabel("The silhouette coefficient values")
    # ax2.set_ylabel("Cluster label")
    ax2.axvline(x=silhouette_avg2, color="red", linestyle="--")
    ax2.set_yticks([])
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.savefig('Silhouette for ' + str(n_clusters) + ' clusters.png')
