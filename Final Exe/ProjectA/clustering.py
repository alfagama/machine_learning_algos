# =============================================================================
#  Imports
# =============================================================================
from sklearn.cluster \
    import KMeans, \
           DBSCAN, \
           MiniBatchKMeans, \
           AffinityPropagation, \
           MeanShift, \
           SpectralClustering, \
           AgglomerativeClustering, \
           OPTICS, \
           Birch
from sklearn import metrics
import numpy as np

# =============================================================================
#  KMeans
# =============================================================================
def kmeans(train_set, test_set, target):
    kmeans = KMeans(n_clusters=2,
                    init='k-means++',
                    n_init=70,
                    max_iter=300,
                    tol=1e-06,
                    precompute_distances='deprecated',
                    verbose=0,
                    random_state=11,
                    copy_x=True,
                    n_jobs='deprecated',
                    algorithm='auto')
    print("KMeans...:")
    # 0->attack, 1->normal
    kmeans.fit(train_set)
    kmeans_pred = kmeans.predict(test_set)
    print("ACC: ", metrics.accuracy_score(target, kmeans_pred),
          "PRE: ", metrics.precision_score(target, kmeans_pred, average="macro"),
          "REC: ", metrics.recall_score(target, kmeans_pred, average="macro"),
          "F1: ", metrics.f1_score(target, kmeans_pred, average="macro"))


# =============================================================================
#  MiniBatchKMeans
# =============================================================================
def minibatchkmeans(train_set, test_set, target):
    mbkmeans = MiniBatchKMeans(n_clusters=2,
                               init='k-means++',
                               max_iter=300,
                               batch_size=64,
                               verbose=0,
                               compute_labels=True,
                               random_state=11,
                               tol=1e-04,
                               max_no_improvement=5,
                               init_size=None,
                               n_init=70,
                               reassignment_ratio=0.01)
    print("MiniBatchKMeans..")
    mbkmeans.fit(train_set)
    mbkmeans_pred = mbkmeans.predict(test_set)
    print("ACC: ", metrics.accuracy_score(target, mbkmeans_pred),
          "PRE: ", metrics.precision_score(target, mbkmeans_pred, average="macro"),
          "REC: ", metrics.recall_score(target, mbkmeans_pred, average="macro"),
          "F1: ", metrics.f1_score(target, mbkmeans_pred, average="macro"))


# =============================================================================
#  SpectralClustering
# =============================================================================
def spectral(train_set, test_set, target):
    spectral = SpectralClustering(n_clusters=2,
                                  eigen_solver=None,
                                  n_components=None,
                                  random_state=None,
                                  n_init=10,
                                  gamma=1.0,
                                  affinity='rbf',
                                  n_neighbors=10,
                                  eigen_tol=0.0,
                                  assign_labels='kmeans',
                                  degree=3,
                                  coef0=1,
                                  kernel_params=None,
                                  n_jobs=None)
    print("SpectralClustering..")
    spectral.fit(train_set)
    spectral_pred = spectral.fit_predict(test_set)
    print("ACC: ", metrics.accuracy_score(target, spectral_pred),
          "PRE: ", metrics.precision_score(target, spectral_pred, average="macro"),
          "REC: ", metrics.recall_score(target, spectral_pred, average="macro"),
          "F1: ", metrics.f1_score(target, spectral_pred, average="macro"))


# =============================================================================
#  AgglomerativeClustering / Hierarchical clustering
# =============================================================================
def agglomerative(train_set, test_set, target):
    agglomerative_cl = AgglomerativeClustering(n_clusters=2,
                                               affinity='euclidean',
                                               memory=None,
                                               connectivity=None,
                                               compute_full_tree=True,
                                               linkage='average')
    print("AgglomerativeClustering..")
    agglomerative_cl.fit(train_set)
    agglomerative_cl_pred = agglomerative_cl.fit_predict(test_set)
    for i in agglomerative_cl_pred:
        print(i)
    print(target)
    print("ACC: ", metrics.accuracy_score(target, agglomerative_cl_pred),
          "PRE: ", metrics.precision_score(target, agglomerative_cl_pred, average="macro"),
          "REC: ", metrics.recall_score(target, agglomerative_cl_pred, average="macro"),
          "F1: ", metrics.f1_score(target, agglomerative_cl_pred, average="macro"))


# =============================================================================
#  Birch
# =============================================================================
def birch(train_set, test_set, target):
    thresholds = [0.3, 0.5, 0.7, 0.9]
    branching_factors = [50]  # [10, 30, 50, 70, 90]
    for threshold in thresholds:
        for branching_factor in branching_factors:
            birch = Birch(threshold=threshold,
                          branching_factor=branching_factor,
                          n_clusters=2,
                          compute_labels=False,
                          copy=False)
            print("Birch..", threshold, branching_factor)
            birch.fit(train_set)
            birch_pred = birch.predict(test_set)
            print(birch_pred)  # -> ola 1
            print("ACC: ", metrics.accuracy_score(target, birch_pred),
                  "PRE: ", metrics.precision_score(target, birch_pred, average="macro"),
                  "REC: ", metrics.recall_score(target, birch_pred, average="macro"),
                  "F1: ", metrics.f1_score(target, birch_pred, average="macro"))


# =============================================================================
#  Results
# =============================================================================
def results(train_set, test_set, target):
    kmeans(train_set, test_set, target)
    minibatchkmeans(train_set, test_set, target)
    spectral(train_set, test_set, target)
    agglomerative(train_set, test_set, target)
    birch(train_set, test_set, target)
