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
                    n_init=50,
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
    # print(kmeans_pred)
    # print(metrics.accuracy_score(target, kmeans_pred))
    # print(metrics.precision_score(target, kmeans_pred, average="macro"))
    # print(metrics.recall_score(target, kmeans_pred, average="macro"))
    # print(metrics.f1_score(target, kmeans_pred, average="macro"))
    print(metrics.accuracy_score(target, kmeans_pred),
          metrics.precision_score(target, kmeans_pred, average="macro"),
          metrics.recall_score(target, kmeans_pred, average="macro"),
          metrics.f1_score(target, kmeans_pred, average="macro"))


# =============================================================================
#  DBSCAN
# =============================================================================
def dbscan(train_set, test_set, target):
    eps_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    min_samples = [1, 3, 5, 7, 10]
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_sizes = [10, 20, 30, 40, 50]
    p_values = [None]

    for eps in eps_vals:
        for min_sample in min_samples:
            for algorithm in algorithms:
                for leaf_size in leaf_sizes:
                    for p in p_values:
                        dbscan = DBSCAN(eps=eps,
                                        min_samples=min_sample,
                                        metric='euclidean',
                                        metric_params=None,
                                        algorithm=algorithm,
                                        leaf_size=leaf_size,
                                        p=p,
                                        n_jobs=None)

                        print("DBSCAN..", eps, min_sample, algorithm, leaf_size, p)
                        dbscan.fit(train_set)
                        dbscan_pred = dbscan.fit_predict(test_set)
                        print(metrics.accuracy_score(target, dbscan_pred),
                              metrics.precision_score(target, dbscan_pred, average="macro"),
                              metrics.recall_score(target, dbscan_pred, average="macro"),
                              metrics.f1_score(target, dbscan_pred, average="macro"))


# =============================================================================
#  MiniBatchKMeans
# =============================================================================
def minibatchkmeans(train_set, test_set, target):

    mbkmeans = MiniBatchKMeans(n_clusters=2,
                               init='k-means++',
                               max_iter=300,
                               batch_size=128,
                               verbose=0,
                               compute_labels=True,
                               random_state=11,
                               tol=1e-06,
                               max_no_improvement=10,
                               init_size=None,
                               n_init=50,
                               reassignment_ratio=0.01)
    print("MiniBatchKMeans..")
    mbkmeans.fit(train_set)
    mbkmeans_pred = mbkmeans.predict(test_set)
    print(metrics.accuracy_score(target, mbkmeans_pred),
          metrics.precision_score(target, mbkmeans_pred, average="macro"),
          metrics.recall_score(target, mbkmeans_pred, average="macro"),
          metrics.f1_score(target, mbkmeans_pred, average="macro"))


# =============================================================================
#  AffinityPropagation
# =============================================================================
def affinity_propagation(train_set, test_set, target):
    aff_prop = AffinityPropagation(damping=0.5,
                                   max_iter=200,
                                   convergence_iter=15,
                                   copy=True,
                                   preference=None,
                                   affinity='euclidean',
                                   verbose=False,
                                   random_state='warn')
    print("AffinityPropagation..")
    aff_prop.fit(train_set)
    aff_prop_pred = aff_prop.predict(test_set)
    print(metrics.accuracy_score(target, aff_prop_pred),
          metrics.precision_score(target, aff_prop_pred, average="macro"),
          metrics.recall_score(target, aff_prop_pred, average="macro"),
          metrics.f1_score(target, aff_prop_pred, average="macro"))


# =============================================================================
#  MeanShift
# =============================================================================
def meanshift(train_set, test_set, target):
    meanshift = MeanShift(bandwidth=None,
                          seeds=None,
                          bin_seeding=False,
                          min_bin_freq=1,
                          cluster_all=True,
                          n_jobs=None,
                          max_iter=300)
    print("MeanShift..")
    meanshift.fit(train_set)
    meanshift_pred = meanshift.predict(test_set)
    print(metrics.accuracy_score(target, meanshift_pred),
          metrics.precision_score(target, meanshift_pred, average="macro"),
          metrics.recall_score(target, meanshift_pred, average="macro"),
          metrics.f1_score(target, meanshift_pred, average="macro"))


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
    spectral_pred = spectral.predict(test_set)
    print(metrics.accuracy_score(target, spectral_pred),
          metrics.precision_score(target, spectral_pred, average="macro"),
          metrics.recall_score(target, spectral_pred, average="macro"),
          metrics.f1_score(target, spectral_pred, average="macro"))


# =============================================================================
#  AgglomerativeClustering
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
    print(metrics.accuracy_score(target, agglomerative_cl_pred),
          metrics.precision_score(target, agglomerative_cl_pred, average="macro"),
          metrics.recall_score(target, agglomerative_cl_pred, average="macro"),
          metrics.f1_score(target, agglomerative_cl_pred, average="macro"))


# =============================================================================
#  OPTICS
# =============================================================================
def optics(train_set, test_set, target):
    optics = OPTICS(min_samples=5,
                    max_eps=np.inf,
                    metric='minkowski',
                    p=2,
                    metric_params=None,
                    cluster_method='xi',
                    eps=None,
                    xi=0.05,
                    predecessor_correction=True,
                    min_cluster_size=None,
                    algorithm='auto',
                    leaf_size=30,
                    n_jobs=None)
    print("OPTICS..")
    optics.fit(train_set)
    optics_pred = optics.predict(test_set)
    print(optics_pred)
    print(metrics.accuracy_score(target, optics_pred),
          metrics.precision_score(target, optics_pred, average="macro"),
          metrics.recall_score(target, optics_pred, average="macro"),
          metrics.f1_score(target, optics_pred, average="macro"))


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
            print(metrics.accuracy_score(target, birch_pred),
                  metrics.precision_score(target, birch_pred, average="macro"),
                  metrics.recall_score(target, birch_pred, average="macro"),
                  metrics.f1_score(target, birch_pred, average="macro"))


# =============================================================================
#  Results
# =============================================================================
def results(train_set, test_set, target):
    # kmeans(train_set, test_set, target)
    # dbscan(train_set, test_set, target)
    # minibatchkmeans(train_set, test_set, target)
    # affinity_propagation(train_set, test_set, target)
    # meanshift(train_set, test_set, target)
    # spectral(train_set, test_set, target)
    agglomerative(train_set, test_set, target)
    # optics(train_set, test_set, target)
    # birch(train_set, test_set, target)
