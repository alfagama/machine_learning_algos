# =============================================================================
#  Imports
# =============================================================================
from sklearn.model_selection import GridSearchCV
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
#  Classifiers
# =============================================================================
#   -----The concept here is:
#   1.  Lists with model parameters
#   2.  Run GridSearchCV for many possible combination
#   3.  Print best results for: ACC, PRE, REC, F1
# =============================================================================

# =============================================================================
#  KMeans
# =============================================================================
def kmeans(train_set, test_set, target):
    print("KMeans...")

    parametes = {
        'n_clusters': [2],
        'init': ['k-means++'],
        'n_init': [50, 70, 100],
        'max_iter': [100, 300, 500],
        'tol': [1e-04, 1e-05, 1e-06],
        'precompute_distances': ['deprecated'],
        'verbose': [0],
        'random_state': [11],
        'copy_x': [True],
        'n_jobs': ['deprecated'],
        'algorithm': ['auto']
    }
    model = KMeans()
    kmeans_m = GridSearchCV(model, parametes)
    kmeans_m = KMeans(n_clusters=2,
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
    kmeans_m.fit(train_set)
    kmeans_pred = kmeans_m.predict(test_set)
    print("ACC: ", metrics.accuracy_score(target, kmeans_pred),
          "PRE: ", metrics.precision_score(target, kmeans_pred, average="macro"),
          "REC: ", metrics.recall_score(target, kmeans_pred, average="macro"),
          "F1: ", metrics.f1_score(target, kmeans_pred, average="macro"))


# =============================================================================
#  MiniBatchKMeans
# =============================================================================
def minibatchkmeans(train_set, test_set, target):
    print("MiniBatchKMeans...")

    # parametes = {
    #     'n_clusters': [2],
    #     'init': ['k-means++'],
    #     'max_iter': [100, 300, 500],
    #     'batch_size': [32, 64, 128],
    #     'verbose': [0],
    #     'compute_labels': [True],
    #     'random_state': [11],
    #     'tol': [1e-04, 1e-05, 1e-06],
    #     'max_no_improvement': [3, 5, 10],
    #     'init_size': [None],
    #     'n_init': [50, 70, 100],
    #     'reassignment_ratio': [0.01]
    # }
    # model = KMeans()
    # mbkmeans = GridSearchCV(model, parametes)
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
    mbkmeans.fit(train_set)
    mbkmeans_pred = mbkmeans.predict(test_set)
    print("ACC: ", metrics.accuracy_score(target, mbkmeans_pred),
          "PRE: ", metrics.precision_score(target, mbkmeans_pred, average="macro"),
          "REC: ", metrics.recall_score(target, mbkmeans_pred, average="macro"),
          "F1: ", metrics.f1_score(target, mbkmeans_pred, average="macro"))


# =============================================================================
#  AgglomerativeClustering / Hierarchical clustering
# =============================================================================
def agglomerative(train_set, test_set, target):
    print("AgglomerativeClustering...")

    agglomerative_cl = AgglomerativeClustering(n_clusters=2,
                                               affinity='euclidean',
                                               memory=None,
                                               connectivity=None,
                                               compute_full_tree=True,
                                               linkage='average')
    agglomerative_cl.fit(train_set)
    agglomerative_cl_pred = agglomerative_cl.fit_predict(test_set)
    print("ACC: ", metrics.accuracy_score(target, agglomerative_cl_pred),
          "PRE: ", metrics.precision_score(target, agglomerative_cl_pred, average="macro"),
          "REC: ", metrics.recall_score(target, agglomerative_cl_pred, average="macro"),
          "F1: ", metrics.f1_score(target, agglomerative_cl_pred, average="macro"))


# =============================================================================
#  Birch
# =============================================================================
def birch(train_set, test_set, target):
    print("Birch...")

    birch = Birch(threshold=1.0,
                  branching_factor=150,
                  # compute_labels=False,
                  # copy=False,
                  n_clusters=2)

    birch.fit(train_set)
    birch_pred = birch.predict(test_set)
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
    agglomerative(train_set, test_set, target)
    birch(train_set, test_set, target)
