from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt

def silhuette_score(X):
    fig, ax = plt.subplots(2, 2, figsize=(15,8))
    range_n_clusters = [2, 3, 4, 5]
    for n_cluster in range_n_clusters:
        '''
        Create KMeans instance for different number of clusters
        '''
        # Create the model
        km = KMeans(n_clusters=n_cluster,
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
        # Fit the KMeans model
        km.fit_predict(X)
        # Calculate Silhoutte Score
        score = silhouette_score(X, km.labels_, metric='euclidean')
        # Print the score
        print('Silhouetter Score of K-Means for ', n_cluster, 'number of clusters: %.6f' % score)
        q, mod = divmod(n_cluster, 2)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        '''
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(X)
