# =============================================================================
# Imports
# =============================================================================
from sklearn import tree, datasets, metrics, model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
# =============================================================================
# Load breastCancer data
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
# =============================================================================
X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)

# =============================================================================
# Split into train and test
# =============================================================================
Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.25, random_state=1)

# =============================================================================
# MLPClassifier Parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
# =============================================================================
hidden_layer_sizes = [10, 20, 20, [50, 50, 50], 50, [100, 100, 100]]
activation = ['relu', 'tanh', 'tanh', 'relu', 'tanh', 'relu']
solver = ['sgd', 'sgd', 'adam', 'adam', 'lbfgs', 'lbfgs']
tol = [0.0001, 0.0001, 0.00001, 0.00001, 0.00001, 0.00001]
max_iter = [100, 100, 100, 100, 100, 100]

# =============================================================================
# MLPClassifier
# =============================================================================
def mlp_scores(X_train, X_test, y_train, y_test):
    for i in range(0, 6):
        mlp_clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes[i],
            activation=activation[i],
            solver=solver[i],
            tol=tol[i],
            max_iter=max_iter[i],
            random_state=1
        )
        mlp_clf.fit(X_train, y_train)
        y_predicted = mlp_clf.predict(X_test)

        print("Hidden layer sizes: ", hidden_layer_sizes[i], " Activation: ", activation[i], " Solver: ", solver[i],
              " Tolerance: ", tol[i], " Maximum iterations:", max_iter[i])
        print("Accuracy: ", metrics.accuracy_score(y_test, y_predicted))
        print("Precision:", metrics.precision_score(y_test, y_predicted, average="macro"))
        print("Recall: ", metrics.recall_score(y_test, y_predicted, average="macro"))
        print("F1: ", metrics.f1_score(y_test, y_predicted, average="macro"))


# =============================================================================
# Calling MLP
# =============================================================================
print()
print("Multi-layer Perceptron, with Parameters: ")
mlp_scores(Xtrain, Xtest, ytrain, ytest)

# =============================================================================
# scaled features
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# =============================================================================
X_sf_train = Xtrain.values
X_std_train = StandardScaler().fit_transform(X_sf_train)
X_sf_test = Xtest.values
X_std_test = StandardScaler().fit_transform(X_sf_test)

# =============================================================================
# PCA
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# =============================================================================
pca = PCA(n_components=4)
# fir train
pca.fit(X_std_train)
X_4d_train = pca.transform(X_std_train)
# fit test
pca.fit(X_std_test)
X_4d_test = pca.transform(X_std_test)

# =============================================================================
# Calling MLP with new features
# =============================================================================
print()
print("Multi-layer Perceptron, after PCA(n_components=4) with Parameters: ")
mlp_scores(X_std_train, X_std_test, ytrain, ytest)
