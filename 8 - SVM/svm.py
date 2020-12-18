# =============================================================================
# import stuff!
# =============================================================================
import imblearn
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# =============================================================================
# read dataset
# =============================================================================
creditcard_dataset = pd.read_csv(
    "creditcard.csv",
    sep=',',
    # names=names,
    header=0,  # no header, alternative header = header_col
    index_col=None,  # no index, alternative header = index_row
    skiprows=0
)
# print(creditcard_dataset.head())

# =============================================================================
# check dataset imbalance
# =============================================================================
# print(creditcard_dataset.describe())
num_ok_transaction = len(creditcard_dataset[creditcard_dataset["Class"] == 0])
num_fraud_detected = len(creditcard_dataset[creditcard_dataset["Class"] == 1])
print("Number of proper transaction (class 0) are: ", num_ok_transaction, ",",
      (num_ok_transaction / (num_ok_transaction + num_fraud_detected)), " % of the dataset.")
print("Number of fraudulent transaction (class 1) are: ", num_fraud_detected, ",",
      (num_fraud_detected / (num_ok_transaction + num_fraud_detected)), " % of the dataset.")

# =============================================================================
# drop column "Time"
# =============================================================================
creditcard_dataset = creditcard_dataset.drop("Time", axis=1)

# =============================================================================
# use StandardScaler for "Amount" column, creating "Scaled_Amount" column
# =============================================================================
sc = StandardScaler()
creditcard_dataset["Scaled_Amount"] = sc.fit_transform(creditcard_dataset.iloc[:, 28].values.reshape(-1, 1))

# =============================================================================
# drop column "Amount"
# =============================================================================
creditcard_dataset = creditcard_dataset.drop("Amount", axis=1)
# print(creditcard_dataset.head())

# =============================================================================
# split dataset in features and label
# =============================================================================
X = creditcard_dataset.iloc[:, creditcard_dataset.columns != "Class"].values
y = creditcard_dataset.iloc[:, creditcard_dataset.columns == "Class"].values

# =============================================================================
# split dataset in train and test
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print("Size of training set: ", X_train.shape)

# =============================================================================
# smv.SVC model parameters
# =============================================================================
# ### smv.SVC basic parameters >>>
# C : float, default=1.0
# kernel : {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
# degree : int, default=3
# gamma : {‘scale’, ‘auto’} or float, default=’scale’
#       Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
#       if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
#       if ‘auto’, uses 1 / n_features.
# ### smv.SVC basic parameters <<<
c_vals = [0.1, 10, 0.1, 10, 0.1, 10, 100]
k_vals = ['poly', 'poly', 'rbf', 'rbf', 'sigmoid', 'sigmoid', 'sigmoid']
g_vals = [0.2, 6, 0.3, 5, 0.5, 2, 5]
d_vals = [2, 5, 3, 3, 3, 3, 3]

# =============================================================================
# metrics for SVC
# =============================================================================
def svc_imbalanced():
    for i in range(0, 7):
        svc_model = svm.SVC(
            C=c_vals[i],
            kernel=k_vals[i],
            degree=d_vals[i],
            gamma=g_vals[i],
            random_state=1
        )
        svc_model.fit(X_train, y_train)
        y_predicted = svc_model.predict(X_test)
        print("SVC for C: ", c_vals[i], ", kernel: ", k_vals[i], " , gamma: ", g_vals[i], " degree: ", d_vals[i], ".")
        print(metrics.accuracy_score(y_test, y_predicted))
        print(metrics.precision_score(y_test, y_predicted, average="macro"))
        print(metrics.recall_score(y_test, y_predicted, average="macro"))
        print(metrics.f1_score(y_test, y_predicted, average="macro"))


print("SVC Results with no preprocesing: ")
svc_imbalanced()

# =============================================================================
# metrics for SVC - Undersampling
# =============================================================================
def undersampling(method):
    X_undersampled, y_undersampled = method.fit_resample(X, y)
    X_train_undersampled, X_test_undersampled, y_train_undersampled, y_test_undersampled = train_test_split(
        X_undersampled, y_undersampled, test_size=0.3, random_state=1)
    print("Size of training set: ", X_undersampled.shape)
    for i in range(0, 7):
        svc_model = svm.SVC(
            C=c_vals[i],
            kernel=k_vals[i],
            degree=d_vals[i],
            gamma=g_vals[i],
            random_state=1
        )
        svc_model.fit(X_train_undersampled, y_train_undersampled)
        y_predicted_undersampled = svc_model.predict(X_test_undersampled)
        print("SVC for C: ", c_vals[i], ", kernel: ", k_vals[i], " , gamma: ", g_vals[i], " degree: ", d_vals[i], ".")
        print(metrics.accuracy_score(y_test_undersampled, y_predicted_undersampled))
        print(metrics.precision_score(y_test_undersampled, y_predicted_undersampled, average="macro"))
        print(metrics.recall_score(y_test_undersampled, y_predicted_undersampled, average="macro"))
        print(metrics.f1_score(y_test_undersampled, y_predicted_undersampled, average="macro"))


print("Undersampling Methods")
print()
print("Methods that Select Examples to Keep")
for version_num in range(1, 4):
    print("Undersampling - Near Miss Undersampling (version: ", version_num, ")")
    undersampling(imblearn.under_sampling.NearMiss(version=version_num, n_neighbors=25))
print("Condensed Nearest Neighbor Rule")
undersampling(imblearn.under_sampling.CondensedNearestNeighbour(n_neighbors=1))
print()
print("Methods that Select Examples to Delete")
print("Tomek Links")
undersampling(imblearn.under_sampling.TomekLinks())
print("Edited Nearest Neighbors Rule")
undersampling(imblearn.under_sampling.EditedNearestNeighbours(n_neighbors=3))
print()
print("Combinations of Keep and Delete Methods")
print("One-Sided Selection")
undersampling(imblearn.under_sampling.OneSidedSelection(n_neighbors=1, n_seeds_S=200))
print("Neighborhood Cleaning Rule")
undersampling(imblearn.under_sampling.NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5))

# =============================================================================
# SMOTE plot - Oversampling

# =============================================================================
# from collections import Counter
# from numpy import where
# from matplotlib import pyplot
#
# # transform the dataset
# oversample = imblearn.over_sampling.SMOTE()
# X_smote, y_smote = oversample.fit_resample(X, y)
# # plot after SMOTE
# counter = Counter(y_smote)
# for label, _ in counter.items():
# 	row_ix = where(y_smote == label)[0]
# 	pyplot.scatter(X_smote[row_ix, 0], X_smote[row_ix, 1], label=str(label))
# pyplot.legend()
# # pyplot.savefig("Fig/SMOTE.png")
# # pyplot.show()

# =============================================================================
# metrics for SVC - Oversampling
# =============================================================================
def svc_smote(method):
    X_smote, y_smote = method.fit_resample(X, y)
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(
        X_smote, y_smote, test_size=0.3, random_state=1)
    print("Size of training set: ", X_smote.shape)
    for i in range(0, 7):
        svc_model = svm.SVC(
            C=c_vals[i],
            kernel=k_vals[i],
            degree=d_vals[i],
            gamma=g_vals[i],
            random_state=1
        )
        svc_model.fit(X_train_smote, y_train_smote)
        y_predicted_smote = svc_model.predict(X_test_smote)
        print("SVC for C: ", c_vals[i], ", kernel: ", k_vals[i], " , gamma: ", g_vals[i], " degree: ", d_vals[i], ".")
        print(metrics.accuracy_score(y_test_smote, y_predicted_smote))
        print(metrics.precision_score(y_test_smote, y_predicted_smote, average="macro"))
        print(metrics.recall_score(y_test_smote, y_predicted_smote, average="macro"))
        print(metrics.f1_score(y_test_smote, y_predicted_smote, average="macro"))


print("Oversampling Methods")
print()
print("SMOTE")
svc_smote(imblearn.over_sampling.SMOTE())
print("Borderline-SMOTE SVM")
svc_smote(imblearn.over_sampling.SVMSMOTE())
