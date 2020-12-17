# =============================================================================
# import stuff!
# =============================================================================
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

# =============================================================================
# metrics for SVC - Undersampling
# =============================================================================


# =============================================================================
# metrics for SVC - Oversampling
# =============================================================================


# =============================================================================
# show plot
# =============================================================================
# from collections import Counter
# from matplotlib import pyplot
# from numpy import where
# # summarize class distribution
# counter = Counter(y)
# print(counter)
# # scatter plot of examples by class label
# for label, _ in counter.items():
# 	row_ix = where(y == label)[0]
# 	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
# pyplot.legend()
# pyplot.show()
