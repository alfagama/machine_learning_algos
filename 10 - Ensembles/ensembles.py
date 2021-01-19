# =============================================================================
# Imports
# =============================================================================
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, metrics, model_selection

# =============================================================================
# Load breastCancer data
# =============================================================================
breastCancer = datasets.load_breast_cancer()

# =============================================================================
# For a fair comparison we choose the same features as in Random Forest exercise
# =============================================================================
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

# =============================================================================
# Split
# =============================================================================
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=1)
# print(x_train.shape)    # (426, 10)
# print(x_test.shape)     # (143, 10)
# print(y_train.shape)    # (426,)
# print(y_test.shape)     # (143,)

# =============================================================================
# Estimators
# =============================================================================
estimators = []

# ---------- Logistic Regression ----------------------------------------------
LR = LogisticRegression(
    C=1.0, class_weight=None, dual=False, fit_intercept=True,
    intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
    penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
    verbose=0, warm_start=False
)
estimators.append(LR)

# ---------- Naive Base Gaussian ----------------------------------------------
GaussianNB = GaussianNB(
    priors=None
)
estimators.append(GaussianNB)

# ---------- Decision Tree ----------------------------------------------------
DecisionTrees = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10
)
estimators.append(DecisionTrees)

# Decision Tree ---------------------------------------------------------------
RandomForest = RandomForestClassifier(
    n_estimators=200,
    criterion='gini',
    max_depth=10
)
estimators.append(RandomForest)

# KNN -------------------------------------------------------------------------
KNN = KNeighborsClassifier(
    n_neighbors=int(math.sqrt(len(y_train))),
    weights='uniform',
    metric='minkowski',
    p=1
)
estimators.append(KNN)

# =============================================================================
# Compute BaggingClassifier for all estimators
# =============================================================================
results = []
for estimator in estimators:
    model = BaggingClassifier(
        base_estimator=estimator,
        n_estimators=200,
        bootstrap=True,
        warm_start=True,
        random_state=11
    )
    model = model.fit(x_train, y_train)
    ytest_pred = model.predict(x_test)

    print(estimator)
    acc = metrics.accuracy_score(y_test, ytest_pred)
    pre = metrics.precision_score(y_test, ytest_pred, average="macro")
    rec = metrics.recall_score(y_test, ytest_pred, average="macro")
    print("F1: ", metrics.f1_score(y_test, ytest_pred, average="macro"))
    f1 = metrics.f1_score(y_test, ytest_pred, average="macro")
    result = [acc, pre, rec, f1]
    results.append(result)


# =============================================================================
# Find winner
# =============================================================================
compare_position, current_position, best_position = -1, -1, -1
for result in results:
    current_position += 1
    if result[3] > compare_position:
        compare_position = result[3]
        best_position = current_position

# =============================================================================
# Prepare X vs Y
# =============================================================================
names = ['Bagging Classifier->LR', 'Bagging Classifier->GNB',
         'Bagging Classifier->DT', 'Bagging Classifier->RF', 'Bagging Classifier->KNN']
labels = ['Accuracy', 'Precision', 'Recall', 'F1']
hihgest_ensemble = [round(elem, 2) for elem in results[best_position]]
old_random_forest = [0.9300699300699301, 0.9289308176100629, 0.9227272727272727, 0.9256138160632543]
old_random_forest = [round(elem, 2) for elem in old_random_forest]

# =============================================================================
# Plot
# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
# =============================================================================
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, hihgest_ensemble, width, color='Blue', label=names[best_position])
rects2 = ax.bar(x + width/2, old_random_forest, width, color='Pink', label='Random Forest')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage %')
ax.set_title('Scores (sklearn.metrics)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(
    loc=4,
    fancybox=True
)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.savefig('plot.png')
plt.show()
