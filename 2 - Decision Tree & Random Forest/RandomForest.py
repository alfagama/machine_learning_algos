# =============================================================================
# HOMEWORK 2 - DECISION TREES
# RANDOM FOREST ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================
# =============================================================================
# ARAMPATZIS GEORGIOS, AEM: 28
# =============================================================================
# =============================================================================


# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'ensemble' package, for calling the Random Forest classifier
# 'model_selection', (instead of the 'cross_validation' package), which will help validate our results.
# =============================================================================
from sklearn import ensemble, datasets, metrics, model_selection
import matplotlib.pyplot as plt

# =============================================================================
# Load breastCancer data
# =============================================================================
breastCancer = datasets.load_breast_cancer()

# =============================================================================
# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

# Split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# This proportion can be changed using the 'test_size' or 'train_size' parameter.
# Also, passing an (arbitrary) value to the parameter 'random_state' "freezes" the splitting procedure
# so that each run of the script always produces the same results (highly recommended).
# Apart from the train_test_function, this parameter is present in many routines and should be
# used whenever possible.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=1)

# RandomForestClassifier() is the core of this script. You can call it from the 'ensemble' class.
# You can customize its functionality in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the Information Gain.
# 'n_estimators': The number of trees in the forest. The larger the better, but it will take longer to compute. Also,
#                 there is a critical number after which there is no significant improvement in the results
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================
model = ensemble.RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=3,
    random_state=1
)

# =============================================================================
# Let's train our model.
# =============================================================================
model.fit(x_train, y_train)

# =============================================================================
# Ok, now let's predict the output for the test set
# =============================================================================
y_predicted = model.predict(x_test)

# =============================================================================
# Time to measure scores. We will compare predicted output (from input of second subset, i.e. x_test)
# with the real output (output of second subset, i.e. y_test).
# You can call 'accuracy_score', 'recall_score', 'precision_score', 'f1_score' or any other available metric
# from the 'sklearn.metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# One of the following can be used for this example, but it is recommended that 'macro' is used (for now):
# 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
# 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
#             This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
# =============================================================================
print("Random Forest - Model Evaluation: ")
print("Accuracy: ", metrics.accuracy_score(y_test, y_predicted))
print("Precision:", metrics.precision_score(y_test, y_predicted, average="macro"))
print("Recall: ", metrics.recall_score(y_test, y_predicted, average="macro"))
print("F1: ", metrics.f1_score(y_test, y_predicted, average="macro"))

# =============================================================================
# A Random Forest has been trained now, but let's train more models,
# with different number of estimators each, and plot performance in terms of
# the difference metrics. In other words, we need to make 'n'(e.g. 200) models,
# evaluate them on the aforementioned metrics, and plot 4 performance figures
# (one for each metric).
# In essence, the same pipeline as previously will be followed.
# =============================================================================
n_estimators = 200
# Create empty lists for all metrics + 1 for n_estimators
num_of_trees = []
accuracy = []
precision = []
recall = []
f1score = []

for i in range(1, n_estimators + 1):
    # Creating models
    forest = ensemble.RandomForestClassifier(
        n_estimators=i,
        criterion='gini',
        max_depth=3,
        # warm_start=True,
        random_state=1)
    # Training models
    forest.fit(x_train, y_train)
    # Prediction of models
    y_pred = forest.predict(x_test)
    # Filling lists with metrics + n_estimators
    num_of_trees.append(i)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
    precision.append(metrics.precision_score(y_test, y_pred, average="macro"))
    recall.append(metrics.recall_score(y_test, y_pred, average="macro"))
    f1score.append(metrics.f1_score(y_test, y_pred, average="macro"))

fig, axs = plt.subplots(2, 2,
                        gridspec_kw={'hspace': 0.5, 'wspace': 0.5})
axs[0, 0].plot(num_of_trees, accuracy, 'tab:blue')
axs[0, 0].set_title('Accuracy')
axs[0, 1].plot(num_of_trees, precision, 'tab:orange')
axs[0, 1].set_title('Precision')
axs[1, 0].plot(num_of_trees, recall, 'tab:green')
axs[1, 0].set_title('Recall')
axs[1, 1].plot(num_of_trees, f1score, 'tab:red')
axs[1, 1].set_title('F1 Score')

for ax in axs.flat:
    ax.set(xlabel='Number of Estimators', ylabel='Percentage (%)')

# # Save plot
plt.savefig('RandomForest_Metrics_BreastCancerDB.png')

# # Show plot
#plt.show()
# =============================================================================
# After finishing the above plots, try doing the same thing on the train data
# Hint: you can plot on the same figure in order to add a second line.
# Change the line color to distinguish performance metrics on train/test data
# In the end, you should have 4 figures (one for each metric)
# And each figure should have 2 lines (one for train data and one for test data)
# =============================================================================
overfit_num_of_trees = []
overfit_accuracy = []
overfit_precision = []
overfit_recall = []
overfit_f1score = []

for i in range(1, n_estimators + 1):
    # Creating models
    overfit_forest = ensemble.RandomForestClassifier(
        n_estimators=i,
        criterion='gini',
        max_depth=3,
        random_state=1)
    # Training models
    overfit_forest.fit(x_train, y_train)
    # Prediction of models
    overfit_y_pred = overfit_forest.predict(x_train)
    # Filling lists with metrics + n_estimators
    overfit_num_of_trees.append(i)
    overfit_accuracy.append(metrics.accuracy_score(y_train, overfit_y_pred))
    overfit_precision.append(metrics.precision_score(y_train, overfit_y_pred, average="macro"))
    overfit_recall.append(metrics.recall_score(y_train, overfit_y_pred, average="macro"))
    overfit_f1score.append(metrics.f1_score(y_train, overfit_y_pred, average="macro"))

fig2, axs2 = plt.subplots(2, 2,
                        gridspec_kw={'hspace': 0.5, 'wspace': 0.5})

axs2[0, 0].plot(num_of_trees, accuracy, 'tab:orange', label='test set')
axs2[0, 0].plot(overfit_num_of_trees, overfit_accuracy, label='train set')
axs2[0, 0].set_title('Accuracy')
axs2[0, 1].plot(num_of_trees, precision, 'tab:orange')
axs2[0, 1].plot(overfit_num_of_trees, overfit_precision)
axs2[0, 1].set_title('Precision')
axs2[1, 0].plot(num_of_trees, recall, 'tab:orange')
axs2[1, 0].plot(overfit_num_of_trees, overfit_recall)
axs2[1, 0].set_title('Recall')
axs2[1, 1].plot(num_of_trees, f1score, 'tab:orange')
axs2[1, 1].plot(overfit_num_of_trees, overfit_f1score)
axs2[1, 1].set_title('F1 Score')

for ax in axs2.flat:
    ax.set(xlabel='Number of Estimators', ylabel='Percentage (%)')

fig2.legend(shadow=True, fancybox=True, loc=1)

# # Save plot
plt.savefig('RandomForest_Metrics_Overfit_BreastCancerDB.png')

# # Show plot
plt.show()

# =============================================================================
