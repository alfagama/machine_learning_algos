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


# IMPORT NECESSARY LIBRARIES HERE
from sklearn import ensemble, datasets, metrics, model_selection


# =============================================================================



# Load breastCancer data
# =============================================================================


# ADD COMMAND TO LOAD DATA HERE
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


# ADD COMMAND TO CREATE RANDOM FOREST CLASSIFIER MODEL HERE
model = ensemble.RandomForestClassifier(
  n_estimators=100,
  criterion='gini',
  max_depth=3,
  random_state=1
 )


# =============================================================================







# Let's train our model.
# =============================================================================


# ADD COMMAND TO TRAIN YOUR MODEL HERE
model.fit(x_train, y_train)

# =============================================================================




# Ok, now let's predict the output for the test set
# =============================================================================


# ADD COMMAND TO MAKE A PREDICTION HERE
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



# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("Random Forest - Model Evaluation: ")
print("Recall: ", metrics.recall_score(y_test, y_predicted, average="macro"))
print("Precision:", metrics.precision_score(y_test, y_predicted, average="macro"))
print("Accuracy: ", metrics.accuracy_score(y_test, y_predicted))
print("F1: ", metrics.f1_score(y_test, y_predicted, average="macro"))

# =============================================================================



# A Random Forest has been trained now, but let's train more models, 
# with different number of estimators each, and plot performance in terms of
# the difference metrics. In other words, we need to make 'n'(e.g. 200) models,
# evaluate them on the aforementioned metrics, and plot 4 performance figures
# (one for each metric).
# In essence, the same pipeline as previously will be followed.
# =============================================================================

# After finishing the above plots, try doing the same thing on the train data
# Hint: you can plot on the same figure in order to add a second line.
# Change the line color to distinguish performance metrics on train/test data
# In the end, you should have 4 figures (one for each metric)
# And each figure should have 2 lines (one for train data and one for test data)



# CREATE MODELS AND PLOTS HERE


# =============================================================================