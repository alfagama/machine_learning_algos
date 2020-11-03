# =============================================================================
# HOMEWORK 1 - Supervised learning
# LINEAR REGRESSION ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================
# =============================================================================
# ARAMPATZIS GEORGIOS, AEM: 28
# =============================================================================
# =============================================================================

# From 'sklearn' library, we need to import:
# 'datasets', for loading our data
# 'metrics', for measuring scores
# 'linear_model', which includes the LinearRegression() method
# From 'scipy' library, we need to import:
# 'stats', which includes the spearmanr() and pearsonr() methods for computing correlation
# Additionally, we need to import
# 'pyplot' from package 'matplotlib' for our visualization purposes
# 'numpy', which implementse a wide variety of operations
# =============================================================================
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets, metrics, model_selection
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
# =============================================================================
# Load diabetes data from 'datasets' class
# =============================================================================
diabetes = datasets.load_diabetes()

# =============================================================================
# Get samples from the data, and keep only the features that you wish.
# =============================================================================

# Load just 1 feature for simplicity and visualization purposes...
# X: features
# Y: target value
#### Attribute Information:
## age age in years
## sex
## bmi body mass index
## bp average blood pressure
## s1 tc, T-Cells (a type of white blood cells)
## s2 ldl, low-density lipoproteins
## s3 hdl, high-density lipoproteins
## s4 tch, thyroid stimulating hormone
## s5 ltg, lamotrigine
## s6 glu, blood sugar level
X = diabetes.data[:, np.newaxis, 2] ## X with 1 feature for simplicity
## Target : Column 11 is a quantitative measure of disease progression one year after baseline
y = diabetes.target                 ## y with target/label of data

# =============================================================================
# Create linear regression model. All models behave differently, according to
# their own, model-specific parameter values. In our case, however, the linear
# regression model does not have any substancial parameters to tune. Refer
# to the documentation of this technique for more information.
# =============================================================================
#### Parameters
## fit_interceptbool,   default=True
## normalizebool,       default=False
## copy_Xbool,          default=True
## n_jobsint,           default=None
linearRegressionModel = linear_model.LinearRegression() #called with default parameters

# =============================================================================
# Split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# This proportion can be changed using the 'test_size' or 'train_size' parameter.
# Alsao, passing an (arbitrary) value to the parameter 'random_state' "freezes" the splitting procedure
# so that each run of the script always produces the same results (highly recommended).
# Apart from the train_test_function, this parameter is present in many routines and should be
# used whenever possible.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 0) # train-set size = 75% and test-set size = 25%

# =============================================================================
# Let's train our model.
# =============================================================================
linearRegressionModel.fit(x_train, y_train) # train model with X train-set and y train-set (our known values)

# =============================================================================
# Ok, now let's predict the output for the test input set
# =============================================================================
y_predicted = linearRegressionModel.predict(x_test) # after training we test with x test-set and get the y_predicted to compare with y test-set (supervised learning)

# =============================================================================
# Time to measure scores. We will compare predicted output (resulting from input x_test)
# with the true output (i.e. y_test).
# You can call 'pearsonr()' or 'spearmanr()' methods for computing correlation,
# 'mean_squared_error()' for computing MSE,
# 'r2_score()' for computing r^2 coefficient.
# =============================================================================
correlation_spearmanr, pvalue_spearmanr = stats.spearmanr(y_test, y_predicted)
print('Correlation using spearmanr: ', correlation_spearmanr, 'and pvalue: ', pvalue_spearmanr)
correlation_pearsonr, pvalue_pearsonr = stats.pearsonr(y_test, y_predicted)
print('Correlation using pearsonr: ', correlation_pearsonr, 'and pvalue: ', pvalue_pearsonr)
print('Mean squared error:', metrics.mean_squared_error(y_test, y_predicted))
print('Coefficient of determination:', metrics.r2_score(y_test, y_predicted))
# Coefficient of determination can also be calculated using score() function from model
# print(linearRegressionModel.score(x_test, y_test))

# =============================================================================
# Plot results in a 2D plot (scatter() plot, line plot())
# =============================================================================
# # Display 'ticks' in x-axis and y-axis
plt.scatter(x_test, y_test,  color='blue')
plt.plot(x_test, y_predicted, color='red', linewidth=1)

plt.xticks()
plt.yticks()

# # Save plot
plt.savefig('LinearRegression_DiabetesDS.png')

# # Show plot
plt.show()

# =============================================================================