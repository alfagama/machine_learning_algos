# =============================================================================
# HOMEWORK 4 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email us: arislaza@csd.auth.gr, ipierros@csd.auth.gr
# =============================================================================
# =============================================================================
# ARAMPATZIS GEORGIOS, AEM: 28
# =============================================================================
# =============================================================================



#=============================================================================
# import the KNeighborsClassifier
# if you want to do the hard task, also import the KNNImputer
import numpy as np
import random
import matplotlib.pyplot as plt
#import KNNImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import warnings

# Suppress warning:
# SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame
# warnings.filterwarnings("ignore")

random.seed = 42
np.random.seed(666)

# Import the titanic dataset
# Decide which features you want to use (some of them are useless, ie PassengerId).
#
# Feature 'Sex': because this is categorical instead of numerical, KNN can't deal with it, so drop it
# Note: another solution is to use one-hot-encoding, but it's out of the scope for this exercise.
#
# Feature 'Age': because this column contains missing values, KNN can't deal with it, so drop it
# If you want to do the harder task, don't drop it.
#
# =============================================================================
#   Reading .csv
titanic = pd.read_csv("titanic.csv",
                       sep=',',
                       header=0,  # no header, alternative header = header_col
                       #index_col=None,  # no index, alternative header = index_row
                       #skiprows=0  # how many rows to skip / not include in read_csv
                       )
pd.set_option('display.max_columns', None)
#   make sure that it got read OK
print("Number of data: ", len(titanic))
# print(titanic.head())
#   check data stats
# print(titanic.describe())

#   DROP unwanted columns that make no difference
drop_columns = ["PassengerId", "Name", "Ticket", "Fare"]
titanic = titanic.drop(drop_columns, axis=1)
#   Change Cabin to binary 0/1 if Cabin = Nan then hasCabin = 0, else hasCabin = 1
# titanic.rename(columns={'Cabin': 'hasCabin'}, inplace=True)
titanic['hasCabin'] = 0
titanic["Cabin"] = titanic["Cabin"].fillna(0)
titanic["hasCabin"][titanic["Cabin"] != 0] = 1
titanic = titanic.drop("Cabin", axis=1)
#   One-Hot-Encoding column: "Sex" into "Male" or "Female"
titanic["Male"] = 0
titanic["Female"] = 0
titanic["Male"][titanic["Sex"] == "male"] = 1
titanic["Female"][titanic["Sex"] == "female"] = 1
titanic = titanic.drop("Sex", axis=1)
# print(titanic.head(20))
#   Check for missing values
columns = list(titanic.columns.values)
# print("\nNull values in columns: ...")
# for column in columns:
#     print(column, ": ", sum(pd.isnull(titanic[column])))
#   Results:
    # Survived :  0
    # Pclass :  0
    # Age :  177
    # SibSp :  0
    # Parch :  0
    # Embarked :  2
    # Male :  0
    # Female :  0
#   Replace Embarked NaN with most common value
# print(titanic.groupby('Embarked').size())
# Embarked
# C    168
# Q     77
# S    644
embarked = titanic.groupby('Embarked').size().sort_values(ascending=False)
titanic["Embarked"] = titanic["Embarked"].fillna(list(dict(embarked))[0])
# and then map these values to integers
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
#   Replace Age NaN with mean OR median
mean = round(float(pd.DataFrame(titanic.describe())[['Age']].loc['mean']), 2)
# print(mean)                     # 29.7
# print(titanic["Age"].median())  # 28.0
titanic["Age"] = titanic["Age"].fillna(mean)
    # titanic["Age_noNaN"] = 0
    # titanic_no_NaN = titanic
    # titanic = titanic.drop("Age_noNaN", axis=1)
    # titanic_no_NaN["Age_noNaN"] = titanic_no_NaN["Age_noNaN"].fillna(mean)
    # titanic_no_NaN = titanic_no_NaN.drop("Age", axis=1)
    # titanic_no_NaN.rename(columns={'Age_noNaN': 'Age'}, inplace=True)
# titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
#   Check again if there any NaN values in the columns
# columns = list(titanic.columns.values)
# print("\n2nd check for Null values: ...")
# for column in columns:
    # print(column, ": ", sum(pd.isnull(titanic[column])))
#   Results:
    # 2nd check for Null values: ...
    # Survived :  0
    # Pclass :  0
    # Age :  0
    # SibSp :  0
    # Parch :  0
    # Embarked :  0
    # Male :  0
    # Female :  0


# print(titanic.dtypes)

print(titanic.head(20))



#  Split the dataset
# =============================================================================
#   With Age
X = titanic.iloc[:, 1:len(titanic)]
y = titanic.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(
    X,y,random_state=42,train_size=0.75)
#   Without empty columns
drop_columns_with_empty_values = ["Age", "Embarked", "hasCabin"]
titanic_no_nan = titanic.drop(drop_columns_with_empty_values, axis=1)
print(titanic_no_nan.head(20))
X_no_nan = titanic_no_nan.iloc[:, 1:len(titanic_no_nan)]
y_no_nan = titanic_no_nan.iloc[:, 0]
X_train_no_nan, X_test_no_nan, y_train_no_nan, y_test_no_nan = train_test_split(
    X_no_nan,y_no_nan,random_state=42,train_size=0.75)

# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================
# scaler =
scaler = MinMaxScaler()
scaler.fit(X=X_train)
scaled_features = pd.DataFrame(scaler.transform(X))
print(scaled_features)
# _no_nan
scaler_no_nan = MinMaxScaler()
scaler_no_nan.fit(X=X_no_nan)
scaled_features_no_nan = pd.DataFrame(scaler_no_nan.transform(X_no_nan))
print(scaled_features_no_nan)

# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer. 
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================
# imputer =
# imputer = KNNImputer(n_neighbors=3)
# imputer.fit_transform(X)



# Create your KNeighborsClassifier models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================
n_neighbours = 200
num_of_neighbours = list(np.arange(1, 201, 1))
#
f1_score_u_2 = []
f1_score_d_2 = []
f1_score_u_1 = []
f1_score_d_1 = []
f1_score_u_any = []
f1_score_d_any = []
#   _no_nan
f1_score_u_2_no_nan = []
f1_score_d_2_no_nan = []
f1_score_u_1_no_nan = []
f1_score_d_1_no_nan = []
f1_score_u_any_no_nan = []
f1_score_d_any_no_nan = []
weighted = ["uniform", "distance"]
p_list = [1, 2, "any"]

for neighbours in range(1, n_neighbours + 1):
    for weight in weighted:
        for p_value in p_list:
            if p_value != "any":
                knn = KNeighborsClassifier(
                    n_neighbors=neighbours,
                    weights=weight,
                    metric='minkowski',
                    p=p_value  #p = 1, manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
                    # For arbitrary p,
                )
                knn.fit(X_train, y_train)
                y_predicted = knn.predict(X_test)
                if weight == "uniform":
                    if p_value == 1:
                        f1_score_u_1.append(metrics.f1_score(y_test, y_predicted, average="macro"))
                    elif p_value == 2:
                        f1_score_u_2.append(metrics.f1_score(y_test, y_predicted, average="macro"))
                elif weight == "distance":
                    if p_value == 1:
                        f1_score_d_1.append(metrics.f1_score(y_test, y_predicted, average="macro"))
                    elif p_value == 2:
                        f1_score_d_2.append(metrics.f1_score(y_test, y_predicted, average="macro"))
                knn.fit(X_train_no_nan, y_train_no_nan)
                y_predicted_no_nan = knn.predict(X_test_no_nan)
                if weight == "uniform":
                    if p_value == 1:
                        f1_score_u_1_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))
                    elif p_value == 2:
                        f1_score_u_2_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))
                elif weight == "distance":
                    if p_value == 1:
                        f1_score_d_1_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))
                    elif p_value == 2:
                        f1_score_d_2_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))
            else:
                knn = KNeighborsClassifier(
                    n_neighbors=neighbours,
                    weights=weight,
                    metric='minkowski'
                )
                knn.fit(X_train, y_train)
                y_predicted = knn.predict(X_test)
                if weight == "uniform":
                    f1_score_u_any.append(metrics.f1_score(y_test, y_predicted, average="macro"))
                elif weight == "distance":
                    f1_score_d_any.append(metrics.f1_score(y_test, y_predicted, average="macro"))
                knn.fit(X_train_no_nan, y_train_no_nan)
                y_predicted_no_nan = knn.predict(X_test_no_nan)
                if weight == "uniform":
                    f1_score_u_any_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))
                elif weight == "distance":
                    f1_score_d_any_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))

# print(f1_score_u_2)
# print(f1_score_d_2)
# print(f1_score_u_1)
# print(f1_score_d_1)
# print(f1_score_u_any)
# print(f1_score_d_any)
# print(num_of_neighbours)
#
# print(f1_score_u_2_no_nan)
# print(f1_score_d_2_no_nan)
# print(f1_score_u_1_no_nan)
# print(f1_score_d_1_no_nan)
# print(f1_score_u_any_no_nan)
# print(f1_score_d_any_no_nan)
# print(num_of_neighbours)


print(max(enumerate(f1_score_u_2, 1), key=(lambda x: x[1])))
print(max(enumerate(f1_score_d_2, 1), key=(lambda x: x[1])))
print(max(enumerate(f1_score_u_1, 1), key=(lambda x: x[1])))
print(max(enumerate(f1_score_d_1, 1), key=(lambda x: x[1])))
print(max(enumerate(f1_score_u_any, 1), key=(lambda x: x[1])))
print(max(enumerate(f1_score_d_any, 1), key=(lambda x: x[1])))

print(max(enumerate(f1_score_u_2_no_nan, 1), key=(lambda x: x[1])))
print(max(enumerate(f1_score_d_2_no_nan, 1), key=(lambda x: x[1])))
print(max(enumerate(f1_score_u_1_no_nan, 1), key=(lambda x: x[1])))
print(max(enumerate(f1_score_d_1_no_nan, 1), key=(lambda x: x[1])))
print(max(enumerate(f1_score_u_any_no_nan, 1), key=(lambda x: x[1])))
print(max(enumerate(f1_score_d_any_no_nan, 1), key=(lambda x: x[1])))

# print("Decision Tree - Model Evaluation: ")
# print("Accuracy: ", metrics.accuracy_score(y_test, y_predicted))
# print("Precision:", metrics.precision_score(y_test, y_predicted, average="macro"))
# print("Recall: ", metrics.recall_score(y_test, y_predicted, average="macro"))
# print("F1: ", metrics.f1_score(y_test, y_predicted, average="macro"))

# Plot the F1 performance results for any combination οf parameter values of your choice.
# If you want to do the hard task, also plot the F1 results with/without imputation (in the same figure)
# =============================================================================
# plt.title('k-Nearest Neighbors (Weights = '<?>', Metric = '<?>', p = <?>)')
# plt.plot(f1_impute, label='with impute')
# plt.plot(f1_no_impute, label='without impute')
# plt.legend()
# plt.xlabel('Number of neighbors')
# plt.ylabel('F1')
# plt.show()

# fig, axs = plt.subplots(3, 2, sharex='col', sharey='row',
#                         gridspec_kw={'hspace': 0, 'wspace': 0})

fig, axs = plt.subplots(3, 2,
                        gridspec_kw={'hspace': 1, 'wspace': 0.2})

axs[0, 0].plot(num_of_neighbours, f1_score_u_2, 'tab:orange', label='with filled NaN')
axs[0, 0].plot(num_of_neighbours, f1_score_u_2_no_nan, label='no NaN cols')
axs[0, 0].set_title('Uniform, p=2')
axs[0, 1].plot(num_of_neighbours, f1_score_d_2, 'tab:orange')
axs[0, 1].plot(num_of_neighbours, f1_score_d_2_no_nan)
axs[0, 1].set_title('Distance, p=2')
axs[1, 0].plot(num_of_neighbours, f1_score_u_1, 'tab:orange')
axs[1, 0].plot(num_of_neighbours, f1_score_u_1_no_nan)
axs[1, 0].set_title('Uniform, p=1')
axs[1, 1].plot(num_of_neighbours, f1_score_d_1, 'tab:orange')
axs[1, 1].plot(num_of_neighbours, f1_score_d_1_no_nan)
axs[1, 1].set_title('Distance, p=1')
axs[2, 0].plot(num_of_neighbours, f1_score_u_any, 'tab:orange')
axs[2, 0].plot(num_of_neighbours, f1_score_u_any_no_nan)
axs[2, 0].set_title('Uniform, p=any')
axs[2, 1].plot(num_of_neighbours, f1_score_d_any, 'tab:orange')
axs[2, 1].plot(num_of_neighbours, f1_score_d_any_no_nan)
axs[2, 1].set_title('Distance, p=any')

# for ax in axs.flat:
#     ax.set(xlabel='k-NN number of neighbours', ylabel='F1 Score')

fig.suptitle('X: F1 Score Y:num of neighbours, k-NN')


# fig.xlabel='k-NN number of neighbours'
# fig.ylabel='F1 Score'

fig.legend(shadow=True, fancybox=True, loc='best')

# # Save plot
plt.savefig('k-NN F1 Scoreshgm09kl.png')

# # Show plot
plt.show()

