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
# print(titanic.describe()) # will be needed alter to fill NaN values

#   DROP unwanted columns that make no difference
drop_columns = ["PassengerId", "Name", "Ticket", "Fare"]
titanic = titanic.drop(drop_columns, axis=1)
# titanic = titanic.drop("PassengerId", axis=1)   # Id not wanted
# titanic = titanic.drop("Name", axis=1)          # Name not wanted
# titanic = titanic.drop("Ticket", axis=1)        # Number of ticket not wanted
# titanic = titanic.drop("Fare", axis=1)          # Cost of ticket not wanted
# print(titanic.head())
# print(titanic.dtypes)
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
    #titanic = titanic[['Pclass', 'Age', 'Male', 'Female',
    #                   'SibSp',  'Parch',  'Embarked',  'hasCabin',
    #                   'Survived']]
    #X = titanic.iloc[:, -len(titanic):-1]
    #y = titanic.iloc[:, -1]
X = titanic.iloc[:, 1:len(titanic)]
y = titanic.iloc[:, 0]
# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(
    X,y,random_state=42,train_size=0.7)

    # X_no_NaN = titanic_no_NaN.iloc[:, 1:len(titanic)]
    # y_no_NaN = titanic_no_NaN.iloc[:, 0]
    # # print(X)
    # # print(y)
    # X_train_no_NaN, X_test_no_NaN, y_train_no_NaN, y_test_no_NaN = train_test_split(
    #     X,y,random_state=42,train_size=0.7)

# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================
# scaler =
scaler = MinMaxScaler()
scaler.fit(X=X_train)
scaled_features = pd.DataFrame(scaler.transform(X))

print(scaled_features)

    # scaler_no_NaN = MinMaxScaler()
    # scaler_no_NaN.fit(X=X_train_no_NaN)
    # scaled_features_no_NaN = pd.DataFrame(scaler_no_NaN.transform(X_no_NaN))
    #
    # print(scaled_features_no_NaN)

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
knn = KNeighborsClassifier(
    n_neighbors=3,
    weights='uniform',  # /distance
    metric='minkowski',
    p=2, #p = 1, manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
    # For arbitrary p, minkowski_distance (l_p) is used.
    
)
knn.fit(X_train, y_train)
y_predicted = knn.predict(X_test)

print("Decision Tree - Model Evaluation: ")
print("Accuracy: ", metrics.accuracy_score(y_test, y_predicted))
print("Precision:", metrics.precision_score(y_test, y_predicted, average="macro"))
print("Recall: ", metrics.recall_score(y_test, y_predicted, average="macro"))
print("F1: ", metrics.f1_score(y_test, y_predicted, average="macro"))

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

