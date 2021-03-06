# =============================================================================
# import the KNeighborsClassifier
# if you want to do the hard task, also import the KNNImputer
import numpy as np
import random
import matplotlib.pyplot as plt
# import KNNImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import warnings

# =============================================================================
# Suppress warning:
# SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame
warnings.filterwarnings("ignore")
# =============================================================================
random.seed = 42
np.random.seed(666)
# =============================================================================
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
                      # index_col=None,  # no index, alternative header = index_row
                      # skiprows=0  # how many rows to skip / not include in read_csv
                      )
pd.set_option('display.max_columns', None)
#   make sure that it got read OK
# print("Number of data: ", len(titanic))
# print(titanic.head())
#   check data stats
# print(titanic.describe())

#   DROP unwanted columns that make no difference
drop_columns = ["PassengerId", "Name", "Ticket", "Fare"]
titanic = titanic.drop(drop_columns, axis=1)
#   Change Cabin to binary 0/1 if Cabin = Nan then hasCabin = 0, else hasCabin = 1
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

#print(titanic.dtypes)
# print(titanic.head(20))

# =============================================================================
#  Split the dataset
# =============================================================================
#   With Age
X = titanic.iloc[:, 1:len(titanic)]
y = titanic.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, train_size=0.75)
#   Without empty columns
drop_columns_with_empty_values = ["Age", "Embarked", "hasCabin"]
titanic_no_nan = titanic.drop(drop_columns_with_empty_values, axis=1)
# print(titanic_no_nan.head(20))
X_no_nan = titanic_no_nan.iloc[:, 1:len(titanic_no_nan)]
y_no_nan = titanic_no_nan.iloc[:, 0]
X_train_no_nan, X_test_no_nan, y_train_no_nan, y_test_no_nan = train_test_split(
    X_no_nan, y_no_nan, random_state=42, train_size=0.75)

# =============================================================================
# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================
# with NaN filled out
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# _no_nan
scaler_no_nan = MinMaxScaler(copy=True, feature_range=(0, 1))
X_train_no_nan = scaler.fit_transform(X_train_no_nan)
X_test_no_nan = scaler.transform(X_test_no_nan)
# =============================================================================
# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer. 
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================
# imputer = KNNImputer(n_neighbors=3)
# imputer.fit_transform(X)

# =============================================================================
# Create your KNeighborsClassifier models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================
n_neighbours = 200
num_of_neighbours = list(np.arange(1, 201, 1))
# filled NaN value tables
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
p_list = [1, 2, 3]

print("Running k-NN: ... ")
for neighbours in range(1, n_neighbours + 1):
    for weight in weighted:
        for p_value in p_list:
            knn = KNeighborsClassifier(
                n_neighbors=neighbours,
                weights=weight,
                metric='minkowski',
                p=p_value
                # p = 1, manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
            )
            #   Filled NaN columns
            knn.fit(X_train, y_train)
            y_predicted = knn.predict(X_test)
            if weight == "uniform":
                if p_value == 1:
                    f1_score_u_1.append(metrics.f1_score(y_test, y_predicted, average="macro"))
                elif p_value == 2:
                    f1_score_u_2.append(metrics.f1_score(y_test, y_predicted, average="macro"))
                elif p_value == 3:
                    f1_score_u_any.append(metrics.f1_score(y_test, y_predicted, average="macro"))
            elif weight == "distance":
                if p_value == 1:
                    f1_score_d_1.append(metrics.f1_score(y_test, y_predicted, average="macro"))
                elif p_value == 2:
                    f1_score_d_2.append(metrics.f1_score(y_test, y_predicted, average="macro"))
                elif p_value == 3:
                    f1_score_d_any.append(metrics.f1_score(y_test, y_predicted, average="macro"))
            #   no_NaN columns
            knn.fit(X_train_no_nan, y_train_no_nan)
            y_predicted_no_nan = knn.predict(X_test_no_nan)
            if weight == "uniform":
                if p_value == 1:
                    f1_score_u_1_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))
                elif p_value == 2:
                    f1_score_u_2_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))
                elif p_value == 3:
                    f1_score_u_any_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))
            elif weight == "distance":
                if p_value == 1:
                    f1_score_d_1_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))
                elif p_value == 2:
                    f1_score_d_2_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))
                elif p_value == 3:
                    f1_score_d_any_no_nan.append(metrics.f1_score(y_test_no_nan, y_predicted_no_nan, average="macro"))
print("Done! ...")

# -----------------------------------------------------------------------------------------------
df_final = pd.DataFrame(columns=['Parameters', 'Num_of_NN', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
df_filled_NaN_cols = pd.DataFrame(columns=['Parameters', 'Num_of_NN', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
# -----------------------------------------------------------------------------------------------
def kNN_call_with_best_neighbours(neighbours, weights, p, X_train_m, X_test_m, y_train_m, y_test_m):
    df_line = pd.DataFrame(columns=['Parameters', 'Num_of_NN', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    knn_m = KNeighborsClassifier(
        n_neighbors=neighbours,
        weights=weights,
        metric='minkowski',
        p=p
    )
    knn_m.fit(X_train_m, y_train_m)
    y_predicted_m = knn_m.predict(X_test_m)
    df_line = df_line.append({'Parameters': str(weights) + '-minkowski-p=' + str(p),
                              'Num_of_NN': neighbours,
                              'Accuracy': metrics.accuracy_score(y_test_m, y_predicted_m),
                              'Precision': metrics.precision_score(y_test_m, y_predicted_m, average="macro"),
                              'Recall': metrics.recall_score(y_test_m, y_predicted_m, average="macro"),
                              'F1 Score': metrics.f1_score(y_test_m, y_predicted_m, average="macro")
                              }, ignore_index=True)
    return df_line


# -----------------------------------------------------------------------------------------------
df_final = df_final.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_u_1_no_nan, 1), key=(lambda x: x[1]))[0], 'uniform', 1, X_train_no_nan, X_test_no_nan, y_train_no_nan, y_test_no_nan))
df_final = df_final.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_d_1_no_nan, 1), key=(lambda x: x[1]))[0], 'distance', 1, X_train_no_nan, X_test_no_nan, y_train_no_nan, y_test_no_nan))
df_final = df_final.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_u_2_no_nan, 1), key=(lambda x: x[1]))[0], 'uniform', 2, X_train_no_nan, X_test_no_nan, y_train_no_nan, y_test_no_nan))
df_final = df_final.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_d_2_no_nan, 1), key=(lambda x: x[1]))[0], 'distance', 2, X_train_no_nan, X_test_no_nan, y_train_no_nan, y_test_no_nan))
df_final = df_final.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_u_any_no_nan, 1), key=(lambda x: x[1]))[0], 'uniform', 3, X_train_no_nan, X_test_no_nan, y_train_no_nan, y_test_no_nan))
df_final = df_final.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_d_any_no_nan, 1), key=(lambda x: x[1]))[0], 'distance', 3, X_train_no_nan, X_test_no_nan, y_train_no_nan, y_test_no_nan))
# -----------------------------------------------------------------------------------------------
df_filled_NaN_cols = df_filled_NaN_cols.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_u_1, 1), key=(lambda x: x[1]))[0], 'uniform', 1, X_train, X_test, y_train, y_test))
df_filled_NaN_cols = df_filled_NaN_cols.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_d_1, 1), key=(lambda x: x[1]))[0], 'distance', 1, X_train, X_test, y_train, y_test))
df_filled_NaN_cols = df_filled_NaN_cols.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_u_2, 1), key=(lambda x: x[1]))[0], 'uniform', 2, X_train, X_test, y_train, y_test))
df_filled_NaN_cols = df_filled_NaN_cols.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_d_2, 1), key=(lambda x: x[1]))[0], 'distance', 2, X_train, X_test, y_train, y_test))
df_filled_NaN_cols = df_filled_NaN_cols.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_u_any, 1), key=(lambda x: x[1]))[0], 'uniform', 3, X_train, X_test, y_train, y_test))
df_filled_NaN_cols = df_filled_NaN_cols.append(kNN_call_with_best_neighbours(max(enumerate(f1_score_d_any, 1), key=(lambda x: x[1]))[0], 'distance', 3, X_train, X_test, y_train, y_test))
# -----------------------------------------------------------------------------------------------
print("DataFrame with results when we dropped columns containing NaN values: ... ")
print(df_final)
print()
print("DataFrame with results after filling NaN columns and not dropping them: ...")
print(df_filled_NaN_cols)
# =============================================================================
# Plot the F1 performance results for any combination οf parameter values of your choice.
# If you want to do the hard task, also plot the F1 results with/without imputation (in the same figure)
# =============================================================================
fig, axs = plt.subplots(3, 2,
                        sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0, 'wspace': 0})

axs[0, 0].plot(num_of_neighbours, f1_score_u_1, 'tab:orange', label='with filled NaN')
axs[0, 0].plot(num_of_neighbours, f1_score_u_1_no_nan, label='drop NaN cols')
axs[0, 1].plot(num_of_neighbours, f1_score_d_1, 'tab:orange')
axs[0, 1].plot(num_of_neighbours, f1_score_d_1_no_nan)
axs[1, 0].plot(num_of_neighbours, f1_score_u_2, 'tab:orange')
axs[1, 0].plot(num_of_neighbours, f1_score_u_2_no_nan)
axs[1, 1].plot(num_of_neighbours, f1_score_d_2, 'tab:orange')
axs[1, 1].plot(num_of_neighbours, f1_score_d_2_no_nan)
axs[2, 0].plot(num_of_neighbours, f1_score_u_any, 'tab:orange')
axs[2, 0].plot(num_of_neighbours, f1_score_u_any_no_nan)
axs[2, 1].plot(num_of_neighbours, f1_score_d_any, 'tab:orange')
axs[2, 1].plot(num_of_neighbours, f1_score_d_any_no_nan)

fig.suptitle('X:NumOfNeighbours Y:F1 Score')

fig.text(0.3, 0.04, 'Uniform', ha='center')
fig.text(0.7, 0.04, 'Distance', ha='center')
fig.text(0.01, 0.75, 'p=1', va='center')
fig.text(0.01, 0.5, 'p=2', va='center')
fig.text(0.01, 0.25, 'p=3', va='center')

fig.legend(shadow=True, fancybox=True, loc=1)

# # Save plot
plt.savefig('k-NN F1Score - All Results.png')

# # Show plot
plt.show()
# =============================================================================
