#   Dataset Link:
#   https://drive.google.com/drive/folders/1d5NCf33sX4ikTtyG0H8A4I18V-mGIdWm
#   Dataset Info:
#
# $ pipenv shell
# $ jupyter notebook
# =============================================================================
#  Imports
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, NuSVR



from sklearn.metrics import mean_absolute_error, mean_squared_error #, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest
from matplotlib import pyplot
from sklearn.feature_selection import f_regression

# from classifiers import results
from models import results
import warnings

warnings.filterwarnings("ignore")
# =============================================================================
#  Read the dataset
# =============================================================================
dataset = pd.read_csv("Data/fuel_emissions.csv",
                      sep=',',
                      header=0,  # no header, alternative header = header_col
                      index_col=None  # no index, alternative header = index_row
                      )
# print(len(dataset))
# print(dataset.head(5))
# print(dataset.info())
# print(dataset.describe())
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# =============================================================================
#  Change position of 'target' col
# =============================================================================
# nulls = dataset.isnull().sum()
# print(nulls)

# =============================================================================
#  Drop cols with na > round((len(dataset) / 3), 0)
# =============================================================================
# drop_columns_with_empty_values = ["file"]
# dataset = dataset.drop(drop_columns_with_empty_values, axis=1)
dataset = dataset.dropna(thresh=len(dataset) - (round((len(dataset) / 3), 0)), axis=1)
drop_columns = ["file", "manufacturer", "model", "description"]
dataset = dataset.drop(drop_columns, axis=1)
#
transmission_type_high = dataset.groupby('transmission_type').size().sort_values(ascending=False)
dataset["transmission_type"] = dataset["transmission_type"].fillna(list(dict(transmission_type_high))[0])
#
transmission_high = dataset.groupby('transmission').size().sort_values(ascending=False)
dataset["transmission"] = dataset["transmission"].fillna(list(dict(transmission_high))[0])
#
dataset = dataset[dataset.fuel_cost_12000_miles != 0]
dataset.dropna(subset=["fuel_cost_12000_miles"], inplace=True)
#
# print(dataset.mean())
dataset = dataset.fillna(dataset.mean())
#
# print(dataset.info())
# print(dataset.isnull().sum())
dataset.sort_values("fuel_cost_12000_miles")
# print(dataset.head(20))
# dataset.to_csv('b.csv')
#
pd.get_dummies(dataset, columns=['transmission_type']).head()
print(dataset['transmission_type'].value_counts())
types = {"transmission_type":
             {"Manual": 0,
              "Automatic": 1}}
dataset = dataset.replace(types)
#
print(dataset['fuel_type'].value_counts())
#
dataset["transmission_M6"] = np.where(dataset["transmission"].str.contains("M6"), 1, 0)
dataset["transmission_M5"] = np.where(dataset["transmission"].str.contains("M5"), 1, 0)
dataset["transmission_A6"] = np.where(dataset["transmission"].str.contains("A6"), 1, 0)
dataset["transmission_A5"] = np.where(dataset["transmission"].str.contains("A5"), 1, 0)
dataset["transmission_A7"] = np.where(dataset["transmission"].str.contains("A7"), 1, 0)
dataset["transmission_A4"] = np.where(dataset["transmission"].str.contains("A4"), 1, 0)
dataset["transmission_D6"] = np.where(dataset["transmission"].str.contains("D6"), 1, 0)
dataset["transmission_CVT"] = np.where(dataset["transmission"].str.contains("CVT"), 1, 0)
dataset["transmission_QM6"] = np.where(dataset["transmission"].str.contains("QM6"), 1, 0)
dataset["transmission_A8"] = np.where(dataset["transmission"].str.contains("A8"), 1, 0)
dataset["transmission_5MT"] = np.where(dataset["transmission"].str.contains("5MT"), 1, 0)
dataset["transmission_6MT"] = np.where(dataset["transmission"].str.contains("6MT"), 1, 0)
dataset["transmission_D7"] = np.where(dataset["transmission"].str.contains("D7"), 1, 0)
dataset["transmission_QA6"] = np.where(dataset["transmission"].str.contains("QA6"), 1, 0)
dataset["transmission_AV"] = np.where(dataset["transmission"].str.contains("AV"), 1, 0)
dataset["transmission_6AT"] = np.where(dataset["transmission"].str.contains("6AT"), 1, 0)
dataset["transmission_5AT"] = np.where(dataset["transmission"].str.contains("5AT"), 1, 0)
dataset["transmission_QD7"] = np.where(dataset["transmission"].str.contains("QD7"), 1, 0)

dataset = dataset.drop("transmission", axis=1)
#
dataset["fuel_type_Petrol"] = np.where(dataset["fuel_type"].str.contains("Petrol"), 1, 0)
dataset["fuel_type_Diesel"] = np.where(dataset["fuel_type"].str.contains("Diesel"), 1, 0)
print(dataset['fuel_type_Petrol'].value_counts())
print(dataset['fuel_type_Diesel'].value_counts())
dataset = dataset.drop("fuel_type", axis=1)
#
# print(dataset['transmission'].value_counts())
# print(dataset.head(5))





# drop_columns = ["fuel_type_Petrol", "fuel_type_Diesel", "co_emissions", "nox_emissions",
#                 "year", "euro_standard", "transmission_type", "nox_emissions", "noise_level"]
# dataset = dataset.drop(drop_columns, axis=1)


# =============================================================================
#  Split the dataset
# =============================================================================
dataset = dataset[[col for col in dataset.columns if col != 'fuel_cost_12000_miles'] + ['fuel_cost_12000_miles']]
print(dataset.head(5))
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=11)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train[0])

# =============================================================================
#  Featrue Selection
# =============================================================================
# def select_features(X_train, y_train, X_test):
# 	# configure to select all features
# 	fs = SelectKBest(score_func=f_regression, k='all')
# 	# learn relationship from training data
# 	fs.fit(X_train, y_train)
# 	# transform train input data
# 	X_train_fs = fs.transform(X_train)
# 	# transform test input data
# 	X_test_fs = fs.transform(X_test)
# 	return X_train_fs, X_test_fs, fs
#
#
# X_train_fs, X_test_fs, fs = select_features(X_train, Y_train, X_test)
# # what are scores for the features
# for i in range(len(fs.scores_)):
# 	print('Feature %d: %f' % (i, fs.scores_[i]))
# # plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()

results(X_train, X_test, Y_train, Y_test)
