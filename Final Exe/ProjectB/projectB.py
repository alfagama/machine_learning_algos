#   Dataset Link:
#   https://drive.google.com/drive/folders/1d5NCf33sX4ikTtyG0H8A4I18V-mGIdWm
#   Dataset Info:
#
# $ pipenv shell
# $ jupyter notebook
# =============================================================================
#  Imports
# =============================================================================
#   import method for results!
from classifiers import results
#
import numpy as np
import pandas as pd
#
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
#
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

#
py.init_notebook_mode(connected=True)
#   feature selection
from sklearn.feature_selection import SelectKBest
from matplotlib import pyplot
from sklearn.feature_selection import f_regression
import warnings

#
warnings.filterwarnings("ignore")
# =============================================================================
#  Read the dataset
# =============================================================================
#   reading dataset with header -> column 1 in excel
dataset = pd.read_csv("Data/fuel_emissions.csv",
                      sep=',',
                      header=0,  # no header, alternative header = header_col
                      index_col=None  # no index, alternative header = index_row
                      )
#   print everything from now on!
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#   check length of dataset
# print(len(dataset))
#   check dataset values
# print(dataset.head(5))
#   see info
# print(dataset.info())
#   describe
# print(dataset.describe())
#   check empty fields in all cols
# print(dataset.isnull().sum())
# =============================================================================
#
# =============================================================================
#   Drop cols with na > round((len(dataset) / 3), 0) -> Too many empty fields
dataset = dataset.dropna(thresh=len(dataset) - (round((len(dataset) / 3), 0)), axis=1)
#   Drop cols that offer no information
drop_columns = ["file", "manufacturer", "model", "description"]
dataset = dataset.drop(drop_columns, axis=1)
#   Drop rows that target (fuel_cost_12000_miles) has value  = 0
dataset = dataset[dataset.fuel_cost_12000_miles != 0]
#   Drop rows that target (fuel_cost_12000_miles) has no value
dataset.dropna(subset=["fuel_cost_12000_miles"], inplace=True)
#   Fill empty values in transmission_type with max value
transmission_type_high = dataset.groupby('transmission_type').size().sort_values(ascending=False)
dataset["transmission_type"] = dataset["transmission_type"].fillna(list(dict(transmission_type_high))[0])
#   Fill empty values in transmission with max value
transmission_high = dataset.groupby('transmission').size().sort_values(ascending=False)
dataset["transmission"] = dataset["transmission"].fillna(list(dict(transmission_high))[0])
#   Fill the remaining int/float fields in each col with .mean()
dataset = dataset.fillna(dataset.mean())
#   Replace in col transmission_type. Manual -> 0, Automatic -> 1 (since it's binary there is no problem)
pd.get_dummies(dataset, columns=['transmission_type']).head()
print(dataset['transmission_type'].value_counts())
types = {"transmission_type": {"Manual": 0, "Automatic": 1}}
dataset = dataset.replace(types)
#   Print 'fuel_type'.value_counts()
# print(dataset['fuel_type'].value_counts())
#       Petrol                      18261
#       Diesel                      14357
#       Petrol / E85 (Flex Fuel)      136
#       Petrol Hybrid                 126
#       Petrol / E85                   72
#       LPG                            49
#       Diesel Electric                22
#       Petrol Electric                17
#       LPG / Petrol                    9
#       Electricity                     7
#       CNG                             6
#       Electricity/Petrol              5
#       Electricity/Diesel              1
#   Since there are only 2 major classes and the rest are < 5%, plus some of the small classes also belong to
#   either Petrol or Diesel, -> create 2 new cols. Petrol with 1 or 0, and Diesel with 1 or 0.
#   Those with 0 in both are LPG + Electricity + CNG = 62 / 33068
dataset["fuel_type_Petrol"] = np.where(dataset["fuel_type"].str.contains("Petrol"), 1, 0)
dataset["fuel_type_Diesel"] = np.where(dataset["fuel_type"].str.contains("Diesel"), 1, 0)
# print(dataset['fuel_type_Petrol'].value_counts())
# print(dataset['fuel_type_Diesel'].value_counts())
#   Drop the original fuel_type col
dataset = dataset.drop("fuel_type", axis=1)
#   One-Hot-Encoding -> 74 unique values
#   Choosing the first ones to create new columns with 0/1
dataset["transmission_M6"] = np.where(dataset["transmission"].str.contains("M6"), 1, 0)    # M6            9720
dataset["transmission_M5"] = np.where(dataset["transmission"].str.contains("M5"), 1, 0)    # M5            7535
dataset["transmission_A6"] = np.where(dataset["transmission"].str.contains("A6"), 1, 0)    # A6            3221
dataset["transmission_A5"] = np.where(dataset["transmission"].str.contains("A5"), 1, 0)    # A5            2582
dataset["transmission_A7"] = np.where(dataset["transmission"].str.contains("A7"), 1, 0)    # A7            1693
dataset["transmission_A4"] = np.where(dataset["transmission"].str.contains("A4"), 1, 0)    # A4            1322
# maybe cutting point? -> almost no difference in DT & Bagging
# dataset["transmission_D6"] = np.where(dataset["transmission"].str.contains("D6"), 1, 0)    # D6             731
# dataset["transmission_CVT"] = np.where(dataset["transmission"].str.contains("CVT"), 1, 0)  # CVT            666
# dataset["transmission_QM6"] = np.where(dataset["transmission"].str.contains("QM6"), 1, 0)  # QM6            624
# dataset["transmission_A8"] = np.where(dataset["transmission"].str.contains("A8"), 1, 0)    # A8             606
# dataset["transmission_5MT"] = np.where(dataset["transmission"].str.contains("5MT"), 1, 0)  # 5MT            555
# dataset["transmission_6MT"] = np.where(dataset["transmission"].str.contains("6MT"), 1, 0)  # 6MT            545
# dataset["transmission_D7"] = np.where(dataset["transmission"].str.contains("D7"), 1, 0)    # D7             449
# dataset["transmission_QA6"] = np.where(dataset["transmission"].str.contains("QA6"), 1, 0)  # QA6            372
# dataset["transmission_AV"] = np.where(dataset["transmission"].str.contains("AV"), 1, 0)    # AV             368
# dataset["transmission_6AT"] = np.where(dataset["transmission"].str.contains("6AT"), 1, 0)  # 6AT            203
# dataset["transmission_5AT"] = np.where(dataset["transmission"].str.contains("5AT"), 1, 0)  # 5AT            156
# dataset["transmission_QD7"] = np.where(dataset["transmission"].str.contains("QD7"), 1, 0)  # QD7            132
#   Drop the original transmission col
dataset = dataset.drop("transmission", axis=1)

# =============================================================================
#  Split the dataset into X & Y
# =============================================================================
#   Move col fuel_cost_12000_miles to the end
dataset = dataset[[col for col in dataset.columns if col != 'fuel_cost_12000_miles'] + ['fuel_cost_12000_miles']]
# print(dataset.head(5))
#   Split into X and Y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1:]

# =============================================================================
#  Split the dataset into Train & Test
# =============================================================================
#   Split into Train & Test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=11)
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)


# =============================================================================
#  Feature Selection
# =============================================================================
# def select_features(X_train, y_train, X_test):
#     #   configure to select all features
#     fs = SelectKBest(score_func=f_regression, k='all')
#     #   learn relationship from training data
#     fs.fit(X_train, y_train)
#     #   transform train input data
#     X_train_fs = fs.transform(X_train)
#     #   transform test input data
#     X_test_fs = fs.transform(X_test)
#     return X_train_fs, X_test_fs, fs
#
#
# X_train_fs, X_test_fs, fs = select_features(X_train, Y_train, X_test)
# #  what are scores for the features
# for i in range(len(fs.scores_)):
#     print('Feature %d: %f' % (i, fs.scores_[i]))
# #  plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()

# =============================================================================
#  Drop columns based on Featrue Selection (..not good idea!)
# =============================================================================
# drop_columns = ["fuel_type_Petrol", "fuel_type_Diesel", "co_emissions", "nox_emissions",
#                 "year", "euro_standard", "transmission_type", "nox_emissions", "noise_level"]
# dataset = dataset.drop(drop_columns, axis=1)

# =============================================================================
#  Scale
# =============================================================================
#   Tried MinMaxScaler -> worse results than StandardScaler
# scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
#   StandardScaler()
scaler = StandardScaler()
#   Transform -> X
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train[0])


# =============================================================================
#  PCA
# =============================================================================
def pca_method(train, test, y_train, y_test):
    n_comp = [5, 10, 15, 20]
    for comp in n_comp:
        pca = PCA(n_components=comp,
                  copy=True,
                  whiten=False,
                  svd_solver='auto',
                  tol=0.0,
                  iterated_power='auto',
                  random_state=None)
        #   fit train
        train_pca = pca.fit_transform(train)
        #   fit test
        test_pca = pca.transform(test)
        #   results for
        print("---PCA-------------------", comp)
        results(train_pca, test_pca, y_train, y_test)


# =============================================================================
#  Results
# =============================================================================
#   Results without PCA
results(X_train, X_test, Y_train, Y_test)
#   Results with PCA
pca_method(X_train, X_test, Y_train, Y_test)
