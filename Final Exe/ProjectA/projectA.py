#   Dataset Link:
#   https://drive.google.com/drive/folders/1J4MQNwbOTz_YOWvMeVIubeXqo54BBO1R
#   Dataset Info:
#   http://kdd.ics.uci.edu/databases/kddcup99/task.html
# =============================================================================
#  Imports
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from projectA_clustering import *
from projectA_silhuette import silhuette_score
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# =============================================================================
#  Read the dataset
# =============================================================================
#   -----reading train_set with header -> column 1 in excel
train_set = pd.read_csv("Data/NSL-KDDTrain.csv",
                        sep=',',
                        header=0,  # no header, alternative header = header_col
                        index_col=None,  # no index, alternative header = index_row
                        )
#   -----check length of dataset
print(len(train_set))  # 125973
#   -----check dataset values
# print(train_set.head(5))
#   -----reading test_set with header -> column 1 in excel
test_set = pd.read_csv("Data/NSL-KDDTest.csv",
                       sep=',',
                       header=0,  # no header, alternative header = header_col
                       index_col=None,  # no index, alternative header = index_row
                       )
#   -----check length of dataset
print(len(test_set))  # 22544
#   -----check dataset values
# print(test_set.head(5))
#   -----see info
# print(train_set.info())  # dtypes: float64(15), int64(23), object(3)
# print(test_set.info())  # dtypes: float64(15), int64(23), object(4)
#   -----describe
# print(train_set.describe())
# print(test_set.describe())
#   -----check empty fields in all cols
# print(train_set.isnull().sum())  # no null
# print(test_set.isnull().sum())  # no null
#   -----see values in categorical columns to see how to handle them
# print(train_set['protocol_type'].value_counts())  # 3
# tcp     102689
# udp      14993
# icmp      8291
# Name: protocol_type, dtype: int64
# print(test_set['protocol_type'].value_counts())  # 3
# tcp     18880
# udp      2621
# icmp     1043
# Name: protocol_type, dtype: int64
# print(train_set['service'].value_counts())  # 70
# http           40338
# private        21853
# domain_u        9043
# smtp            7313
# ftp_data        6860
# eco_i           4586
# other           4359
# ecr_i           3077
# telnet          2353
# finger          1767
# ftp             1754
# auth             955
# ..
# Name: service, dtype: int64
# print(test_set['service'].value_counts())  # 64
# http           7853
# private        4774
# telnet         1626
# pop_3          1019
# smtp            934
# domain_u        894
# ftp_data        851
# other           838
# ecr_i           752
# ftp             692
# imap4           306
# ..
# Name: service, dtype: int64
# print(train_set['flag'].value_counts())  # 11
# SF        74945
# S0        34851
# REJ       11233
# RSTR       2421
# RSTO       1562
# S1          365
# SH          271
# S2          127
# RSTOS0      103
# S3           49
# OTH          46
# Name: flag, dtype: int64
# print(test_set['flag'].value_counts())  # 11
# SF        14875
# REJ        3850
# S0         2013
# RSTO        773
# RSTR        669
# S3          249
# SH           73
# S1           21
# S2           15
# OTH           4
# RSTOS0        2
# Name: flag, dtype: int64

# =============================================================================
# Small dataset -> many algorithms take hours to finish -> yes, i should buy a
# new pc, but GPUs are hard to find now in reasonable prices! ;(
# =============================================================================
# train_set = shuffle(train_set)
# train_set = train_set.head(10000)
# print(len(train_set))

# =============================================================================
# Preprocessing
# 5% -> 6.298
# =============================================================================
#   TRAIN SET------------------------------------------------------------------------------------------------------
#   -----One-Hot-Encoding -> column 'flag' -> 11 unique values
train_set["flag_OTH"] = np.where(train_set["flag"].str.contains("OTH"), 1, 0)           # OTH          46
train_set["flag_S3"] = np.where(train_set["flag"].str.contains("S3"), 1, 0)             # S3           49
train_set["flag_RSTOS0"] = np.where(train_set["flag"].str.contains("RSTOS0"), 1, 0)     # RSTOS0      103
train_set["flag_S2"] = np.where(train_set["flag"].str.contains("S2"), 1, 0)             # S2          127
train_set["flag_SH"] = np.where(train_set["flag"].str.contains("SH"), 1, 0)             # SH          271
train_set["flag_S1"] = np.where(train_set["flag"].str.contains("S1"), 1, 0)             # S1          365
train_set["flag_RSTO"] = np.where(train_set["flag"].str.contains("RSTO"), 1, 0)         # RSTO       1562
train_set["flag_RSTR"] = np.where(train_set["flag"].str.contains("RSTR"), 1, 0)         # RSTR       2421
train_set["flag_REJ"] = np.where(train_set["flag"].str.contains("REJ"), 1, 0)           # REJ       11233
train_set["flag_S0"] = np.where(train_set["flag"].str.contains("S0"), 1, 0)             # S0        34851
train_set["flag_SF"] = np.where(train_set["flag"].str.contains("SF"), 1, 0)             # SF        74945
#   -----drop old flag column
train_set = train_set.drop("flag", axis=1)
#   -----One-Hot-Encoding -> column 'protocol_type' -> 3 unique values
train_set["protocol_type_tcp"] = np.where(train_set["protocol_type"].str.contains("tcp"), 1, 0)     # tcp     102689
train_set["protocol_type_udp"] = np.where(train_set["protocol_type"].str.contains("udp"), 1, 0)     # udp      14993
train_set["protocol_type_icmp"] = np.where(train_set["protocol_type"].str.contains("icmp"), 1, 0)   # icmp      8291
#   -----drop old protocol_type column
train_set = train_set.drop("protocol_type", axis=1)
#   -----One-Hot-Encoding -> column 'protocol_type' -> 70 unique values
train_set["service_http"] = np.where(train_set["service"].str.contains("http"), 1, 0)           # http           40338
train_set["service_private"] = np.where(train_set["service"].str.contains("private"), 1, 0)     # private        21853
train_set["service_domain_u"] = np.where(train_set["service"].str.contains("domain_u"), 1, 0)   # domain_u        9043
train_set["service_smtp"] = np.where(train_set["service"].str.contains("smtp"), 1, 0)           # smtp            7313
train_set["service_ftp_data"] = np.where(train_set["service"].str.contains("ftp_data"), 1, 0)   # ftp_data        6860
train_set["service_eco_i"] = np.where(train_set["service"].str.contains("eco_i"), 1, 0)         # eco_i           4586
train_set["service_other"] = np.where(train_set["service"].str.contains("other"), 1, 0)         # other           4359
train_set["service_ecr_i"] = np.where(train_set["service"].str.contains("ecr_i"), 1, 0)         # ecr_i           3077
train_set["service_telnet"] = np.where(train_set["service"].str.contains("telnet"), 1, 0)       # telnet          2353
train_set["service_finger"] = np.where(train_set["service"].str.contains("finger"), 1, 0)       # finger          1767
train_set["service_ftp"] = np.where(train_set["service"].str.contains("ftp"), 1, 0)             # ftp             1754
# -----cutting point
# auth             955
# Z39_50           862
# uucp             780
# courier          734
# bgp              710
# train_set["service_pop_3"] = np.where(train_set["service"].str.contains("pop_3"), 1, 0)         # pop_3           264
# train_set["service_imap4"] = np.where(train_set["service"].str.contains("imap4"), 1, 0)         # imap4           647
# train_set["service_sunrpc"] = np.where(train_set["service"].str.contains("sunrpc"), 1, 0)       # sunrpc          381
#   -----drop old service column
train_set = train_set.drop("service", axis=1)

# print(train_set.info())

#   TEST SET------------------------------------------------------------------------------------------------------
#   -----One-Hot-Encoding -> column 'flag' -> 11 unique values
test_set["flag_OTH"] = np.where(test_set["flag"].str.contains("OTH"), 1, 0)         # OTH           4
test_set["flag_S3"] = np.where(test_set["flag"].str.contains("S3"), 1, 0)           # S3          249
test_set["flag_RSTOS0"] = np.where(test_set["flag"].str.contains("RSTOS0"), 1, 0)   # RSTOS0        2
test_set["flag_S2"] = np.where(test_set["flag"].str.contains("S2"), 1, 0)           # S2           15
test_set["flag_SH"] = np.where(test_set["flag"].str.contains("SH"), 1, 0)           # SH           73
test_set["flag_S1"] = np.where(test_set["flag"].str.contains("S1"), 1, 0)           # S1           21
test_set["flag_RSTO"] = np.where(test_set["flag"].str.contains("RSTO"), 1, 0)       # RSTO        773
test_set["flag_RSTR"] = np.where(test_set["flag"].str.contains("RSTR"), 1, 0)       # RSTR        669
test_set["flag_REJ"] = np.where(test_set["flag"].str.contains("REJ"), 1, 0)         # REJ        3850
test_set["flag_S0"] = np.where(test_set["flag"].str.contains("S0"), 1, 0)           # S0         2013
test_set["flag_SF"] = np.where(test_set["flag"].str.contains("SF"), 1, 0)           # SF        14875
#   -----drop old flag column
test_set = test_set.drop("flag", axis=1)
#   -----One-Hot-Encoding -> column 'protocol_type' -> 3 unique values
test_set["protocol_type_tcp"] = np.where(test_set["protocol_type"].str.contains("tcp"), 1, 0)    # tcp     18880
test_set["protocol_type_udp"] = np.where(test_set["protocol_type"].str.contains("udp"), 1, 0)    # udp      2621
test_set["protocol_type_icmp"] = np.where(test_set["protocol_type"].str.contains("icmp"), 1, 0)  # icmp     1043
#   -----drop old protocol_type column
test_set = test_set.drop("protocol_type", axis=1)
#   -----One-Hot-Encoding -> column 'protocol_type' -> 70 unique values
test_set["service_http"] = np.where(test_set["service"].str.contains("http"), 1, 0)           # http           7853
test_set["service_private"] = np.where(test_set["service"].str.contains("private"), 1, 0)     # private        4774
test_set["service_domain_u"] = np.where(test_set["service"].str.contains("domain_u"), 1, 0)   # domain_u        894
test_set["service_smtp"] = np.where(test_set["service"].str.contains("smtp"), 1, 0)           # smtp            934
test_set["service_ftp_data"] = np.where(test_set["service"].str.contains("ftp_data"), 1, 0)   # ftp_data        851
test_set["service_eco_i"] = np.where(test_set["service"].str.contains("eco_i"), 1, 0)         # eco_i           262
test_set["service_other"] = np.where(test_set["service"].str.contains("other"), 1, 0)         # other           838
test_set["service_ecr_i"] = np.where(test_set["service"].str.contains("ecr_i"), 1, 0)         # ecr_i           752
test_set["service_telnet"] = np.where(test_set["service"].str.contains("telnet"), 1, 0)       # telnet         1626
test_set["service_finger"] = np.where(test_set["service"].str.contains("finger"), 1, 0)       # finger          136
test_set["service_ftp"] = np.where(test_set["service"].str.contains("ftp"), 1, 0)             # ftp             692
# -----cutting point
# test_set["service_pop_3"] = np.where(test_set["service"].str.contains("pop_3"), 1, 0)         # pop_3          1019
# test_set["service_imap4"] = np.where(test_set["service"].str.contains("imap4"), 1, 0)         # imap4           306
# test_set["service_sunrpc"] = np.where(test_set["service"].str.contains("sunrpc"), 1, 0)       # sunrpc          159
#   -----drop old service column
test_set = test_set.drop("service", axis=1)
#   -----OrdinalEncoder to transform  0->attack, 1->normal
ord_enc = OrdinalEncoder()
test_set["label"] = ord_enc.fit_transform(test_set[["target"]])
test_set[["target", "label"]].head(11)
test_set = test_set.drop("target", axis=1)
# print(test_set.info())
# print(test_set.head(20))

# =============================================================================
#  Split the test set into test_set & target
# =============================================================================
#   -----split label from rest of test_set
target = test_set["label"]
test_set = test_set.drop("label", axis=1)
# print(target)

# =============================================================================
#  Feature Selection
# =============================================================================
#   -----well.. we could do in test_set, but that would be cheating
#   since test_set is unknown to us!

# =============================================================================
#  Results 1 - without scaling!
# =============================================================================
#   -----Results without scaling!
print("-----Results without scaling:...")
results(train_set, test_set, target)

# =============================================================================
#  Scale
# =============================================================================
#   -----StandardScaler()
scaler = StandardScaler()
#   -----MinMaxScaler()
# scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
#   -----Transform -> X
train_set = scaler.fit_transform(train_set)
test_set = scaler.transform(test_set)
# print(train_set[0])

# =============================================================================
#  Silhuette score
# =============================================================================
print("Silhuette score - Train set")
# silhuette_score(train_set)
print("Silhuette score - Test set")
# silhuette_score(test_set)

# =============================================================================
#  Results 2 - After scaling
# =============================================================================
#   -----Results without PCA, with scaling
# print(X_train)
print("-----Results after scaling:...")
results(train_set, test_set, target)

# =============================================================================
#  PCA
# =============================================================================
def pca(train_set, test_set, comp_num):
    # -----This was with a for loop, but now I know best n_components needed
    pca = PCA(n_components=comp_num,
              copy=True,
              whiten=False,
              svd_solver='auto',
              tol=1e-06,
              iterated_power='auto',
              random_state=11)
    #   fit train
    train_set_pca = pca.fit_transform(train_set)
    #   fit test
    test_set_pca = pca.transform(test_set)
    #   return
    return train_set_pca, test_set_pca


# =============================================================================
#  Results 3 - With PCA
# =============================================================================
print("-----Results with PCA:...")
train_after_pca, test_after_pca = pca(train_set, test_set, 5)
kmeans(train_after_pca, test_after_pca, target)
train_after_pca, test_after_pca = pca(train_set, test_set, 24)
minibatchkmeans(train_after_pca, test_after_pca, target)
train_after_pca, test_after_pca = pca(train_set, test_set, 1)
agglomerative(train_after_pca, test_after_pca, target)
train_after_pca, test_after_pca = pca(train_set, test_set, 1)
birch(train_after_pca, test_after_pca, target)
