import pandas as pd
from scipy.stats import friedmanchisquare as fr

data_set = pd.read_csv("algo_performance.csv", sep=',')
c45 = data_set['C4.5']
oNN = data_set['1-NN']
mNB = data_set['NaiveBayes']
svm = data_set['Kernel']
cn2 = data_set['CN2']

stat, p = fr(c45, oNN, mNB, svm, cn2)
print('Statistics=%.3f, p=%.11f' % (stat, p))

alpha = [0.0001, 0.001, 0.01, 0.1, 0.5, 1]

for a in alpha:
    if p >= a:
        # print('Same distributions (fail to reject H0), for alpha = ', a)
        print('Algorithms do not have statistical difference, for alpha = ', a)
    elif p < a:
        # print('Different distributions (reject H0), for alpha = ', a)
        print('Algorithms have statistical difference, for alpha = ', a)
