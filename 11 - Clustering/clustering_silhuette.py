# =============================================================================
# import stuff!
# =============================================================================
from numpy.ma import array
from sklearn.model_selection import LeaveOneOut
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

# =============================================================================
# Read dataset
# =============================================================================
dataset = datasets.load_wine()
# X, y = datasets.load_wine(return_X_y=True)                        #split into X & Y

# =============================================================================
# DF
# =============================================================================
# X = pd.DataFrame(dataset.data, columns=dataset.feature_names)     #only X into df
df = pd.DataFrame(data=np.c_[dataset['data'], dataset['target']],
                  columns=dataset['feature_names'] + ['target'])
print(df.head())
print(df.shape)


