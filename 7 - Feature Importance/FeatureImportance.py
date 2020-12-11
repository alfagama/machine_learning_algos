# =============================================================================
# import stuff!
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# =============================================================================
# import stuff!
# =============================================================================
htru2 = pd.read_csv("HTRU_2.csv",
                      sep=',',
                      header=None)
pd.set_option('display.max_columns', None)

# =============================================================================
# set X and y
# =============================================================================
X = htru2.iloc[:, -len(htru2):-1]
y = htru2.iloc[:, -1]

# =============================================================================
# train/test sets
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, train_size=0.80)

# =============================================================================
# Random Forest Classifier
# =============================================================================
model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=3,
    random_state=1
)

# =============================================================================
# fit
# =============================================================================
model.fit(X_train, y_train)

# =============================================================================
# predict
# =============================================================================
y_predicted = model.predict(X_test)

# =============================================================================
# print metrics (ACC, PRE, REC, F1, AUC, ROC AUC)
# =============================================================================
print("Random Forest - Model Evaluation: ")
print("Accuracy: ", metrics.accuracy_score(y_test, y_predicted))
print("Precision:", metrics.precision_score(y_test, y_predicted, average="macro"))
print("Recall: ", metrics.recall_score(y_test, y_predicted, average="macro"))
print("F1: ", metrics.f1_score(y_test, y_predicted, average="macro"))
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_predicted)
print("AUC: ", metrics.auc(false_positive_rate, true_positive_rate))
# may differ sometimes, link: https://stackoverflow.com/questions/31159157/different-result-with-roc-auc-score-and-auc
print("ROC AUC: ", metrics.roc_auc_score(y_test, y_predicted))

# =============================================================================
# PCA
# =============================================================================


# =============================================================================
# print metrics (ACC, PRE, REC, F1, AUC, ROC AUC) after PCA
# =============================================================================
print("Random Forest - Model Evaluation - After PCA: ")
print("Accuracy: ", metrics.accuracy_score(y_test, y_predicted))
print("Precision:", metrics.precision_score(y_test, y_predicted, average="macro"))
print("Recall: ", metrics.recall_score(y_test, y_predicted, average="macro"))
print("F1: ", metrics.f1_score(y_test, y_predicted, average="macro"))
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_predicted)
print("AUC: ", metrics.auc(false_positive_rate, true_positive_rate))
# may differ sometimes, link: https://stackoverflow.com/questions/31159157/different-result-with-roc-auc-score-and-auc
print("ROC AUC: ", metrics.roc_auc_score(y_test, y_predicted))

# =============================================================================
# feature importance : method 1: Random Forest Built-in Feature Importance
# =============================================================================
print()
print("Method 1: Random Forest Built-in Feature Importance")
features = [0, 1, 2, 3, 4, 5, 6, 7]
plt.bar(X.columns, model.feature_importances_, align='center', alpha=0.5)
plt.xticks(X.columns, features)
plt.xlabel('Column Number [0 - 7]')
plt.ylabel('% of Importance')
plt.title('Random Forest Built-in Feature Importance')
plt.savefig("Figs/Built-in Feature Importance.png")
print(model.feature_importances_)
# plt.show()
plt.close()

# =============================================================================
# feature importance : method 2: Permutation Based Feature Importance
# =============================================================================
print()
print("Method 2: Permutation Based Feature Importance")
perm_importance = permutation_importance(model, X_test, y_test)
features = [0, 1, 2, 3, 4, 5, 6, 7]
plt.bar(X.columns, perm_importance.importances_mean, align='center', alpha=0.5)
plt.xticks(X.columns, features)
plt.xlabel('Column Number [0 - 7]')
plt.ylabel('% of Importance')
plt.title('Permutation Based Feature Importance')
plt.savefig("Figs/Permutation Importance.png")
print(perm_importance.importances_mean)
# plt.show()
plt.close()
