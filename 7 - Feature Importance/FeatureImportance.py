# =============================================================================
# import stuff!
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.decomposition import PCA
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
# scaled features - ony for PCA (for some reason! doing tests..)
# =============================================================================
X_sf_train = X_train.values
X_std_train = StandardScaler().fit_transform(X_sf_train)
X_sf_test = X_test.values
X_std_test = StandardScaler().fit_transform(X_sf_test)

# =============================================================================
# PCA
# =============================================================================
pca = PCA(n_components=4)
# fir train
pca.fit(X_std_train)
X_4d_train = pca.transform(X_std_train)
# fit test
pca.fit(X_std_test)
X_4d_test = pca.transform(X_std_test)

# =============================================================================
# Random Forest Classifier
# =============================================================================
model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=3,
    random_state=1
)
model_pca = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=3,
    random_state=1
)

# =============================================================================
# fit
# =============================================================================
model.fit(X_train, y_train)
model_pca.fit(X_4d_train, y_train)

# =============================================================================
# predict
# =============================================================================
y_predicted = model.predict(X_test)
y_predicted_pca = model_pca.predict(X_4d_test)

# =============================================================================
# print metrics (ACC, PRE, REC, F1, AUC, ROC AUC)
# =============================================================================
def metrics_call(pred,name):
    print("Accuracy: ", metrics.accuracy_score(y_test, pred))
    print("Precision:", metrics.precision_score(y_test, pred, average="macro"))
    print("Recall: ", metrics.recall_score(y_test, pred, average="macro"))
    print("F1: ", metrics.f1_score(y_test, pred, average="macro"))
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, pred)
    print("AUC: ", metrics.auc(false_positive_rate, true_positive_rate))
    # may differ sometimes, link: https://stackoverflow.com/questions/31159157/different-result-with-roc-auc-score
    # -and-auc
    print("ROC AUC: ", metrics.roc_auc_score(y_test, pred))
    print("ROC Curve:", metrics.roc_curve(y_test, pred, pos_label=1))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
    # calculate roc curves
    len_y_test = [0 for _ in range(len(y_test))]
    ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, len_y_test)
    lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, pred)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    # plt.show()
    # save the plot
    fig_name = "Figs/ROC_Curve"+name+".png"
    plt.savefig(fig_name)
    plt.close()

print("Random Forest - Model Evaluation: ")
metrics_call(y_predicted, 'without PCA')
print("")
print("Random Forest - Model Evaluation - After PCA: ")
metrics_call(y_predicted_pca, 'with PCA')

# =============================================================================
# feature importance : method 1: Random Forest Built-in Feature Importance
# =============================================================================
print("")
print("Method 1: Random Forest Built-in Feature Importance")
features = [0, 1, 2, 3, 4, 5, 6, 7]
plt.bar(X.columns, model.feature_importances_, align='center', alpha=0.5)
plt.xticks(X.columns, features)
plt.xlabel('Column Number [0 - 7]')
plt.ylabel('% of Importance')
plt.title('Random Forest Built-in Feature Importance')
plt.savefig("Figs/Built-in Feature Importance.png")
# print(model.feature_importances_)
# plt.show()
plt.close()
list_method1 = []
for x in features:
    list_temp1 = [model.feature_importances_[x], x]
    list_method1.append(list_temp1)
print("Sorted: Random Forest Built-in Feature Importance")
print(sorted(list_method1, reverse=True))

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
# print(perm_importance.importances_mean)
# plt.show()
plt.close()
list_method2 = []
for x in features:
    list_temp2 = [perm_importance.importances_mean[x], x]
    list_method2.append(list_temp2)
print("Sorted: Permutation Based Feature Importance")
print(sorted(list_method2, reverse=True))

# =============================================================================
# print metrics (ACC, PRE, REC, F1, AUC, ROC AUC) using the 4 most important features
# =============================================================================
listed_features = sorted(list_method1, reverse=True)
most_important_features = []
most_important_features_number = []
new_df = pd.DataFrame(X)
for x in range(0, 4):
    most_important_features.append(listed_features[x])
for x in (i[1] for i in most_important_features):
    most_important_features_number.append(x)
drop_features = [x for x in features if x not in most_important_features_number]
for dropped in drop_features:
    new_df = new_df.drop([dropped], axis=1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    new_df, y, random_state=42, train_size=0.80)
model_with_4_features = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=3,
    random_state=1
)
model_with_4_features.fit(X_train2, y_train2)
y_predicted3 = model_with_4_features.predict(X_test2)
print("")
print("Random Forest - Model Evaluation - 4 best features: ")
metrics_call(y_predicted3, 'with 4 features')
