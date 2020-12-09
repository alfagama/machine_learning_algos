import pandas as pd
import shap as shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

htru2 = pd.read_csv("HTRU_2.csv",
                      sep=',',
                      header=None)
pd.set_option('display.max_columns', None)

# print(htru2.head(5))

X = htru2.iloc[:, -len(htru2):-1]
y = htru2.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, train_size=0.80)

model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=3,
    random_state=1
)

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)


print("Random Forest - Model Evaluation: ")
print("Accuracy: ", metrics.accuracy_score(y_test, y_predicted))
print("Precision:", metrics.precision_score(y_test, y_predicted, average="macro"))
print("Recall: ", metrics.recall_score(y_test, y_predicted, average="macro"))
print("F1: ", metrics.f1_score(y_test, y_predicted, average="macro"))
# print("AUC: ", metrics.auc(y_test, y_predicted))

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

print()
print("Method 2: Permutation Based Feature Importance")
perm_importance = permutation_importance(model, X_test, y_test)

#sorted_idx = perm_importance.importances_mean.argsort()
# plt.barh(boston.feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])
# plt.barh(X.columns, perm_importance.importances_mean)
# plt.xlabel("Permutation Importance")
# xmin, xmax = 0, 0
# for x in perm_importance.importances_mean:
#     print(x)
#     if xmin < x:
#         xmin = x
#     if xmax > x:
#         xmax = x
# # xmin = min([min(x_list) for x_list in perm_importance.importances_mean])
# # xmax = max([max(x_list) for x_list in perm_importance.importances_mean])
# plt.xlim(xmin, xmax)

# print(X.columns)
# plt.show()

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


