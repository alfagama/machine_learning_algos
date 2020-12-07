import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

htru2 = pd.read_csv("HTRU_2.csv",
                      sep=',',
                      header=None)
pd.set_option('display.max_columns', None)

print(htru2.head(5))

X = htru2.iloc[:, -len(htru2):-1]
y = htru2.iloc[:, -1]

# a = 0
# b = 0
# for i in y:
#     if i == 1:
#         a += 1
#     elif i == 0:
#         b += 1
#     else:
#         print(i)
# print(a) # 1639
# print(b) # 16259

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
print("AUC: ", metrics.auc(y_test, y_predicted))
