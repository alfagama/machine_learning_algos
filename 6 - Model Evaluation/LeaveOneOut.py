# =============================================================================
# import stuff!
# =============================================================================
from numpy.ma import array
from sklearn.model_selection import LeaveOneOut
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =============================================================================
# Read dataset
# =============================================================================
X, y = datasets.load_wine(return_X_y=True)

# =============================================================================
# Leave-One-Out Cross Validation
# =============================================================================
loocv = LeaveOneOut()

# =============================================================================
# For-Loop implementation for LOOCV
# =============================================================================
y_true, y_pred = list(), list()
for train, test in loocv.split(X):
    X_train, X_test = X[train, :], X[test, :]
    y_train, y_test = y[train], y[test]
    rfc = RandomForestClassifier(random_state=1)
    rfc.fit(X_train, y_train)
    y_ = rfc.predict(X_test)
    y_true.append(y_test[0])
    y_pred.append(y_[0])

# =============================================================================
# Prints
# =============================================================================
acc = accuracy_score(y_true, y_pred)
print('Accuracy: %.3f' % acc)
cnf_matrix = confusion_matrix(y_true, y_pred)
print("Evaluation for 3 classes")
print("Confusion Matrix:")
print(cnf_matrix)
print("One-of / Multinomial classification")

# =============================================================================
# Method fill_matrix()
# =============================================================================
def fill_matrix(class_name):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y)):
        if y_true[i] == class_name:
            if y_pred[i] == class_name:
                TP += 1
            else:
                FP += 1
        elif y_pred[i] == class_name and y_true[i] != class_name:
            FN += 1
        else:
            TN += 1
    return TP, FP, FN, TN


# =============================================================================
# Fill TP/FP/FN/TN
# =============================================================================
TP_0, FP_0, FN_0, TN_0 = fill_matrix(0)
TP_1, FP_1, FN_1, TN_1 = fill_matrix(1)
TP_2, FP_2, FN_2, TN_2 = fill_matrix(2)

contingency_array = array([[TP_0, FP_0, FN_0, TN_0],
                           [TP_1, FP_1, FN_1, TN_1],
                           [TP_2, FP_2, FN_2, TN_2]])

# =============================================================================
# Print contingency_array
# =============================================================================
for i in range(len(contingency_array)):
    print("Print Contingency table for class ", i)
    print(array([[contingency_array[i][0], contingency_array[i][1]],
                 [contingency_array[i][2], contingency_array[i][3]]]))
