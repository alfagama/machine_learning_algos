import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import datasets

# -------- Data

iris = datasets.load_iris()
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data_set = pd.read_csv("Data/iris.csv",
                       sep=';',
                       names=names,
                       header=None,  # no header, alternative header = header_col
                       index_col=None,  # no index, alternative header = index_row
                       skiprows=0  # how many rows to skip / not include in read_csv
                       )
# shape
print(data_set.shape)
# head
print(data_set.head())
# descriptions
print(data_set.describe())
# take one value from describe to use
sepal_width_mean = float(pd.DataFrame(data_set.describe())[['sepal-width']].loc['mean'])
print("Sepal-width mean is: ", sepal_width_mean)
# class distribution
print(data_set.groupby('class').size())

# -------- Plots

# box and whisker plots
# pyplot.figure()
data_set.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
# pyplot.savefig('box and whisker plots.png')
pyplot.show()
# histograms
# pyplot.figure()
data_set.hist()
# pyplot.savefig('histograms.png')
pyplot.show()
# scatter plot matrix
# pyplot.figure()
scatter_matrix(data_set)
# pyplot.savefig('scatter plot matrix.png')
pyplot.show()

# -------- Algorithm Evaluation

# Split-out validation data_set
array = data_set.values
# X = array[:, 0:4]
# y = array[:, 4]
X = array[:, -len(array):-1]
y = array[:, -1]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.30, random_state=1)

# Spot Check Algorithms
models = [('LR', LogisticRegression(solver='liblinear',
                                    multi_class='ovr')),
          ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('NB', GaussianNB()),
          ('SVM', SVC(gamma='auto'))]
# evaluate each model in turn
results = []
names = []
for name, model in models:
    k_fold = StratifiedKFold(n_splits=10,
                             random_state=1,
                             shuffle=True)
    cv_results = cross_val_score(
        model, X_train, Y_train,
        cv=k_fold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
# pyplot.figure()
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
# pyplot.savefig('Algorithm Comparison.png')
pyplot.show()

# -------- Make Predictions

# Using the most successful model (SVM)

# Make predictions on validation data_set
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
