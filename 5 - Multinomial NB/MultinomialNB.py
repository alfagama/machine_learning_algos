# =============================================================================
# HOMEWORK 5 - BAYESIAN LEARNING
# MULTINOMIAL NAIVE BAYES
# =============================================================================
# =============================================================================
# ARAMPATZIS GEORGIOS, AEM: 28
# =============================================================================
# =============================================================================

# =============================================================================
# import stuff!
# =============================================================================
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# =============================================================================
# Create Vectorizer
# =============================================================================
vectorizer = TfidfVectorizer()

# =============================================================================
# Create train and test sets
# =============================================================================
categories = list(fetch_20newsgroups().target_names)
groups_train = fetch_20newsgroups(subset='train',
                                  remove=('headers', 'footers', 'quotes'),
                                  random_state=1)
vectors = vectorizer.fit_transform(groups_train.data)
# print(vectors.shape)

groups_test = fetch_20newsgroups(subset='test',
                                 remove=('headers', 'footers', 'quotes'),
                                 random_state=1)
vectors_test = vectorizer.transform(groups_test.data)
# print(vectors_test.shape)

# =============================================================================
# Run Multinomial Naive Bayes for alpha[0:1] with step 0.1
# =============================================================================
f1_list = []
for i in np.arange(0, 1.1, 0.1):
    clf = MultinomialNB(alpha=i)
    clf.fit(vectors, groups_train.target)
    pred = clf.predict(vectors_test)
    f1 = ["{:.1f}".format(i),
          metrics.f1_score(groups_test.target, pred, average='macro'),
          metrics.accuracy_score(groups_test.target, pred),
          metrics.recall_score(groups_test.target, pred, average="macro"),
          metrics.f1_score(groups_test.target, pred, average='macro')]
    f1_list.append(f1)
print("Multinomial Naive Bayes for alpha[0:1] with step 0.1")
df = pd.DataFrame(f1_list)
df.rename(columns={0: "Alpha", 1: "Accuracy", 2: "Recall", 3: "Precision", 4: "F1 Score"}, inplace=True)
# df.to_csv(r'Multinomial Naive Bayes.csv', index=False, sep=',')
print(df)

# =============================================================================
# Get the best alpha value to run the test once more and print them! :)
# =============================================================================
best = 0
for item in f1_list:
    if item[1] > best:
        best = item[1]
        alfa = item[0]

# =============================================================================
# Run Multivariate NB for best alpha value!
# =============================================================================
clf = MultinomialNB(alpha=float(alfa))
clf.fit(vectors, groups_train.target)
pred = clf.predict(vectors_test)
accuracy = metrics.accuracy_score(groups_test.target, pred)
recall = metrics.recall_score(groups_test.target, pred, average="macro")
precision = metrics.precision_score(groups_test.target, pred, average="macro")
f1_score = metrics.f1_score(groups_test.target, pred, average='macro')

# =============================================================================
# Create confusion matrix
# =============================================================================
cf_matrix = confusion_matrix(groups_test.target, pred)

# =============================================================================
# Check if all test set is accounted for!
# =============================================================================
# res = np.sum(cf_matrix, 0)
# x = 0
# for i in res:
#     x += i
# if x == vectors_test.shape[0]:
#     print("OK!")

# =============================================================================
# Create heatmap from confusion matrix
# =============================================================================
sb.set(font_scale=1)
heat_map = sb.heatmap(cf_matrix,
                      xticklabels=categories,
                      yticklabels=categories,
                      cmap="YlGnBu",
                      fmt='',
                      annot=True,
                      cbar=False)
fig = plt.gcf()
fig.suptitle("Multinomial NB - Confusion matrix (a="
             + str(alfa) + ") "
             + "[Acc=" + str("{:.5f}".format(accuracy))
             + ",Prec=" + str("{:.5f}".format(precision))
             + ",Rec=" + str("{:.5f}".format(recall))
             + ",F1=" + str("{:.5f}".format(f1_score))
             + "]")
fig.set_size_inches(13.5, 8.5)
fig.savefig('Heatmap - Confusion Matrix.png', dpi=100)
plt.show()
