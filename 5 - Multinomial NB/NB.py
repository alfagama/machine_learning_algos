# =============================================================================
# HOMEWORK 5 - BAYESIAN LEARNING
# MULTINOMIAL NAIVE BAYES
# =============================================================================
# =============================================================================
# ARAMPATZIS GEORGIOS, AEM: 28
# =============================================================================
# =============================================================================

# =============================================================================
#   import stuff
from sklearn import linear_model, datasets, metrics, model_selection

# =============================================================================

dataset = datasets.fetch_20newsgroups()
print(dataset)