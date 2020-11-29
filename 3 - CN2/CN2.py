# For this project, the only thing that we will need to import is the "Orange" library.
# However, before importing it, you must first install the library into Python.
# Read the instructions on how to do that (it might be a bit trickier than usual!)
# =============================================================================
import Orange

# =============================================================================
# Load 'wine' dataset
# =============================================================================
wineData = Orange.data.Table('./wine.csv')

# =============================================================================
# Define the learner that will be trained with the data.
# Try two different learners: an '(Ordered) Learner' and an 'UnorderedLearner'.
# =============================================================================
#CN2
# print("Rule ordering: Unordered, Evaluator: Laplace")
# learner = Orange.classification.CN2UnorderedLearner()
# learner.rule_finder.quality_evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator()

# print("Rule ordering: Ordered, Evaluator: Laplace")
# learner = Orange.classification.CN2Learner()
# learner.rule_finder.quality_evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator()

print("Rule ordering: Ordered, Evaluator: Entropy")
learner = Orange.classification.CN2Learner()
learner.rule_finder.quality_evaluator = Orange.classification.rules.EntropyEvaluator()

#CN2SD
# print("Rule ordering: Unordered, Evaluator: Laplace")
# learner = Orange.classification.CN2SDUnorderedLearner()
# learner.rule_finder.quality_evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator()

# print("Rule ordering: Ordered, Evaluator: Laplace")
# learner = Orange.classification.CN2SDLearner()
# learner.rule_finder.quality_evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator()

# print("Rule ordering: Ordered, Evaluator: Entropy")
# learner = Orange.classification.CN2SDLearner()
# learner.rule_finder.qual
# =============================================================================
# At this step we shall configure the parameters of our learner.
# We can set the evaluator/heuristic ('Entropy', 'Laplace' or 'WRAcc'),
# 'beam_width' (in the range of 3-10), 'min_covered_examples' (start from 7-8 and make your way up),
# and 'max_rule_length' (usual values are in the range of 2-5).
# They are located deep inside the 'learner', within the 'rule_finder' class.
# Note: for the evaluator, set it using one of the Evaluator classes in classification.rules
# =============================================================================
# continuous value space is constrained to reduce computation time
learner.rule_finder.search_strategy.constrain_continuous = True
# consider up to 3/6/10 solution streams at one time
learner.rule_finder.search_algorithm.beam_width = 6
# found rules must cover at least 7/10/15 examples
learner.rule_finder.general_validator.min_covered_examples = 15
# found rules may combine at most 2/5 selectors (conditions)
learner.rule_finder.general_validator.max_rule_length = 5

# =============================================================================
# We want to test our model now. The CrossValidation() function will do all the
# work in this case, which includes splitting the whole dataset into train and test subsets,
# then train the model, and produce results.
# So, simply initialize the CrossValidation() object from the 'testing' library
# and call it with input arguments 1) the dataset and 2) the learner.
# Note that the 'learner' argument should be in array form, i.e. '[learner]'.
cv = Orange.evaluation.CrossValidation()
results = cv(wineData, [learner])

# As for the required metrics, you can get them using the 'evaluation.scoring' library.
# The 'average' parameter of each metric is used while measuring scores to perform
# a type of averaging on the data. DON'T WORRY MUCH ABOUT THAT JUST YET (AGAIN). USE EITHER
# 'MICRO' OR 'MACRO' (preferably 'macro', at least for final results).
# =============================================================================
print("Accuracy: {:.5f}".format(Orange.evaluation.scoring.CA(results)[0]))
print("Precision: %.5f" % Orange.evaluation.Precision(results, average='macro'))
print("Recall: %.5f" % Orange.evaluation.Recall(results, average='macro'))
print("F1: %.5f" % Orange.evaluation.F1(results, average='macro'))

# =============================================================================
# Ok, now let's train our learner manually to see how it can classify our data
# using rules.You just want to feed it some data- nothing else.
# =============================================================================
classifier = learner(wineData)

# =============================================================================
# Now we can print the derived rules. To do that, we need to iterate through
# the 'rule_list' of our classifier.
for rule in classifier.rule_list:
    print(rule)

# =============================================================================
