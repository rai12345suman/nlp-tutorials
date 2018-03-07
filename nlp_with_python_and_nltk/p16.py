# Combining Algos with a Vote - Natural Language Processing With Python and NLTK p.16
# Now that we have many classifiers, what if we created a new classifier, which combined the votes
# of all of the classifiers, and then classified the text whatever the majority vote was? 
#
# Turns out, doing this is super easy. NLTK has considered this in advance, allowing us to inherit
# from their ClassifierI class from nltk.classify, which will give us the attributes of a
# classifier, yet allow us to write our own custom classifier code. 
# https://youtu.be/vlTQLb_a564

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class Vote_Classifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers


# list of tuples
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)


def find_features(document):
    words = set(document)
    # use top 3000 words
    word_features = list(all_words.keys())[:3000]
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

train_set, test_set = featuresets[1900:], featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print("Classic Naive Bayes Classifier accuracy percent:",
      (nltk.classify.accuracy(classifier, test_set)) * 100)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("Multinomial Naive Bayes Classifier accuracy percent:",
      (nltk.classify.accuracy(MNB_classifier, test_set)) * 100)

print("Skipping Gaussian Bayes Classifier accuracy percent")
# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(train_set)
# print("Gaussian Naive Bayes Classifier accuracy percent:", (nltk.classify.accuracy(GNB_classifier, test_set))*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(train_set)
print("Bernoulli Naive Bayes Classifier accuracy percent:",
      (nltk.classify.accuracy(BNB_classifier, test_set)) * 100)

LG_classifier = SklearnClassifier(LogisticRegression())
LG_classifier.train(train_set)
print("Logistic Regression Classifier accuracy percent:",
      (nltk.classify.accuracy(LG_classifier, test_set)) * 100)

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(train_set)
print("Stochastic Gradient Descent Classifier accuracy percent:",
      (nltk.classify.accuracy(SGD_classifier, test_set)) * 100)

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(train_set)
# print("C-Support Vector Classifier accuracy percent:",
#      (nltk.classify.accuracy(SVC_classifier, test_set)) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("Linear Support Vector Classifier accuracy percent:",
      (nltk.classify.accuracy(LinearSVC_classifier, test_set)) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(train_set)
print("Nu-Support Vector Classifier accuracy percent:",
      (nltk.classify.accuracy(NuSVC_classifier, test_set)) * 100)
