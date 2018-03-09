# Scikit-Learn incorporation - Natural Language Processing With Python and NLTK p.15
# Despite coming packed with some classifiers, NLTK is mainly a toolkit focused on natural language
# processing, and not machine learning specifically.
#
# A module that is focused on machine learning is scikit-learn, which is packed with a large array
# of machine learning algorithms which are optimized in C.
#
# Luckily NLTK has recognized this and comes packaged with a special classifier that wraps around
# scikit learn. In NLTK, this is: nltk.classify.scikitlearn, specifically the class: SklearnClassifier is what we're interested in.
# This allows us to port over any of the scikit-learn classifiers that are compatible, which is most
# https://youtu.be/nla4C-VYNEU

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import string

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


def find_features(document, all_words):
    words = set(document)

    # use top 3000 words
    word_features = list(all_words.keys())[:3000]
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


def remove_punctuation_from_string(s):
    translator = str.maketrans('', '', string.punctuation)
    return s.translate(translator)


def remove_punctuation_from_list(all_words):
    all_words = [''.join(c for c in s if c not in string.punctuation)
                 for s in all_words]
    # Remove the empty strings:
    all_words = [s for s in all_words if s]
    return all_words


def train_and_test_classifiers(train_set, test_set):
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

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(train_set)
    print("C-Support Vector Classifier accuracy percent:",
          (nltk.classify.accuracy(SVC_classifier, test_set)) * 100)

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(train_set)
    print("Linear Support Vector Classifier accuracy percent:",
          (nltk.classify.accuracy(LinearSVC_classifier, test_set)) * 100)

    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(train_set)
    print("Nu-Support Vector Classifier accuracy percent:",
          (nltk.classify.accuracy(NuSVC_classifier, test_set)) * 100)

# Main
def main():

    # list of tuples
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

    # Everyday I'm shufflin'
    random.shuffle(documents)

    all_words = []
    
    # add all words in the movie reviews to list
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = remove_punctuation_from_list(all_words)
    all_words = nltk.FreqDist(all_words)

    featuresets = [(find_features(rev, all_words), category)
                   for (rev, category) in documents]

    # split into training and testing
    train_set = featuresets[:1900]
    test_set = featuresets[1900:]

    # new code
    train_and_test_classifiers(train_set, test_set)

if __name__ == '__main__':
    main()
