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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
import string

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors.nearest_centroid import NearestCentroid

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import HashingVectorizer
# from sklearn.feature_selection import SelectFromModel
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

from nltk.classify import ClassifierI
from statistics import mode, StatisticsError
from sklearn import metrics

# new code
class VoteClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        try:
            most_common_vote = mode(votes)
            return most_common_vote
        except StatisticsError:
                print ('No unique mode found, returning 1st vote')
                # TODO if no unique mode, see if classifiers with highest and second highest
                # accuracy agree. if so do that. if not, just use highest accuracy classifier
                return votes[0]

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        try:
            most_common_vote = mode(votes)
            choice_votes = votes.count(mode(votes))
            conf = choice_votes / len(votes)
            return conf
        except StatisticsError:
            print ('No unique mode found, returning 1st vote')
            return .50

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

def remove_stop_words_from_list(all_words):
    stop_words = set(stopwords.words('english'))
    # word_tokens = word_tokenize(example_sent)
 
    filtered_sentence = [w for w in all_words if not w in stop_words]
 
    filtered_sentence = []
 
    for w in all_words:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def train_and_test_classifiers(train_set, test_set):
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("Classic Naive Bayes Classifier accuracy percent:",
          (nltk.classify.accuracy(classifier, test_set)) * 100)
    # classifier.show_most_informative_features(15)
    MNB_classifier = SklearnClassifier(MultinomialNB(alpha=0.01, fit_prior=False))
    MNB_classifier.train(train_set)
    print("Multinomial Naive Bayes Classifier accuracy percent:",
          (nltk.classify.accuracy(MNB_classifier, test_set)) * 100)

    print("Skipping Gaussian Bayes Classifier accuracy percent")
    # GNB_classifier = SklearnClassifier(GaussianNB())
    # GNB_classifier.fit(features_train, target_train)
    # target_pred = clf.predict(features_test)
    # GNB_classifier.train(train_set)
    # print("Gaussian Naive Bayes Classifier accuracy percent:", (nltk.classify.accuracy(GNB_classifier, test_set))*100)

    BNB_classifier = SklearnClassifier(BernoulliNB(alpha=.01))
    BNB_classifier.train(train_set)
    print("Bernoulli Naive Bayes Classifier accuracy percent:",
          (nltk.classify.accuracy(BNB_classifier, test_set)) * 100)
    
    LG_classifier = SklearnClassifier(LogisticRegression(random_state=42))
    LG_classifier.train(train_set)
    print("Logistic Regression Classifier accuracy percent:",
          (nltk.classify.accuracy(LG_classifier, test_set)) * 100)
    
    # Train SGD with hinge penalty
    SGD_classifier1 = SklearnClassifier(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                                                     random_state=42, max_iter=1000, tol=None))
    # SGD_classifier = SklearnClassifier(SGDClassifier(alpha=0.0005, max_iter=1000))
    SGD_classifier1.train(train_set)
    print("Stochastic Gradient Descent Classifier 1 accuracy percent:",
          (nltk.classify.accuracy(SGD_classifier1, test_set)) * 100)

    # Train SGD with Elastic Net penalty
    SGD_classifier2 = SklearnClassifier(SGDClassifier(alpha=1e-3, random_state=42, penalty="elasticnet", max_iter=1000, tol=None))
    SGD_classifier2.train(train_set)
    print("Stochastic Gradient Descent Classifier 2 accuracy percent:",
          (nltk.classify.accuracy(SGD_classifier2, test_set)) * 100)

    # print("Skipping C-Support Vector Classifier")
    # print("Skipping Linear-Support Vector Classifier")
    SVC_classifier = SklearnClassifier(SVC(), sparse=False).train(train_set)
    SVC_classifier.train(train_set)
    print("C-Support Vector Classifier accuracy percent:",
          (nltk.classify.accuracy(SVC_classifier, test_set)) * 100)
    LinearSVC_classifier1 = SklearnClassifier(SVC(kernel='linear', probability=True, tol=1e-3))
    LinearSVC_classifier1.train(train_set)
    print("Linear Support Vector Classifier 1 accuracy percent:",
          (nltk.classify.accuracy(LinearSVC_classifier1, test_set)) * 100)
    LinearSVC_classifier2 = SklearnClassifier(LinearSVC("l1", dual=False, tol=1e-3))
    LinearSVC_classifier2.train(train_set)
    print("Linear Support Vector Classifier 2 accuracy percent:",
          (nltk.classify.accuracy(LinearSVC_classifier2, test_set)) * 100)
    LinearSVC_classifier3 = SklearnClassifier(LinearSVC("l2", dual=False, tol=1e-3))
    LinearSVC_classifier3.train(train_set)
    print("Linear Support Vector Classifier 3 accuracy percent:",
          (nltk.classify.accuracy(LinearSVC_classifier3, test_set)) * 100)

    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(train_set)
    print("Nu-Support Vector Classifier accuracy percent:",
          (nltk.classify.accuracy(NuSVC_classifier, test_set)) * 100)

    # new code
    
    # Train NearestCentroid (aka Rocchio classifier) without threshold
    Nearest_Centroid_classifier = SklearnClassifier(NearestCentroid())
    Nearest_Centroid_classifier.train(train_set)
    print("Nearest Centroid Classifier accuracy percent:",
          (nltk.classify.accuracy(Nearest_Centroid_classifier, test_set)) * 100)

    Ridge_classifier = SklearnClassifier(RidgeClassifier(alpha=0.5, tol=1e-2, solver="sag"))
    Ridge_classifier.train(train_set)
    print("Ridge Classifier accuracy percent:",
          (nltk.classify.accuracy(Ridge_classifier, test_set)) * 100)
    
    Perceptron_classifier = SklearnClassifier(Perceptron(max_iter=1000))
    Perceptron_classifier.train(train_set)
    print("Perceptron Classifier accuracy percent:",
          (nltk.classify.accuracy(Perceptron_classifier, test_set)) * 100)
    
    Passive_Aggressive_classifier = SklearnClassifier(PassiveAggressiveClassifier(max_iter=1000))
    Passive_Aggressive_classifier.train(train_set)
    print("Passive-Aggressive Classifier accuracy percent:",
          (nltk.classify.accuracy(Passive_Aggressive_classifier, test_set)) * 100)
    
    kNN_classifier = SklearnClassifier(KNeighborsClassifier(n_neighbors=10))
    kNN_classifier.train(train_set)
    print("kNN Classifier accuracy percent:",
          (nltk.classify.accuracy(kNN_classifier, test_set)) * 100)
    
    voted_classifier = VoteClassifier(
        classifier, MNB_classifier, BNB_classifier, LG_classifier, SGD_classifier2,
    LinearSVC_classifier2, NuSVC_classifier)
    print("Voted Classifier Classifier accuracy percent:",
          (nltk.classify.accuracy(voted_classifier, test_set)) * 100)
    print("Classification: ", voted_classifier.classify(test_set[0][
        0]), "Confidence: %", voted_classifier.confidence(test_set[0][0]) * 100)
    print("Classification: ", voted_classifier.classify(test_set[2][
        0]), "Confidence: %", voted_classifier.confidence(test_set[2][0]) * 100)
    print("Classification: ", voted_classifier.classify(test_set[3][
        0]), "Confidence: %", voted_classifier.confidence(test_set[3][0]) * 100)
    print("Classification: ", voted_classifier.classify(test_set[4][
        0]), "Confidence: %", voted_classifier.confidence(test_set[4][0]) * 100)

    # print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

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
    all_words = remove_stop_words_from_list(all_words)
    all_words = nltk.FreqDist(all_words)

    featuresets = [(find_features(rev, all_words), category)
                   for (rev, category) in documents]

    # split into train and test
    train_set = featuresets[:1900]
    test_set = featuresets[1900:]

    train_and_test_classifiers(train_set, test_set)

if __name__ == '__main__':
    main()
