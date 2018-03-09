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
import string

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode, StatisticsError

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


def train_and_test_classifiers(train_set, test_set):
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("Classic Naive Bayes Classifier accuracy percent:",
          (nltk.classify.accuracy(classifier, test_set)) * 100)
    MNB_classifier = SklearnClassifier(MultinomialNB(alpha=0.01))
    MNB_classifier.train(train_set)
    print("Multinomial Naive Bayes Classifier accuracy percent:",
          (nltk.classify.accuracy(MNB_classifier, test_set)) * 100)

    print("Skipping Gaussian Bayes Classifier accuracy percent")
    GNB_classifier = SklearnClassifier(GaussianNB())
    #GNB_classifier.fit(features_train, target_train)
    # target_pred = clf.predict(features_test)
    # GNB_classifier.train(train_set)
    # print("Gaussian Naive Bayes Classifier accuracy percent:", (nltk.classify.accuracy(GNB_classifier, test_set))*100)

    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(train_set)
    print("Bernoulli Naive Bayes Classifier accuracy percent:",
          (nltk.classify.accuracy(BNB_classifier, test_set)) * 100)
    
    LG_classifier = SklearnClassifier(LogisticRegression(random_state=0))
    LG_classifier.train(train_set)
    print("Logistic Regression Classifier accuracy percent:",
          (nltk.classify.accuracy(LG_classifier, test_set)) * 100)

    SGD_classifier = SklearnClassifier(SGDClassifier(alpha=0.0005, max_iter=1000))
    SGD_classifier.train(train_set)
    print("Stochastic Gradient Descent Classifier accuracy percent:",
          (nltk.classify.accuracy(SGD_classifier, test_set)) * 100)

    # print("Skipping C-Support Vector Classifier")
    # print("Skipping Linear-Support Vector Classifier")
    SVC_classifier = SklearnClassifier(SVC(), sparse=False).train(train_set)
    # SVC_classifier.train(train_set)
    print("C-Support Vector Classifier accuracy percent:",
          (nltk.classify.accuracy(SVC_classifier, test_set)) * 100)
    LinearSVC_classifier = SklearnClassifier(SVC(kernel='linear', probability=True))
    # LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(train_set)
    print("Linear Support Vector Classifier accuracy percent:",
          (nltk.classify.accuracy(LinearSVC_classifier, test_set)) * 100)

    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(train_set)
    print("Nu-Support Vector Classifier accuracy percent:",
          (nltk.classify.accuracy(NuSVC_classifier, test_set)) * 100)

    # new code
    # voted_classifier = VoteClassifier(
     #   classifier, MNB_classifier, BNB_classifier, LG_classifier, SGD_classifier, SVC_classifier,
     #   LinearSVC_classifier, NuSVC_classifier)
    #print("Voted Classifier Classifier accuracy percent:",
    #      (nltk.classify.accuracy(voted_classifier, test_set)) * 100)
    #print("Classification: ", voted_classifier.classify(test_set[0][
    #      0]), "Confidence: %", voted_classifier.confidence(test_set[0][0]) * 100)
    #print("Classification: ", voted_classifier.classify(test_set[2][
    #      0]), "Confidence: %", voted_classifier.confidence(test_set[2][0]) * 100)
    #print("Classification: ", voted_classifier.classify(test_set[3][
    #      0]), "Confidence: %", voted_classifier.confidence(test_set[3][0]) * 100)
    #print("Classification: ", voted_classifier.classify(test_set[4][
    #      0]), "Confidence: %", voted_classifier.confidence(test_set[4][0]) * 100)

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

    # split into train and test
    train_set = featuresets[:1900]
    test_set = featuresets[1900:]

    train_and_test_classifiers(train_set, test_set)

if __name__ == '__main__':
    main()
