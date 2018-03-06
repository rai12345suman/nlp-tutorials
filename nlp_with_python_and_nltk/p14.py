# Save Classifier with Pickle - Natural Language Processing With Python and NLTK p.14
# As you will likely find with any form of data analysis, there is going to be some sort of
# processing bottleneck, that you repeat over and over, often yielding the same object in Python
# memory. 
#
# Examples of this might be loading a massive dataset into memory, some basic pre-processing of a
# static dataset, or, like in our case, the training of a classifier. 
#
# In our case, we spend much time on training our classifier, and soon we may add more. It is a wise choice to go ahead and pickle the trained classifer. This way, we can load in the trained
# classifier in a matter of milliseconds, rather than waiting 3-5+ minutes for the classifier to be trained. 
#
# To do this, we use the standard library's "pickle" module. What pickle does is serialize, or
# de-serialize, python objects. This could be lists, dictionaries, or even things like our trained classifier!
# https://youtu.be/ReakZVh2Xwk

import nltk
import random
from nltk.corpus import movie_reviews
import pickle

# list of tuples
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

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
print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, test_set))*100)
print('15 most informative feature: ')
classifier.show_most_informative_features(15)

print('Reading from Pickle')
save_classifier = open('naive_bayes.pickle','wb')
pickle.dump(classifier,save_classifier)
save_classifier.close()
classifier_f = open('naive_bayes.pickle','rb')
classifier = pickle.load(classifier_f)
classifier_f.close()
print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, test_set))*100)
print('15 most informative feature: ')
classifier.show_most_informative_features(15)
