# Naive Bayes - Natural Language Processing With Python and NLTK p.13
# The algorithm of choice, at least at a basic level, for text analysis is often the Naive Bayes
# classifier. Part of the reason for this is that text data is almost always massive in size. The
# Naive Bayes algorithm is so simple that it can be used at scale very easily with minimal process
# requirements.
# https://youtu.be/rISOsUaTrO4

import nltk
import random
from nltk.corpus import movie_reviews

# list of tuples
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

# use top 3000 words
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_feaures:
        features[w] = {w in words}
    return features
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
