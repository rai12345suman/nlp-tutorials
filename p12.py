# Words as Features for Learning - Natural Language Processing With Python and NLTK p.12
# For our text classification, we have to find some way to "describe" bits of data, which are
# labeled as either positive or negative for machine learning training purposes. 
#
# These descriptions are called "features" in machine learning. For our project, we're just going to
# simply classify each word within a positive or negative review as a "feature" of that review. 
#
# Then, as we go on, we can train a classifier by showing it all of the features of positive and
# negative reviews (all the words), and let it try to figure out the more meaningful differences
# between a positive review and a negative review, by simply looking for common negative review
# words and common positive review words.
# https://youtu.be/-vVskDsHcVc

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
