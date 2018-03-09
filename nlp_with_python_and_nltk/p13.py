# Naive Bayes - Natural Language Processing With Python and NLTK p.13
# The algorithm of choice, at least at a basic level, for text analysis is often the Naive Bayes
# classifier. Part of the reason for this is that text data is almost always massive in size. The
# Naive Bayes algorithm is so simple that it can be used at scale very easily with minimal process requirements.
# https://youtu.be/rISOsUaTrO4

import nltk
import random
from nltk.corpus import movie_reviews
import string

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
    all_words = [''.join(c for c in s if c not in string.punctuation) for s in all_words]
    # Remove the empty strings:
    all_words = [s for s in all_words if s]
    return all_words

# Main
def main():

    # list of tuples of movie reviews, positive and negative
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

    # new code
    featuresets = [(find_features(rev, all_words), category) for (rev, category) in documents]

    # split into training and testing
    train_set = featuresets[:1900] 
    test_set = featuresets[1900:]

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, test_set))*100)
    print('15 most informative feature: ')
    classifier.show_most_informative_features(15)

if __name__ == '__main__':
    main()
