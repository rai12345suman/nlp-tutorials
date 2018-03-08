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
    all_features = find_features(movie_reviews.words('neg/cv000_29416.txt'), all_words)
    
    # TODO print 100 most common words 

    # print 100 features (aka words)
    i = 0
    for key, value in all_features.items():
        print('{0:10} ==> {1:10}'.format(str(key), str(value)))
        i = i + 1
        if i > 100:
            break

    featuresets = [(find_features(rev, all_words), category)
                   for (rev, category) in documents]

if __name__ == '__main__':
    main()
