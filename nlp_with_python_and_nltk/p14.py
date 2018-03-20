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
import string
import pickle

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

    featuresets = [(find_features(rev, all_words), category) for (rev, category) in documents]

    # split into training and testing
    train_set = featuresets[:1900]
    test_set = featuresets[1900:]

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, test_set))*100)
    print('15 most informative feature: ')
    classifier.show_most_informative_features(15)

    # new code
    try:
        filename = 'naive_bayes.pickle'
        with open(filename, 'wb') as save_classifier:
            print('Writing with Pickle')
            pickle.dump(classifier, save_classifier)
        with open(filename, 'rb') as classifier_f:
            print('Reading from Pickle')
            classifier = pickle.load(classifier_f)
        print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, test_set))*100)
    except pickle.UnpicklingError as e:
        # normal, somewhat expected
        pass
    except (AttributeError,  EOFError, ImportError, IndexError) as e:
        # secondary errors
        print(traceback.format_exc(e))
        pass
    except Exception as e:
        # everything else, possibly fatal
        print(traceback.format_exc(e))
        return

if __name__ == '__main__':
    main()
