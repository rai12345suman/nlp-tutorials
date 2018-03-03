import os
from text.classifiers import NaiveBayesClassifier

train = [
    ('amor', "spanish"),
    ("perro", "spanish"),
    ("playa", "spanish"),
    ("sal", "spanish"),
    ("oceano", "spanish"),
    ("love", "english"),
    ("dog", "english"),
    ("beach", "english"),
    ("salt", "english"),
    ("ocean", "english")
]
test = [
    ("ropa", "spanish"),
    ("comprar", "spanish"),
    ("camisa", "spanish"),
    ("agua", "spanish"),
    ("telefono", "spanish"),
    ("clothes", "english"),
    ("buy", "english"),
    ("shirt", "english"),
    ("water", "english"),
    ("telephone", "english")
]

def extractor(word):
    '''Extract the last letter of a word as the only feature.'''
    feats = {}
    last_letter = word[-1]
    feats["last_letter({0})".format(last_letter)] = True
    return feats

lang_detector = NaiveBayesClassifier(train, feature_extractor=extractor)
print(lang_detector.accuracy(test))
print(lang_detector.show_informative_features(5))
