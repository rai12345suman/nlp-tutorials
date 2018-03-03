# Text Classification - Natural Language Processing With Python and NLTK p.11
# Now that we understand some of the basics of of natural language processing with the Python NLTK
# module, we're ready to try out text classification. This is where we attempt to identify a body of
# text with some sort of label. 
#
# To start, we're going to use some sort of binary label. Examples of this could be identifying text
# as spam or not, or, like what we'll be doing, positive sentiment or negative sentiment. 
# https://youtu.be/zi16nl82AMA

import nltk
import random
from nltk.corpus import movie_reviews

# list of tuples
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# print all words in the document
# print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print("15 most common words: ")
# 15 most common words
print(all_words.most_common(15))

print("How many times does stupid appear")
# how many times does 'stupid' appear
print(all_words["stupid"])

