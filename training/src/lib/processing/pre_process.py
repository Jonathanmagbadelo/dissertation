import string
import enchant
import numpy as np
dictionary = enchant.Dict('en_US')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def tokenize(document):
    pattern_one = r'\[.*?\]'
    pattern_two = r'\(.*?\)'
    document = document.str.lower()
    document = document.str.replace(pattern_one, '')
    document = document.str.replace(pattern_two, '')
    document = document.str.translate(str.maketrans('', '', string.punctuation))
    document = document.str.strip()
    document = document.str.split()
    return document


def remove_non_words(documents):
    return [[word for word in lyric if word and word.isalpha() and len(word) > 2 and dictionary.check(word)] for lyric in documents]


def remove_stopwords(documents, probs, co_variate_list):
    return [[word for word in lyric if not under_sample(word, probs[co_variate][word])] for lyric, co_variate in zip(documents, co_variate_list)]


def under_sample(word, prob):
    if word in stop_words and prob > np.random.rand():
        return True
    else:
        return len(word) <= 3


def chunks(flat_corpus, chunk_size):
    args = [iter(flat_corpus)] * chunk_size
    return zip(*args)
