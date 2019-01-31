from string import punctuation
import re


def tokenize_document(document):
    if document:
        document = re.sub(' +', ' ', document)
        return document.lower().translate(str.maketrans('', '', punctuation)).split(' ')
    return []


def get_n_grams(indexes, window_size):
    n_grams = {}
    for i, left_index in enumerate(indexes):
        window = indexes[i + 1:i + window_size + 1]
        for distance, right_index in enumerate(window):
            if int(left_index) > 0 and int(right_index) > 0:
                n_grams[(left_index, right_index)] = n_grams.get((left_index, right_index), 0) + (1. / (distance + 1))
    return n_grams
