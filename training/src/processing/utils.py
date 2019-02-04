from string import punctuation
import re


def tokenize_document(document):
    if document:
        document = re.sub(' +', ' ', document)
        document = document.strip()
        return document.lower().translate(str.maketrans('', '', punctuation)).split(' ')
    return []


def get_n_grams(indexes, window_size):
    n_grams = {}
    for i, left_index in enumerate(indexes):
        window = indexes[i + 1:i + window_size + 1]
        for distance, right_index in enumerate(window):
            if int(left_index) > 0 and int(right_index) > 0:
                n_grams[(int(left_index), int(right_index))] = n_grams.get((int(left_index), int(right_index)), 0) + (1. / (distance + 1))
    return n_grams
