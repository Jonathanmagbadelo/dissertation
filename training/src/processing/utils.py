import re
import string
from string import punctuation

from nltk.corpus import stopwords, words


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
                n_grams[(int(left_index), int(right_index))] = n_grams.get((int(left_index), int(right_index)), 0) + (
                        1. / (distance + 1))
    return n_grams


def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens


def pre_process(lyrics):
    lyrics = tokenize(lyrics)
    lyrics = normalize(lyrics)
    return lyrics


def normalize(lyrics):
    lyrics = __lowercase__(lyrics)
    lyrics = __remove_stopwords__(lyrics)
    lyrics = __remove_punctuation__(lyrics)
    lyrics = __remove_non_english_words__(lyrics)
    return lyrics


def tokenize(lyrics):
    return lyrics.split() if isinstance(lyrics, str) else []


def __remove_stopwords__(lyrics):
    stop_words = set(stopwords.words('english'))
    return [word for word in lyrics if word not in stop_words]


def __lowercase__(lyrics):
    return [word.lower() for word in lyrics]


def __remove_punctuation__(lyrics):
    return [word.rstrip(string.punctuation) for word in lyrics]


def __remove_non_english_words__(lyrics):
    dictionary = set(words.words())
    return [word for word in lyrics if word in dictionary]
