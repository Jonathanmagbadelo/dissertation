import numpy as np
from collections import Counter
from flatten_dict import flatten


def transform(lyric_list, co_variate_list, token_to_id):
    return [[token_to_id[word] if word in token_to_id else 0 for word in lyric] for lyric, co_variate in
            zip(lyric_list, co_variate_list)]


def get_prob(co_variate_word_count, co_variate_word_total, threshold):
    word_prob = co_variate_word_count / co_variate_word_total
    return (np.sqrt(word_prob / threshold) + 1) * (
            threshold / word_prob)


def build_vocab(lyric_array, co_variate_array=None, co_variate_word_counters=None):
    for co_variate, lyric in zip(co_variate_array, lyric_array):
        co_variate_word_counters[co_variate].update(lyric)
    return sum([value for key, value in co_variate_word_counters.items()], Counter())


def flatten_co_values(nested_co_values):
    return flatten(nested_co_values)
