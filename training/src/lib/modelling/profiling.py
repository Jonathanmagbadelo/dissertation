from collections import defaultdict, Counter
import ngrams
from flatten_dict import flatten

corpus = ["i put my heart and soul into this game im feeling drained tired of coming up short i need my name up in lights".split() for i in range(50000)]
co_vals = ([0] * 20000) + ([1] * 20000) + ([2] * 10000)
data = list(zip(co_vals, corpus))
WINDOW_SIZE = 5


def method_one(corpus):
    nest = lambda: defaultdict(lambda: defaultdict(float))
    counts = defaultdict(nest)
    for document_co_variate, document in corpus:
        doc_size = len(document)
        for i in range(doc_size):
            for j in range(1, WINDOW_SIZE):
                ind = document[i]
                if i - j > 0:
                    lind = document[i - j]
                    counts[document_co_variate][ind][lind] += 1.0 / j
                if i + j < doc_size:
                    rind = document[i + j]
                    counts[document_co_variate][ind][rind] += 1.0 / j
    return flatten(counts)


def method_four(corpus):
    counts = defaultdict(float)
    for document_co_variate, document in corpus:
        doc_size = len(document)
        for i in range(doc_size):
            for j in range(1, WINDOW_SIZE):
                ind = document[i]
                if i - j > 0:
                    lind = document[i - j]
                    counts[document_co_variate, ind, lind] += 1.0 / j
                if i + j < doc_size:
                    rind = document[i + j]
                    counts[document_co_variate, ind, rind] += 1.0 / j
    return counts

def method_two(data):
    counts = ngrams.get_co_values_d(data, WINDOW_SIZE)
    return flatten(counts)


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


def method_three(data):
    cooccurrence_counts = defaultdict(float)
    for co, region in data:
        for l_context, word, r_context in _context_windows(region, WINDOW_SIZE, WINDOW_SIZE):
            for i, context_word in enumerate(l_context[::-1]):
                cooccurrence_counts[(co, word, context_word)] += 1 / (i + 1)
            for i, context_word in enumerate(r_context):
                cooccurrence_counts[(co, word, context_word)] += 1 / (i + 1)
    return cooccurrence_counts


x = method_one(data)
y = method_two(data)
z = method_three(data)
a = method_four(data)

print("Done")