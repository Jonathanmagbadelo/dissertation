from better_profanity import profanity
from pronouncing import  pronunciations
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import random

WORD_LIST = ["this", "is", "test", "yo", "fuck", "this", "shit", "nigga", "pussy", "bitch", "twerk", "ass", "dick", "hoes", "niggas"]

TEST_WORD_LIST = ["This", "is", "a", "test", "to", "see", "how", "stuff", "changes"]

glove_file = "embeddings/qvec-master/base_embedding.txt"
tmp_file = get_tmpfile("test_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)


# possibly use bigrams/trigrams
def predict_words(clean, rhyme, context_word):
    words = TEST_WORD_LIST
    words = filter_suggested_words(words) if clean else words
    words = filter_rhyme_words(context_word, words) if rhyme else words
    random.shuffle(words)
    return words


def suggest_words(word, clean):
    model = KeyedVectors.load_word2vec_format(tmp_file)
    result = model.similar_by_word(word)
    words = result['data']
    return filter_suggested_words(words) if clean else words


def filter_suggested_words(words):
    return [word for word in words if not profanity.contains_profanity(word)]


def filter_rhyme_words(context_word, words):
    return words
