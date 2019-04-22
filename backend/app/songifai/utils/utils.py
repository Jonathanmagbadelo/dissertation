from better_profanity import profanity
from pronouncing import  pronunciations
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import random

WORD_LIST = ["this", "is", "test", "yo", "fuck", "this", "shit", "nigga", "pussy", "bitch", "twerk", "ass", "dick", "hoes", "niggas"]

TEST_WORD_LIST = ["This", "is", "a", "test", "to", "see", "how", "stuff", "changes"]


def load_embeddings(co_variate="base"):
    glove_file = "app/songifai/data/cover_{}_embeddings.txt".format(co_variate)
    tmp_file = get_tmpfile("test_word2vec.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    return tmp_file


def load_prediction_model():
    return None


tmp_file = load_embeddings()


def predict_words(clean, rhyme, context_word):
    words = TEST_WORD_LIST
    words = filter_suggested_words(words) if clean else words
    words = filter_rhyme_words(context_word, words) if rhyme else words
    random.shuffle(words)
    return words[:8]


def suggest_words(word):
    model = KeyedVectors.load_word2vec_format(tmp_file)
    result = [word[0] for word in model.similar_by_word(word.lower(), topn=6)]
    random.shuffle(result)
    return result
    #eturn filter_suggested_words(result) if clean else words


def classify_lyrics(lyrics):
    return "POP"


def filter_suggested_words(words):
    return [word for word in words if not profanity.contains_profanity(word)]


def filter_rhyme_words(context_word, words):
    return words
