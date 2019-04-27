from better_profanity import profanity
from pronouncing import pronunciations
from keras.initializers import Constant
from keras.layers import Dense, Dropout, Embedding, CuDNNLSTM, GlobalMaxPooling1D, LSTM
from keras import Sequential
from keras.optimizers import Adam
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.models import load_model
import random
import numpy as np
import csv
import tensorflow as tf

WORD_LIST = ["this", "is", "test", "yo", "fuck", "this", "shit", "nigga", "pussy", "bitch", "twerk", "ass", "dick", "hoes", "niggas"]

TEST_WORD_LIST = ["This", "is", "a", "test", "to", "see", "how", "stuff", "changes"]

MIN_NUM_WORDS = 20
UNITS = 256
NUM_WORDS = 3486
ACTIVATION_FUNCTION = "softmax"
OPTIM = Adam(clipvalue=5)
LOSS = "sparse_categorical_crossentropy"
EMBEDDING_DIM = 50
SEQUENCE_SIZE = 21

prediction_model = None

global graph
graph = tf.get_default_graph()


def load_embeddings(co_variate="base"):
    glove_file = "app/songifai/data/cover_{}_embeddings.txt".format(co_variate)
    tmp_file = get_tmpfile("test_word2vec.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    return tmp_file


def load_vocab():
    token_to_id = {}
    id_to_token = {}
    with open("app/songifai/data/cover_base_embeddings.txt", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for index, embedding in enumerate(reader):
            token = embedding[0]
            token_to_id[token] = index
            id_to_token[index] = token
    return token_to_id, id_to_token


def load_prediction_model(folder_name='base'):
    global prediction_model
    model = Sequential()
    model.add(Embedding(NUM_WORDS, EMBEDDING_DIM, input_length=SEQUENCE_SIZE - 1))
    model.add(Dropout(0.2))
    model.add(LSTM(UNITS))
    model.add(Dropout(0.4))
    model.add(Dense(units=NUM_WORDS, activation=ACTIVATION_FUNCTION))
    model.compile(optimizer=OPTIM, loss=LOSS, metrics=['accuracy'])
    model.load_weights("app/songifai/models/{}/language_model.h5".format(folder_name))
    prediction_model = model


token_to_id, id_to_token = load_vocab()
tmp_file = load_embeddings()
load_prediction_model()


def predict_words(clean, rhyme, lyric):
    lyric_list = lyric.lower().split()
    lyric_size = len(lyric_list)
    num_words = lyric_size
    if num_words < MIN_NUM_WORDS:
        return ["You need {} more words for predict function".format(MIN_NUM_WORDS - num_words)]
    zeros = np.zeros((1, SEQUENCE_SIZE - 1))
    latest_words = lyric_list[-MIN_NUM_WORDS:]
    re = [token_to_id[word] for word in latest_words]
    words = predict(re)
    #words = filter_suggested_words(words) if clean else words
    #words = filter_rhyme_words(, words) if rhyme else words
    random.shuffle(words)
    return words[:8]


def suggest_words(clean, word):
    model = KeyedVectors.load_word2vec_format(tmp_file)
    result = [word[0] for word in model.similar_by_word(word.lower(), topn=6)]
    return filter_suggested_words(result) if clean else result


def classify_lyrics(lyrics):
    return "POP" if len(lyrics.split()) > MIN_NUM_WORDS else "Need more words to classify"


def filter_suggested_words(words):
    return [word for word in words if not profanity.contains_profanity(word)]


def filter_rhyme_words(context_word, words):
    return words


def predict(word_id_list):
    x = word_id_list
    zeros = np.zeros((1, SEQUENCE_SIZE - 1))
    for t, word in enumerate(x):
        zeros[0, t] = word
    with graph.as_default():
        prediction_model._make_predict_function()
        x = prediction_model.predict(zeros, verbose=0)[0]
        predictions = np.argpartition(x, -8)[-8:]
        return [id_to_token[idx] for idx in predictions]
