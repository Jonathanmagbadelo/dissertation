import logging
import time
import os
import ngrams
import numpy as np
import pandas as pd
import torch

from collections import Counter
from src.lib.processing import pre_process, utils
from src.lib.modelling.embedding.utils import get_prob, transform, flatten_co_values, build_vocab
from tqdm import tqdm

# HYPERPARAMETERS
MAX_WORDS = 200000
MIN_WORD_OCCURRENCE = 25
WINDOW_SIZE = 7
DEVICE_TYPE = 'cuda'
EMBEDDING_DIM = 300
BATCH_SIZE = 512
NUM_EPOCH = 10
X_MAX = 100
ALPHA = 0.75
LR = 0.05
SYMMETRIC_WINDOW = 1
SUBSAMPLE_THRESHOLD = 0.001
CO_VARIATE_WORD_COUNTERS = {0: Counter()}

print(os.getcwd())
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)


def co_variate_to_id(co_variate_list):
    return [0] * len(co_variate_list)


before = time.time()
corpus = pd.read_csv('data/dataset-med.csv')

before_pre = time.time()
corpus['Genre'] = co_variate_to_id(corpus['Genre'].values)
corpus['Lyric'] = pre_process.tokenize(corpus['Lyric'])
corpus['Lyric'] = pre_process.remove_non_words(corpus['Lyric'].values)

documents = corpus['Lyric'].to_list()
co_variate_list = corpus['Genre'].to_list()

word_counts = build_vocab(documents, co_variate_list, CO_VARIATE_WORD_COUNTERS)

CO_VARIATE_WORDS_TOTALS = {key: sum(value.values()) for key, value in CO_VARIATE_WORD_COUNTERS.items()}
WORD_SAMPLE_PROB = {co_variate: {word: get_prob(count, CO_VARIATE_WORDS_TOTALS[co_variate], SUBSAMPLE_THRESHOLD) for word, count in
                                 co_variate_counts.items()} for co_variate, co_variate_counts in
                    CO_VARIATE_WORD_COUNTERS.items()}

vocab = np.array([word for word, count in word_counts.most_common(MAX_WORDS) if count >= MIN_WORD_OCCURRENCE])
token_to_id = {word: i for i, word in enumerate(vocab)}
id_to_token = {i: word for word, i in token_to_id.items()}

idtb = time.time()
corpus['Lyric'] = pre_process.remove_stopwords(corpus['Lyric'].values, WORD_SAMPLE_PROB, co_variate_list)
corpus['Transform'] = transform(corpus['Lyric'].values, co_variate_list, token_to_id)
idta = time.time()
logging.info("Took {} to transform data".format(idta - idtb))

transformed_documents = (np.array(transformed_document, dtype=np.uint16) for transformed_document in
                         corpus['Transform'].to_list())

NUM_WORDS = len(vocab)
after_pre = time.time()
logging.info("Took {} seconds to pre process documents".format(after_pre - before_pre))

before_o = time.time()
co_values = ngrams.get_co_values(list(zip(co_variate_list, transformed_documents)), WINDOW_SIZE, SYMMETRIC_WINDOW)
flattened_co_values = flatten_co_values(co_values)
after_o = time.time()
logging.info("Took {} seconds to build co occurrence tensor".format(after_o - before_o))

indices, values = zip(*flattened_co_values.items())
_, left, right = zip(*indices)

values = np.array(values)
num_occurrences = len(left)

focal_words = torch.tensor(left, device=DEVICE_TYPE)
context_words = torch.tensor(right, device=DEVICE_TYPE)


weights = np.minimum((values / X_MAX) ** ALPHA, 1)
weights = torch.as_tensor(weights, dtype=torch.float, device=DEVICE_TYPE)
log_co_values = torch.as_tensor(np.log(values), dtype=torch.float, device=DEVICE_TYPE)

logging.info("Number of words in vocab is {}".format(NUM_WORDS))

focal_vectors = torch.tensor(np.random.uniform(-.5, .5, [NUM_WORDS, EMBEDDING_DIM]), dtype=torch.float,
                             device=DEVICE_TYPE, requires_grad=True)
context_vectors = torch.tensor(np.random.uniform(-.5, .5, [NUM_WORDS, EMBEDDING_DIM]), dtype=torch.float,
                               device=DEVICE_TYPE, requires_grad=True)

focal_biases = torch.tensor(np.random.uniform(-.5, .5, [NUM_WORDS]), device=DEVICE_TYPE,
                            dtype=torch.float,
                            requires_grad=True)
context_biases = torch.tensor(np.random.uniform(-.5, .5, [NUM_WORDS]), device=DEVICE_TYPE,
                              dtype=torch.float,
                              requires_grad=True)

params = [focal_vectors, context_vectors, focal_biases, context_biases]


def gen_batches():
    i = torch.randperm(num_occurrences, device=DEVICE_TYPE)
    for idx in range(0, num_occurrences - BATCH_SIZE + 1, BATCH_SIZE):
        sample = i[idx:idx + BATCH_SIZE]

        focal_word_sample = focal_words.index_select(index=sample, dim=0)
        context_word_sample = context_words.index_select(index=sample, dim=0)

        focal_vectors_batch = focal_vectors.index_select(index=focal_word_sample, dim=0)
        context_vectors_batch = context_vectors.index_select(index=context_word_sample, dim=0)

        focal_bias_batch = focal_biases[focal_word_sample]
        context_bias_batch = context_biases[context_word_sample]

        weight = weights.index_select(index=sample, dim=0)

        log_co_values_batch = log_co_values.index_select(index=sample, dim=0)

        yield weight, focal_vectors_batch, context_vectors_batch, log_co_values_batch, focal_bias_batch, context_bias_batch


def get_loss(weight, focal_vector, context_vector, log_co_value, focal_bias, context_bias):
    vector_product = (focal_vector * context_vector).sum(1).view(-1)
    mean_square_error = torch.pow((vector_product + focal_bias + context_bias - log_co_value), 2)
    loss = torch.mul(mean_square_error, weight)
    return loss.mean()


def train_model():
    optimizer = torch.optim.Adagrad(params, lr=LR)
    optimizer.zero_grad()
    for epoch in tqdm(range(NUM_EPOCH)):
        logging.info("Start epoch %i", epoch)
        num_batches = torch.tensor((num_occurrences / BATCH_SIZE), dtype=torch.float)
        avg_loss = torch.tensor(0.0, dtype=torch.float)
        n_batch = int(num_occurrences / BATCH_SIZE)
        for batch in tqdm(gen_batches(), total=n_batch, mininterval=1):
            optimizer.zero_grad()
            loss = get_loss(*batch)
            avg_loss += loss / num_batches
            loss.backward()
            optimizer.step()
    logging.info("Average loss for epoch %i: %.5f", epoch + 1, avg_loss.item())
    return focal_vectors + context_vectors


embeddings = train_model()
utils.save_embedding(embeddings, token_to_id, "glove")
after = time.time()
logging.info("Script took {} seconds".format(after - before))
