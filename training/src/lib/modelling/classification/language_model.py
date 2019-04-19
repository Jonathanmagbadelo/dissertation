import itertools
import logging as logger
import os
import random

import numpy as np
from keras import Sequential
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.initializers import Constant
from keras.layers import Bidirectional, Dense, Dropout, Activation, Embedding, CuDNNGRU, regularizers
from keras.optimizers import Adam, RMSprop
from src.lib.processing import utils
from src.lib.processing.pre_process import chunks

os.chdir('../../../../')

FORMAT = '%(asctime)-15s %(message)s'
logger.basicConfig(level=logger.INFO, format=FORMAT)

# Hyperparameters
SEQUENCE_SIZE = 7
TRAINING_SPLIT = 0.2
DROPOUT = 0.2
ACTIVATION_FUNCTION = "softmax"
OPTIM = Adam()
LOSS = "sparse_categorical_crossentropy"
UNITS = 128
BATCH_SIZE = 128
EPOCHS = 10
EMBEDDING_DIM = 300
REGULARIZER = regularizers.l2(0.0001)

token_to_id, id_to_token, embeddings = utils.load_embeddings()
logger.info("Loaded {} embeddings which have {} dimensions.".format(len(embeddings), len(embeddings[0])))

co_variates, corpus = utils.load_corpus()
logger.info("Loaded corpus! There are {} documents.".format(len(corpus)))


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # Randomly pick a seed sequence
    seed_index = np.random.randint(NUM_SENTENCES)
    seed = sentences[seed_index]

    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sentence = list(seed)
        sentence = [id_to_token[index] for index in sentence]
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))

        for i in range(50):
            x_pred = np.zeros((1, SEQUENCE_SIZE - 1))
            for t, word in enumerate(sentence):
                x_pred[0, t] = token_to_id[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = id_to_token[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(" " + next_word)
        examples_file.write('\n')
    examples_file.write('=' * 80 + '\n')

    examples_file.flush()


def transform_chunk(chunk):
    return tuple(token_to_id[token] for token in chunk)


def get_embedding_layer(should_train=True):
    if not sentences:
        logger.info("Load data!")
        return
    embedding_matrix = np.array(embeddings, dtype=np.float)

    return Embedding(NUM_WORDS, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix),
                     input_length=SEQUENCE_SIZE - 1, trainable=should_train)


vocab = {word for word in token_to_id}
flattened_corpus = itertools.chain.from_iterable(corpus)
corpus_chunks = chunks(flattened_corpus, SEQUENCE_SIZE)
filtered_chunks = [transform_chunk(chunk) for chunk in corpus_chunks if set(chunk).issubset(vocab)]
split_chunks = [(np.array(chunk[:-1]), np.array(chunk[-1])) for chunk in filtered_chunks]

NUM_SENTENCES = len(split_chunks)
NUM_WORDS = len(vocab)

# shuffle
random.shuffle(split_chunks)
sentences, next_words = zip(*split_chunks)

logger.info("Finished corpus formatting! There are {} examples".format(
    len(sentences)))

model = Sequential()
model.add(get_embedding_layer())
model.add(Bidirectional(CuDNNGRU(UNITS)))
model.add(Dropout(DROPOUT))
model.add(Dense(units=NUM_WORDS))
model.add(Activation(ACTIVATION_FUNCTION))
model.compile(optimizer=OPTIM, loss=LOSS, metrics=['accuracy'])

file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
            "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % \
            (NUM_WORDS, SEQUENCE_SIZE, 25)

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
print_callback = LambdaCallback(on_batch_end=on_epoch_end)
early_stopping = EarlyStopping(monitor='val_acc', patience=20)
callbacks_list = [print_callback, early_stopping]

examples_file = open("examples.txt", "w")
model.fit(np.array(sentences), np.array(next_words), BATCH_SIZE, EPOCHS, callbacks=callbacks_list,
          validation_split=TRAINING_SPLIT)
logger.info("Finished creating language model")