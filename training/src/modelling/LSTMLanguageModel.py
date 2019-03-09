import itertools
import logging as logger
import random
from collections import Counter

import numpy as np
from keras import Sequential
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.initializers import Constant
from keras.layers import Bidirectional, LSTM, Dense, Embedding, Input, Dropout
from keras.optimizers import Adam


class LSTMLanguageModel:
    def __init__(self, embeddings, embedding_dim, corpus, min_occurrence, sequence_size, max_num_words):
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.corpus = corpus
        self.corpus_words = None
        self.word_frequencies = None
        self.min_occurrence = min_occurrence
        self.word_2_id = None
        self.id_2_word = None
        self.sequence_size = sequence_size
        self.ignored_words = set()
        self.sentences = []
        self.next_words = []
        self.testing_sentences = []
        self.training_sentences = []
        self.testing_next_words = []
        self.training_next_words = []
        self.num_words = 0
        self.max_num_words = max_num_words
        self.num_sentences = 0
        self.model = None

    def set_model(self, units, dropout, activation_function='softmax'):
        model = Sequential()
        model.add(Bidirectional(LSTM(units), input_shape=(self.sequence_size, self.num_words)))
        if dropout > 0:
            model.add(Dropout(rate=dropout))
        model.add(Dense(self.num_words, activation=activation_function))
        embedding_layer = self.get_embedding_layer()
        sequence_input = Input(shape=(self.sequence_size,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        self.model = model

    def train(self, batch_size, epochs, lr=0.0001):
        if self.model is None:
            logger.info("You must set the model first by calling the set_model() method!!")
        else:
            optimizer = Adam(lr=lr)
            self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
                        "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % \
                        (self.num_words, self.sequence_size, self.min_occurrence)

            checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
            print_callback = LambdaCallback(on_batch_end=self.on_epoch_end)
            early_stopping = EarlyStopping(monitor='val_acc', patience=20)
            callbacks_list = [checkpoint, print_callback, early_stopping]

            self.model.fit_generator(self.generator(self.training_sentences, self.training_next_words, batch_size),
                                     steps_per_epoch=int(len(self.training_sentences) / batch_size) + 1,
                                     epochs=epochs,
                                     callbacks=callbacks_list,
                                     validation_data=self.generator(self.testing_sentences, self.testing_next_words,
                                                                    batch_size),
                                     validation_steps=int(len(self.testing_sentences) / batch_size) + 1)

    # This assumes that all the words in both the sentences and next words are in the word_2_id dict
    def generator(self, sentences, next_words, batch_size):
        index = 0
        sentences_size = len(sentences)
        while True:
            x = np.zeros((batch_size, self.sequence_size), dtype=np.int32)
            y = np.zeros(batch_size, dtype=np.int32)
            for i in range(batch_size):
                for t, w in enumerate(sentences[index % sentences_size]):
                    x[i, t] = self.word_2_id[w]
                y[i] = self.word_2_id[next_words[index % sentences_size]]
                index = index + 1
            yield x, y

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def on_epoch_end(self, epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        print('\n----- Generating text after Epoch: %d\n' % epoch)

        # Randomly pick a seed sequence
        seed_index = np.random.randint(self.num_sentences)
        seed = self.sentences[seed_index]

        for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
            sentence = seed
            print('----- Diversity:' + str(diversity) + '\n')
            print('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
            print(' '.join(sentence))

            for i in range(50):
                x_pred = np.zeros((1, self.sequence_size))
                for t, word in enumerate(sentence):
                    x_pred[0, t] = self.word_2_id[word]

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_word = self.id_2_word[next_index]

                sentence = sentence[1:]
                sentence.append(next_word)

                print(" " + next_word)
            print('\n')
        print()

    def get_embedding_layer(self, should_train=False):
        if not self.sentences:
            logger.info("Load data!")
            return
        num_of_words = min(self.max_num_words, len(self.word_2_id)) + 1
        embedding_matrix = np.zeros((num_of_words, self.embedding_dim))

        return Embedding(num_of_words, self.embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                         input_length=self.sequence_size, trainable=should_train)

    """"
        Need to format corpus(which is a nested list of strings) so its a list of lists where each list is of uniform length
        Assuming Corpus is a list of lists of strings
    """""

    def format_corpus(self, training_split):
        if 1 < training_split < 0:
            logger.info("The training split should be a number between 0 and 1!")
            return

        self.corpus = (lyric.lower().replace('\n', ' \n ') for lyric in self.corpus)
        self.corpus_words = list(itertools.chain.from_iterable([lyric.split() for lyric in self.corpus]))
        self.word_frequencies = Counter(self.corpus_words)
        words_2_keep = (word if count >= self.min_occurrence else self.ignored_words.add(word) for word, count in
                        self.word_frequencies.items())
        self.word_2_id = {word: index for (index, word) in enumerate(words_2_keep) if word}
        self.id_2_word = {index: word for (word, index) in self.word_2_id.items()}
        chunks = self.chunks(self.corpus_words, self.sequence_size)
        filtered_chunks = [(chunk[:-1], chunk[-1]) for chunk in chunks if set(chunk).isdisjoint(self.ignored_words)]
        self.num_sentences = len(filtered_chunks)

        # shuffle
        random.shuffle(filtered_chunks)
        self.sentences, self.next_words = zip(*filtered_chunks)

        # split
        training_indicies = int(self.num_sentences * training_split)
        self.training_sentences, self.testing_sentences = self.sentences[:training_indicies], self.sentences[
                                                                                              training_indicies:]
        self.training_next_words, self.testing_next_words = self.next_words[:training_indicies], self.next_words[
                                                                                                 training_indicies:]

        logger.info("Finished corpus formatting! There are {} training examples and {} test examples".format(
            len(self.training_sentences), len(self.testing_sentences)))

    def chunks(self, corpus, chunk_size):
        args = [iter(corpus)] * chunk_size
        return zip(*args)
