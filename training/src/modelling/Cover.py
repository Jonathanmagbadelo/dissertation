from pyspark.sql.types import StringType, ArrayType, MapType, IntegerType, FloatType
from pyspark.sql.functions import udf, explode, when, col, row_number, sum as sum_, lit
from pyspark.sql.window import Window as window
from processing.utils import tokenize_document, get_n_grams, pre_process
import torch
from itertools import chain
from tqdm import tqdm
from math import log10
from operator import itemgetter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import time
import numpy as np
from math import log
import nltk
#FORMAT = '%(asctime)-15s %(message)s'
#logging.basicConfig(level=logging.DEBUG, format=FORMAT)


class Cover:
    def __init__(self, spark_session, embedding_size, x_max, alpha, learning_rate, weight_decay, epochs, batch_size, device_type='cpu'):
        self.transformed_data = []
        self.corpus = None
        self.spark_session = spark_session
        self.indexes = None
        self.values = None
        self.coo_dict = None
        self.num_of_words = None
        self.embedding_size = embedding_size
        self.x_max = x_max
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.parameters = []
        self.covariate_dict = {}
        self.num_of_covariates = 0
        self.num_of_occurrences = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.left_word_tensor = None
        self.right_word_tensor = None
        self.left_vectors = None
        self.right_vectors = None
        self.left_bias = None
        self.right_bias = None
        self.covariate_diagonal_tensor = None
        self.covariate_tensor = None
        self.weights = None
        self.log_values = None
        self.embeddings = None
        self.token_to_id = None
        self.id_to_token = None
        print(torch.cuda.is_available())
        self.device_type = torch.device(device_type)
        nltk.download('stopwords')
        nltk.download('words')

    def import_data(self, filename):
        self.corpus = self.spark_session.read. \
            format("csv") \
            .option("header", "True") \
            .option("mode", "DROPMALFORMED") \
            .load(filename)

        print("Corpus has {} documents".format(self.corpus.count()))

    def fit_transform(self, column_name, covariate, min_occurrence_count, window_size):
        if self.corpus is None:
            print("Please load corpus first!")
        else:
            tokenize = udf(lambda document: pre_process(document), ArrayType(StringType()))

            self.corpus.show(10)

            o = self.corpus.withColumn("genre", lit('pop'))

            tokenized_dataframe = o.withColumn('tokens', tokenize(column_name).alias('tokens'))

            tokenized_dataframe.show(20)

            words_dataframe = tokenized_dataframe.withColumn('word', explode(col('tokens'))) \
                .groupBy('word') \
                .count() \
                .sort('count', ascending=True)

            filtered_words = words_dataframe.where(col('count') > min_occurrence_count)

            windowSpec = window.orderBy("count")

            filtered_words_with_id_dataframe = filtered_words.withColumn('id', row_number().over(windowSpec)) \
                .sort('id', ascending=False)
            #bottleneck here
            #filtered_words.show(10)

            token_to_id = filtered_words_with_id_dataframe.rdd.map(lambda row: (row.word, row.id)).collectAsMap()

            id_to_token = {value: key for key, value in token_to_id.items()}

            self.num_of_words = len(id_to_token) + 1

            print("There are {} unique tokens".format(self.num_of_words))

            get_id = udf(lambda x: [token_to_id[word] if word in token_to_id else 0 for word in x], ArrayType(StringType()))
            transformed_dataframe = tokenized_dataframe.withColumn('transform', get_id('tokens').alias('transform'))

            print("Mapped tokens to unique id".format(len(token_to_id)))

            n_grams = udf(lambda indexes: get_n_grams(indexes, window_size),
                          MapType(ArrayType(IntegerType()), FloatType()))

            matrix = transformed_dataframe.withColumn("matrix", n_grams("transform").alias('matrix'))

            reduced = matrix.select(col(covariate), explode(col("matrix")).alias('key', 'value'))\
                .groupBy(col("key"), col(covariate))\
                .agg(sum_("value").alias("value"))

            print("There are {} ij pairs".format(reduced.count()))

            covariate_list = reduced.select(covariate)\
                .distinct().rdd\
                .flatMap(lambda covariate_name: covariate_name)\
                .collect()

            self.num_of_covariates = len(covariate_list)

            covariate_dict = {covariate_name: covariate_id for covariate_id, covariate_name in enumerate(covariate_list)}

            covariate_2_id = udf(lambda covariate: covariate_dict[covariate])

            reduced = reduced.withColumn(covariate, covariate_2_id(covariate).cast(IntegerType()))

            #reduced.show(20)

            x = reduced.select("value", "key", covariate).rdd\
                .map(lambda row: (list(chain([row[covariate]], row['key'])), row['value']))\
                .collect()

            self.coo_dict = {tuple(row[0]): row[1] for row in x}

            self.token_to_id = token_to_id

            self.covariate_dict = covariate_dict

            self.spark_session.stop()

    def build_coo_matrix(self):
        self.indexes, self.values = zip(*self.coo_dict.items())
        self.num_of_occurrences = len(self.values)
        covariate, left, right = zip(*list(self.indexes))
        #print("Number of Covariates should be 2  and is {}".format(self.num_of_covariates))
        self.log_values = torch.tensor([log(val) for val in self.values], device=self.device_type)
        self.left_word_tensor = torch.tensor(left, device=self.device_type)
        self.right_word_tensor = torch.tensor(right, device=self.device_type)
        self.covariate_tensor = torch.tensor(covariate, device=self.device_type)
        self.weights = torch.tensor([self.get_weight(val) for val in self.values], device=self.device_type)

        self.left_vectors = torch.randn(self.num_of_words, self.embedding_size, requires_grad=True, device=self.device_type)
        self.right_vectors = torch.randn(self.num_of_words, self.embedding_size, requires_grad=True, device=self.device_type)
        self.right_bias = torch.randn(self.num_of_covariates, self.num_of_words, requires_grad=True, device=self.device_type)
        self.left_bias = torch.randn(self.num_of_covariates, self.num_of_words, requires_grad=True, device=self.device_type)
        #self.covariate_diagonal_tensor = torch.diag_embed(torch.randn(self.num_of_covariates, self.embedding_size)).clone().detach().cuda().requires_grad_(True)
        self.covariate_diagonal_tensor = torch.randn(self.num_of_covariates, self.embedding_size, requires_grad=True, device=self.device_type)
        self.parameters = [self.left_vectors, self.right_vectors, self.left_bias, self.right_bias]
        print("Ypp")

    def get_weight(self, value):
        return min((value/self.x_max), 1) ** self.alpha

    def get_batch(self):
        indices = torch.randperm(self.num_of_occurrences)
        if self.device_type == 'cuda:0':
            indices = indices.cuda()
        for idx in range(0, self.num_of_occurrences - self.batch_size + 1, self.batch_size):

            sample = indices[idx:idx + self.batch_size]
            covariates, left_words, right_words = self.covariate_tensor[sample], self.left_word_tensor[sample], self.right_word_tensor[sample]

            start = time.time()
            left_vecs = self.left_vectors[left_words]
            right_vecs = self.right_vectors[right_words]
            #covariate = self.covariate_diagonal_tensor[covariates]
            left_bias = self.left_bias[covariates, left_words]
            right_bias = self.right_bias[covariates, right_words]
            end = time.time()
            print(end - start)

            weights = self.weights[sample]
            log_vals = self.log_values[sample]

            yield left_vecs, right_vecs, left_bias, right_bias, log_vals, weights

    def train(self):
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer.zero_grad()
        for epoch in tqdm(range(self.epochs)):
            logging.info("Start epoch %i", epoch)
            num_batches = int(self.num_of_occurrences/self.batch_size)
            avg_loss = 0.0
            n_batch = int(self.num_of_occurrences/self.batch_size)
            for batch in tqdm(self.get_batch(), total=n_batch, mininterval=1):
                optimizer.zero_grad()
                loss = self.get_loss2(*batch)
                avg_loss += loss.data.item() / num_batches
                loss.backward()
                optimizer.step()
        self.embeddings = self.left_vectors + self.right_vectors
        logging.info("Finished training!")

    def get_loss(self, left_words, right_words, covariate, left_bias, right_bias, log_vals, weights):
        left_context = (left_words.unsqueeze(1) * covariate).sum(1)
        right_context = (right_words.unsqueeze(1) * covariate).sum(1)
        sim = left_context.mul(right_context).sum(1).view(-1)
        x = (sim + left_bias + right_bias - log_vals) ** 2
        loss = torch.mul(x, weights)
        return loss.mean()

    def get_loss2(self, l_vecs, r_vecs, l_bias, r_bias, log_covals, weight):
        sim = (l_vecs * r_vecs).sum(1).view(-1)
        x = (sim + l_bias + r_bias - log_covals) ** 2
        loss = torch.mul(x, weight)
        return loss.mean()

    def tsne_plot(self, word_count=1000):
        "Creates and TSNE model and plots it"
        plt.interactive(True)
        labels = []
        tokens = []

        for word, index in self.token_to_id.items():
            tokens.append(self.embeddings[index].tolist())
            labels.append(word)

        tokens, labels = tokens[:word_count], labels[:word_count]

        tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(100, 100))
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig('embd.png')
        plt.show(block=True)
        print('done')
