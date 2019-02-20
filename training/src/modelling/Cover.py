from pyspark.sql.types import StringType, ArrayType, MapType, IntegerType, FloatType
from pyspark.sql.functions import udf, explode, when, col, row_number, sum as sum_, lit
from pyspark.sql.window import Window as window
from processing.utils import tokenize_document, get_n_grams
import torch
from itertools import chain
from tqdm import tqdm
from math import log10
from operator import itemgetter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import time

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
        self.left_bias = None
        self.right_bias = None
        self.covariate_diagonal_tensor = None
        self.weights = None
        self.log_values = None
        self.embeddings = None
        self.token_to_id = None
        self.id_to_token = None
        print(torch.cuda.is_available())
        self.device_type = torch.device(device_type)

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
            tokenize = udf(lambda document: tokenize_document(document), ArrayType(StringType()))

            self.corpus.show(10)

            o = self.corpus.withColumn("genre", lit('pop'))

            tokenized_dataframe = o.withColumn('tokens', tokenize(column_name).alias('tokens'))

            words_dataframe = tokenized_dataframe.withColumn('word', explode(col('tokens'))) \
                .groupBy('word') \
                .count() \
                .sort('count', ascending=True)

            filtered_words = words_dataframe.where(col('count') > min_occurrence_count)

            windowSpec = window.orderBy("count")

            filtered_words_with_id_dataframe = filtered_words.withColumn('id', row_number().over(windowSpec)) \
                .sort('id', ascending=False)

            filtered_words.show(100)

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

            reduced.show(20)

            x = reduced.select("value", "key", covariate).rdd\
                .map(lambda row: (list(chain([row[covariate]], row['key'])), row['value']))\
                .collect()

            self.coo_dict = {tuple(row[0]): row[1] for row in x}

            self.token_to_id = token_to_id

            self.covariate_dict = covariate_dict

            self.spark_session.stop()

    def build_co_occurrence_matrix(self):
        self.indexes, self.values = zip(*self.coo_dict.items())
        self.indexes = list(self.indexes)

        print("Number of Covariates should be 2  and is {}".format(self.num_of_covariates))

        self.num_of_occurrences = len(self.values)

        self.right_bias = torch.randn(self.num_of_covariates, self.num_of_words, requires_grad=True, device=self.device_type)
        self.left_bias = torch.randn(self.num_of_covariates, self.num_of_words, requires_grad=True, device=self.device_type)
        self.left_word_tensor = torch.randn(self.num_of_words, self.embedding_size, requires_grad=True, device=self.device_type)
        self.right_word_tensor = torch.randn(self.num_of_words, self.embedding_size, requires_grad=True, device=self.device_type)
        self.covariate_diagonal_tensor = torch.diag_embed(torch.randn(self.num_of_covariates, self.embedding_size)).clone().detach().cuda().requires_grad_(True)
        self.weights = {key: self.get_weight(value) for key, value in self.coo_dict.items()}
        self.log_values = {key: log10(value) for key, value in self.coo_dict.items()}
        self.parameters = [self.left_word_tensor, self.right_word_tensor, self.covariate_diagonal_tensor, self.left_bias, self.right_bias]

    def get_weight(self, value):
        return min((value/self.x_max), 1) ** self.alpha

    def get_batch(self):
        indices = torch.randperm(self.num_of_occurrences)
        if self.device_type == 'cuda:0':
            indices = indices.cuda()
        for idx in range(0, self.num_of_occurrences - self.batch_size + 1, self.batch_size):

            sample = indices[idx:idx + self.batch_size].tolist()
            covariate_idx, left_idx, right_idx = [torch.tensor(list(x)).cuda() for x in zip(*itemgetter(*sample)(self.indexes))]
            start = time.time()
            left_words = torch.index_select(self.left_word_tensor, 0, left_idx)
            right_words = torch.index_select(self.right_word_tensor, 0, right_idx)
            covariate = torch.index_select(self.covariate_diagonal_tensor, 0, covariate_idx)

            left_bias = torch.index_select(self.left_bias, 0, left_idx)
            right_bias = torch.index_select(self.right_bias, 0, right_idx)
            end = time.time()
            print(end - start)
            log_vals = torch.tensor([self.log_values[x] for x in list(zip(*[covariate_idx, left_idx, right_idx]))], device=self.device_type)
            weights = torch.tensor([self.weights[x] for x in list(zip(*[covariate_idx, left_idx, right_idx]))], device=self.device_type )
            yield left_words, right_words, covariate, left_bias, right_bias, log_vals, weights

    def train(self):
        optimizer = torch.optim.Adam(self.parameters, weight_decay=self.weight_decay)
        optimizer.zero_grad()
        for epoch in tqdm(range(self.epochs)):
            logging.info("Start epoch %i", epoch)
            num_batches = int(self.num_of_occurrences/self.batch_size)
            avg_loss = 0.0
            n_batch = int(self.num_of_occurrences/self.batch_size)
            for batch in tqdm(self.get_batch(), total=n_batch, mininterval=1):
                optimizer.zero_grad()
                loss = self.get_loss(*batch)
                avg_loss += loss.data.item() / num_batches
                loss.backward()
                optimizer.step()
        self.embeddings = self.left_word_tensor + self.right_word_tensor
        logging.info("Finished training!")

    def get_loss(self, left_words, right_words, covariate, left_bias, right_bias, log_vals, weights):
        left_context = (left_words.unsqueeze(1) * covariate).sum(1)
        right_context = (right_words.unsqueeze(1) * covariate).sum(1)
        sim = left_context.mul(right_context).sum(1).view(-1)
        x = (sim + left_bias + right_bias - log_vals) ** 2
        loss = torch.mul(x, weights)
        return loss.mean()

    def tsne_plot(self):
        "Creates and TSNE model and plots it"
        plt.interactive(True)
        labels = []
        tokens = []

        for word, index in self.token_to_id.items():
            tokens.append(self.embeddings[index].tolist())
            labels.append(word)

        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16))
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
