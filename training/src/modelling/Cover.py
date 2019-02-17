from pyspark.sql.types import StringType, ArrayType, MapType, IntegerType, FloatType
from pyspark.sql.functions import udf, explode, when, col, row_number, sum as sum_
from pyspark.sql.window import Window as window
from processing.utils import tokenize_document, get_n_grams
import torch
from itertools import chain
from tqdm import tqdm
from math import log10
from operator import itemgetter
import logging

#FORMAT = '%(asctime)-15s %(message)s'
#logging.basicConfig(level=logging.DEBUG, format=FORMAT)


class Cover:
    def __init__(self, spark_session, embedding_size, x_max, alpha, learning_rate, weight_decay, epochs, batch_size):
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

            tokenized_dataframe = self.corpus.withColumn('tokens', tokenize(column_name).alias('tokens'))

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

            self.covariate_dict = covariate_dict

            self.spark_session.stop()

    def build_co_occurrence_matrix(self):
        self.indexes, self.values = zip(*self.coo_dict.items())
        self.indexes = list(self.indexes)

        print("Number of Covariates should be 2  and is {}".format(self.num_of_covariates))

        self.num_of_occurrences = len(self.values)

        index_tensor = torch.LongTensor(self.indexes)
        value_tensor = torch.FloatTensor(self.values)
        self.right_bias = torch.randn(self.num_of_covariates, self.num_of_words, requires_grad=True)
        self.left_bias = torch.randn(self.num_of_covariates, self.num_of_words, requires_grad=True)
        self.left_word_tensor = torch.randn(self.num_of_words, self.embedding_size, requires_grad=True)
        self.right_word_tensor = torch.randn(self.num_of_words, self.embedding_size, requires_grad=True)
        self.covariate_diagonal_tensor = torch.tensor(torch.diag_embed(torch.randn(self.num_of_covariates, self.embedding_size, requires_grad=True)), requires_grad=True)
        self.weights = {key: self.get_weight(value) for key, value in self.coo_dict.items()}
        self.log_values = {key: log10(value) for key, value in self.coo_dict.items()}
        self.parameters = [self.left_word_tensor, self.right_word_tensor, self.covariate_diagonal_tensor, self.left_bias, self.right_bias]


    def get_weight(self, value):
        return min((value/self.x_max), 1) ** self.alpha

    def get_batch(self):
        indices = torch.randperm(self.num_of_occurrences)
        for idx in range(0, self.num_of_occurrences - self.batch_size + 1, self.batch_size):
            sample = indices[idx:idx + self.batch_size].tolist()
            covariate_idx, left_idx, right_idx = [list(x) for x in zip(*itemgetter(*sample)(self.indexes))]
            left_words = self.left_word_tensor[left_idx]
            right_words = self.right_word_tensor[right_idx]
            covariate = self.covariate_diagonal_tensor[covariate_idx]
            left_bias = self.left_bias[covariate_idx, left_idx]
            right_bias = self.right_bias[covariate_idx, right_idx]
            log_vals = torch.tensor([self.log_values[x] for x in list(zip(*[covariate_idx, left_idx, right_idx]))])
            weights = torch.tensor([self.weights[x] for x in list(zip(*[covariate_idx, left_idx, right_idx]))])
            yield left_words, right_words, covariate, left_bias, right_bias, log_vals, weights

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
