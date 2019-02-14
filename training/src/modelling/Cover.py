from pyspark.sql.types import StringType, ArrayType, MapType, IntegerType, FloatType
from pyspark.sql.functions import udf, explode, when, col, row_number, sum as sum_
from pyspark.sql.window import Window as window
from processing.utils import tokenize_document, get_n_grams
import torch
from itertools import chain
from tqdm import tqdm
import logging


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

            windowSpec = window.orderBy("count")

            words_with_id_dataframe = words_dataframe.withColumn('id', row_number().over(windowSpec) - 1) \
                .sort('id', ascending=False)

            words_with_id_dataframe.show(20)

            filtered_words_with_id_dataframe = words_with_id_dataframe.withColumn('id', when(
                words_with_id_dataframe['count'] <= min_occurrence_count, 0).otherwise(words_with_id_dataframe.id))

            token_to_id = filtered_words_with_id_dataframe.rdd.map(lambda row: (row.word, row.id)).collectAsMap()

            self.num_of_words = len(token_to_id)

            print("There are {} unique tokens".format(self.num_of_words))

            get_id = udf(lambda x: [token_to_id[word] for word in x], ArrayType(StringType()))
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

        print("Number of Covariates should be 2  and is {}".format(self.num_of_covariates))

        self.num_of_occurrences = len(self.values)

        index_tensor = torch.LongTensor(self.indexes)
        value_tensor = torch.FloatTensor(self.values)
        coo_tensor = torch.sparse.FloatTensor(index_tensor.t(), value_tensor, torch.Size([self.num_of_covariates, self.num_of_words, self.num_of_words]))

        self.left_word_tensor = torch.randn(self.num_of_words, self.embedding_size, requires_grad=True)
        self.right_word_tensor = torch.randn(self.num_of_words, self.embedding_size, requires_grad=True)
        self.covariate_diagonal_tensor = torch.diag_embed(torch.randn(self.num_of_covariates, self.embedding_size, requires_grad=True))
        weights = torch.min((value_tensor / self.x_max), torch.ones(value_tensor.shape)) ** self.alpha
        self.weights = torch.sparse.FloatTensor(index_tensor.t(), weights, torch.Size([self.num_of_covariates, self.num_of_words, self.num_of_words]))
        #self.left_bias = torch.sparse.FloatTensor(index_tensor.t(), torch.randn(self.num_of_occurrences, requires_grad=True), torch.Size([self.num_of_covariates, self.num_of_words, self.num_of_words]))
        #self.right_bias = torch.sparse.FloatTensor(index_tensor.t(), torch.randn(self.num_of_occurrences, requires_grad=True), torch.Size([self.num_of_covariates, self.num_of_words, self.num_of_words]))
        log_values = torch.log(value_tensor)
        self.log_values = torch.sparse.FloatTensor(index_tensor.t(), log_values, torch.Size([self.num_of_covariates, self.num_of_words, self.num_of_words]))

    def get_batch(self):
        indices = torch.randperm(self.num_of_occurrences)
        for idx in range(0, self.num_of_occurrences):
            left_idx, right_idx, covariate_idx = zip(*self.indexes[idx].tolist())
            left_words = self.left_word_tensor[left_idx]
            right_words = self.right_word_tensor[right_idx]
            covariate = self.covariate_diagonal_tensor[covariate_idx]
            left_bias = self.left_bias[left_idx, covariate_idx]
            right_bias = self.right_bias[right_idx, covariate_idx]
            log_vals = [self.log_values]
            weights = self.weights[left_idx]
            yield left_words, right_words, covariate, left_bias, right_bias, log_vals, weights

    def train(self):
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer.zero_grad()
        for epoch in tqdm(range(self.epochs)):
            logging.info("Start epoch %i", epoch)
            num_batches = int(self.num_of_occurrences/self.batch_size)
            avg_loss = 0.0
            for batch in tqdm(self.get_batch(), total=num_batches, mininterval=1):
                optimizer.zero_grad()
                loss = self.get_loss(*batch)
                avg_loss += loss.data[0] / num_batches
                loss.backward()
                optimizer.step()

    def get_loss(self, left_words, right_words, covariate, left_bias, right_bias, log_vals, weights):
        left_context = covariate.mul(left_words).sum(1).view(-1)
        right_context = covariate.mul(right_words).sum(1).view(-1)
        sim = left_context.mul(right_context).sum(1).view(-1)
        x = (sim + left_bias + right_bias - log_vals) ** 2
        loss = torch.mul(x, weights)
        return loss.mean()
