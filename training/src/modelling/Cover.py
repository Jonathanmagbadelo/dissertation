from pyspark.sql.types import StringType, ArrayType, MapType, IntegerType, FloatType
from pyspark.sql.functions import udf, explode, when, col, row_number, sum as sum_
from pyspark.sql.window import Window as window
from src.processing.utils import tokenize_document, get_n_grams
from torch import FloatTensor, LongTensor, sparse, Size, randn


class Cover:
    def __init__(self, spark_session, embedding_size):
        self.transformed_data = []
        self.corpus = None
        self.spark_session = spark_session
        self.indexes = None
        self.values = None
        self.coo_list = []
        self.num_of_words = None
        self.embedding_size = embedding_size

    def import_data(self, filename):
        self.corpus = self.spark_session.read. \
            format("csv") \
            .option("header", "True") \
            .option("mode", "DROPMALFORMED") \
            .load(filename)

        print("Corpus has {} documents".format(self.corpus.count()))

    def fit_transform(self, column_name, min_occurrence_count, window_size):
        if self.corpus is None:
            print("Please load corpus first!")
        else:
            tokenize = udf(lambda document: tokenize_document(document), ArrayType(StringType()))

            self.corpus.show(10)

            tokenized_dataframe = self.corpus.withColumn('tokens', tokenize(column_name[0]).alias('tokens'))

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

            reduced = matrix.select('Genre', explode(col("matrix")).alias('key', 'value'))\
                .groupBy(col("key"), col(column_name[1]))\
                .agg(sum_("value").alias("value"))

            print("There are {} ij pairs".format(reduced.count()))

            reduced.show(20)

            x = reduced.select("value", "key", column_name[1]).rdd\
                .map(lambda row: (row.key, row.Genre, row.value))\
                .groupBy(lambda row: row[1])\
                .mapValues(list)\
                .collect()

            self.coo_list = [zip(*y[1]) for y in x]

            self.spark_session.stop()

    def build_co_occurrence_matrix(self):
        self.indexes, genres, self.values = self.coo_list[0]
        num_entries = len(self.values)
        print(num_entries)

        print(self.indexes[:5])
        print(self.values[:5])

        index_tensor = LongTensor(self.indexes)
        value_tensor = FloatTensor(self.values)
        coo_tensor = sparse.FloatTensor(index_tensor.t(), value_tensor, Size([self.num_of_words, self.num_of_words]))

        #TODO
        """
        Need to implement left/right word vectors, bias, and weight function
        """

        left_word_vector = randn(self.num_of_words, self.embedding_size)
        right_word_vector = randn(self.num_of_words, self.embedding_size)
        covariate_vector = randn(self.num_of_words, self.embedding_size)
        left_bias = []
        right_bias = []

