from pyspark.sql.types import StringType, ArrayType, MapType, IntegerType, StructField, StructType, FloatType
from pyspark.sql.functions import udf, explode, when, col, row_number, sum as sum_
from pyspark.sql.window import Window as window
from src.processing.utils import tokenize_document, get_n_grams
import torch


class Cover:
    def __init__(self, spark_session):
        self.transformed_data = []
        self.corpus = None
        self.spark_session = spark_session
        self.coo_tensor = None
        self.value_tensor = None
        self.coo_dict = {}

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
            tokenized_dataframe = self.corpus.withColumn('tokens', tokenize(column_name).alias('tokens'))

            words_dataframe = tokenized_dataframe.withColumn('word', explode(col('tokens'))) \
                .groupBy('word') \
                .count() \
                .sort('count', ascending=True)

            windowSpec = window.orderBy("count")

            words_with_id_dataframe = words_dataframe.withColumn('id', row_number().over(windowSpec)) \
                .sort('id', ascending=False)

            words_with_id_dataframe.show(20)

            filtered_words_with_id_dataframe = words_with_id_dataframe.withColumn('id', when(
                words_with_id_dataframe['count'] <= min_occurrence_count, 0).otherwise(words_with_id_dataframe.id))

            token_to_id = filtered_words_with_id_dataframe.rdd.map(lambda row: (row.word, row.id)).collectAsMap()

            print("There are {} unique tokens".format(len(token_to_id)))

            get_id = udf(lambda x: [token_to_id[word] for word in x], ArrayType(StringType()))
            transformed_dataframe = tokenized_dataframe.withColumn('transform', get_id('tokens').alias('transform'))

            print("Mapped tokens to unique id".format(len(token_to_id)))

            schema = StructType([
                StructField("left_context_id", IntegerType(), False),
                StructField("right_context_id", IntegerType(), False)
            ])

            n_grams = udf(lambda indexes: get_n_grams(indexes, window_size), MapType(schema, FloatType()))

            matrix = transformed_dataframe.withColumn("matrix", n_grams("transform").alias('matrix'))

            matrix.select('matrix').show(10)

            reduced = matrix.select(explode(col("matrix"))).groupBy(col("key")).agg(sum_("value"))

            print("There are {} ij pairs".format(reduced.count()))

            reduced.show(20)

    def build_cooccurence_matrix(self):
        self.coo_tensor = self.coo_dict.keys()
