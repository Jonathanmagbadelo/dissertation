import time
from pyspark.sql import SparkSession
from pyspark import SparkConf
import os
from modelling import Cover

os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'
os.environ['PYTHONPATH'] = '$PYTHONPATH:/opt/training'
os.chdir('/opt/training/src')

conf = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "3g")

spark_session = SparkSession\
    .builder\
    .config(conf=conf)\
    .appName("Cover")\
    .getOrCreate()

spark_session.sparkContext.addPyFile("/opt/training/src/modelling/Cover.py")
spark_session.sparkContext.addPyFile("/opt/training/src/processing/utils.py")

filename = '/opt/training/data/raw/billboard_lyrics_1964-2015.csv'
cover = Cover.Cover(spark_session=spark_session, embedding_size=100, x_max=100, alpha=.75, weight_decay=1e-8, learning_rate=0.05, epochs=1, batch_size=512, device_type='cuda')

start_time = time.time()
cover.import_data(filename)
cover.fit_transform(column_name='Lyrics', covariate='genre', min_occurrence_count=25, window_size=5)
#cover.build_co_occurrence_matrix()
cover.build_coo_matrix()
cover.train()
cover.tsne_plot()
end_time = time.time()

print("Time taken is {}".format(end_time-start_time))
