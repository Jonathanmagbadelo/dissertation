import time
from pyspark.sql import SparkSession
import os
from modelling import Cover

os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'
os.environ['PYTHONPATH'] = '$PYTHONPATH:/opt/training'

spark_session = SparkSession\
    .builder\
    .appName("Cover")\
    .getOrCreate()

spark_session.sparkContext.addPyFile("/opt/training/src/modelling/Cover.py")
spark_session.sparkContext.addPyFile("/opt/training/src/processing/utils.py")

os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'
os.environ['PYTHONPATH'] = '$PYTHONPATH:/opt/training'
os.chdir('/opt/training/src')

spark_session = SparkSession\
    .builder\
    .appName("Cover")\
    .getOrCreate()

spark_session.sparkContext.addPyFile("/opt/training/src/modelling/Cover.py")
spark_session.sparkContext.addPyFile("/opt/training/src/processing/utils.py")

filename = '/opt/training/data/raw/test-lyrics.csv'
cover = Cover.Cover(spark_session=spark_session, embedding_size=300, x_max=100, alpha=.75, weight_decay=1, learning_rate=1e-8, epochs=10, batch_size=512)

start_time = time.time()
cover.import_data(filename)
cover.fit_transform(column_name='Lyrics', covariate='Genre', min_occurrence_count=5, window_size=5)
cover.build_co_occurrence_matrix()
end_time = time.time()

print("Time taken is {}".format(end_time-start_time))