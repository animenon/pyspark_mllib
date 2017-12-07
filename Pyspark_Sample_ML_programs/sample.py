from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark import SparkContext

sc = SparkContext("local", "sample")
sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame([('2017-09-14',)], ['a'])
out = df.select(dayofyear('a').alias('day')).collect()
print out