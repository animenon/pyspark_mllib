from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
dataset = sqlContext.createDataFrame([(0, 9, 2.0, Vectors.dense([0.0, 0.1, 0.2]), 1.0)],
                                     ["id", "hr", "mobile", "userFeatures", "clicked"])
assembler = VectorAssembler(inputCols=["id", "hr", "mobile", "userFeatures", "clicked"], outputCol="assembledFeature")
assembled = assembler.transform(dataset)
print assembled.select("assembledFeature", "clicked").first()

"""OUTPUT
Row(assembledFeature=DenseVector([0.0, 9.0, 2.0, 0.0, 0.1, 0.2, 1.0]), clicked=1.0)"""
