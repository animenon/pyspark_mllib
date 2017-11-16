from pyspark.ml.feature import PolynomialExpansion
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
dataDF = sqlContext.createDataFrame([(Vectors.dense([-2.0, 2.3]),),
                                     (Vectors.dense([0.0, 0.0]),), (Vectors.dense([0.6, -1.1]),)], ["features"])
px = PolynomialExpansion(degree=3, inputCol="features", outputCol="polyFeatures")
polyDF = px.transform(dataDF)
for expanded in polyDF.select("polyFeatures").take(3):
    print expanded

"""OUTPUT 
Row(polyFeatures=DenseVector([-2.0, 4.0, 2.3, -4.6, 5.29]))
Row(polyFeatures=DenseVector([0.0, 0.0, 0.0, 0.0, 0.0]))
Row(polyFeatures=DenseVector([0.6, 0.36, -1.1, -0.66, 1.21]))"""

"""
Row(polyFeatures=DenseVector([-2.0, 4.0, -8.0, 2.3, -4.6, 9.2, 5.29, -10.58, 12,167]))
Row(polyFeatures=DenseVector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
Row(polyFeatures=DenseVector([0.6, 0.36, 0.216, -1.1, -0.66, -0.396, 1.21, 0.72
, -1.331]))"""
