from pyspark.ml.feature import Word2Vec
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "sample")
sqlContext = SQLContext(sc)
documentdata = sqlContext.createDataFrame([("Hi I heard about Spark".split(" "),),
                                           ("I wish Java could use case classes".split(" "),),
                                           ("I wish Java could use case classes".split(" "),),
                                           ("Logistic regression models are neat".split(" "),)], ['text'])
word2vec = Word2Vec(inputCol="text", minCount=0, vectorSize=3, outputCol="result")
model = word2vec.fit(documentdata)
result = model.transform(documentdata)
for feature in result.select("result").take(4):
    print feature

"""Row(result=DenseVector([-0.0365, -0.0298, 0.1027]))
Row(result=DenseVector([0.0073, 0.0195, 0.0015]))
Row(result=DenseVector([0.0487, 0.0023, 0.0424]))"""

"""Row(result=DenseVector([0.0206, -0.006, 0.0263]))
Row(result=DenseVector([-0.0373, 0.0231, 0.0388]))
Row(result=DenseVector([-0.0373, 0.0231, 0.0388]))
Row(result=DenseVector([0.0536, -0.0253, 0.0667]))"""
