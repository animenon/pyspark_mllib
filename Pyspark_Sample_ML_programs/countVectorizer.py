from pyspark.ml.feature import CountVectorizer
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame([(0, ['a', 'b', 'c']), (1, ['a', 'b', 'b', 'c', 'a'])], ["id", "words"])
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)
cvModel = cv.fit(df)
cvModel.transform(df).select("features").show()
"""OUTPUT

+--------------------+
|            features|
+--------------------+
|(3,[0,1,2],[1.0,1...|
|(3,[0,1,2],[2.0,2...|
+--------------------+"""