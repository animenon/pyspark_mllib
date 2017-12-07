from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
stringDF = sqlContext.createDataFrame([(0, 'a'), (1, 'b'), (2, 'c'), (3, 'ha'), (4, '/:'), (5, '*')],
                                      ['id', 'category'])
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed = indexer.fit(stringDF).transform(stringDF)
encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVector")
encoded = encoder.transform(indexed)
encoded.select("id", "categoryVector", "category").show()

"""OUTPUT
+---+--------------+
| id|categoryVector|
+---+--------------+
|  0| (2,[0],[1.0])|
|  1|     (2,[],[])|
|  2| (2,[1],[1.0])|
|  3| (2,[0],[1.0])|
|  4| (2,[0],[1.0])|
|  5| (2,[1],[1.0])|
+---+--------------+

+---+--------------+--------+
| id|categoryVector|category|
+---+--------------+--------+
|  0| (2,[0],[1.0])|       a|
|  1|     (2,[],[])|       b|
|  2| (2,[1],[1.0])|       c|
|  3| (2,[0],[1.0])|       a|
|  4| (2,[0],[1.0])|       a|
|  5| (2,[1],[1.0])|       c|
+---+--------------+--------+


+---+--------------+--------+
| id|categoryVector|category|
+---+--------------+--------+
|  0| (5,[1],[1.0])|       a|
|  1| (5,[4],[1.0])|       b|
|  2|     (5,[],[])|       c|
|  3| (5,[2],[1.0])|      ha|
|  4| (5,[3],[1.0])|      /:|
|  5| (5,[0],[1.0])|       *|
+---+--------------+--------+ """
