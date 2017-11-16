from pyspark.ml.feature import StringIndexer
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
stringDF = sqlContext.createDataFrame([(0, 'aaaaaaaaaaaaaa'),  (2, 'bbbbbbbbbbbbbbb'), (3, 'animal'), (40, 'araku'), (5, 'change'),(10, 'bbc')],
                                      ['id', 'category'])
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed = indexer.fit(stringDF).transform(stringDF)
indexed.show()
for i in indexed.collect():
    print i

""" OUTPUT 
+---+--------+-------------+
| id|category|categoryIndex|
+---+--------+-------------+
|  0|       a|          0.0|
|  1|       b|          2.0|
|  2|       c|          1.0|
|  3|       a|          0.0|
|  4|       a|          0.0|
|  5|       c|          1.0|
+---+--------+-------------+

Row(id=0, category=u'a', categoryIndex=0.0)
Row(id=1, category=u'b', categoryIndex=2.0)
Row(id=2, category=u'c', categoryIndex=1.0)
Row(id=3, category=u'a', categoryIndex=0.0)
Row(id=4, category=u'a', categoryIndex=0.0)
Row(id=5, category=u'c', categoryIndex=1.0)

+---+--------+-------------+
| id|category|categoryIndex|
+---+--------+-------------+
|  0|       a|          0.0|
|  1|     bbc|          5.0|
|  2|     cat|          4.0|
|  3|  animal|          1.0|
|  4|   araku|          3.0|
|  5|  change|          2.0|
+---+--------+-------------+

Row(id=0, category=u'a', categoryIndex=0.0)
Row(id=1, category=u'bbc', categoryIndex=5.0)
Row(id=2, category=u'cat', categoryIndex=4.0)
Row(id=3, category=u'animal', categoryIndex=1.0)
Row(id=4, category=u'araku', categoryIndex=3.0)
Row(id=5, category=u'change', categoryIndex=2.0)"""

