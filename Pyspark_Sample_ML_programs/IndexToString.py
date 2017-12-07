from pyspark.ml.feature import StringIndexer,IndexToString
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
stringDF=sqlContext.createDataFrame([(0, 'a'), (1, 'b'), (2, 'c'), (3, 'a'), (4, 'a'), (5, 'c')], ['id', 'category'])
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed = indexer.fit(stringDF).transform(stringDF)
converter=IndexToString(inputCol="categoryIndex",outputCol="originalCategory")
converted=converter.transform(indexed)
converted.select("id","originalCategory").show()