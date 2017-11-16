from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
data = sqlContext.createDataFrame([(7, Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0),
                                   (8, Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0),
                                   (9, Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0)
                                   ], ["id", "features", "clicked"])
selector=Statistics.chiSqTest(data)#ChiSqSelector is not available in Statistics instead chiSqTest
