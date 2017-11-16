from pyspark.ml.feature import VectorSlicer
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame([(Vectors.dense([-2.0, 2.3, 0.0]),),(Vectors.dense([1.0, 2.0, 3.0]),)],
                                ["userFeatures"])
slicer = VectorSlicer(inputCol="userFeatures", outputCol="features", indices=[1])
output = slicer.transform(df)
output.select("userFeatures", "features").show()

"""OUTPUT
+--------------+--------+
|  userFeatures|features|
+--------------+--------+
|[-2.0,2.3,0.0]|   [2.3]|
| [1.0,2.0,3.0]|   [2.0]|
+--------------+--------+
"""