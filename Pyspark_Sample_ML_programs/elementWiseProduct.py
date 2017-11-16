from pyspark.ml.feature import ElementwiseProduct
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)

data = [(Vectors.dense([1.0, 2.0, 3.0]),), (Vectors.dense([4.0, 5.0, 6.0]),)]
df = sqlContext.createDataFrame(data, ["vector"])
transformer = ElementwiseProduct(scalingVec=Vectors.dense([0.0, 1.0, 2.0]), inputCol="vector",
                                 outputCol="transformedVector")
transformer.transform(df).show()
"""OUTPUT
+-------------+-----------------+
|       vector|transformedVector|
+-------------+-----------------+
|[1.0,2.0,3.0]|    [0.0,2.0,6.0]|
|[4.0,5.0,6.0]|   [0.0,5.0,12.0]|
+-------------+-----------------+
"""