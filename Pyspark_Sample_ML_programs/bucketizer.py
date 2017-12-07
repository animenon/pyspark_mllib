from pyspark.ml.feature import Bucketizer
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
splits = [-float("inf"), -0.5, 0.0, 0.5, float("inf")]
data = [(-0.5,), (-0.3,), (0.0,), (0.2,), (1.2,), (-2.8,), (-5.9,), (5.9,), (-0.75,), (-0.45,), (-0.55,),(0.5,)]
dataFrame = sqlContext.createDataFrame(data, ["features"])
bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bucketfeatures")
bucketized = bucketizer.transform(dataFrame)
bucketized.show()
"""OUTPUT
+--------+--------------+
|features|bucketfeatures|
+--------+--------------+
|    -0.5|           1.0|
|    -0.3|           1.0|
|     0.0|           2.0|
|     0.2|           2.0|
+--------+--------------+

+--------+--------------+
|features|bucketfeatures|
+--------+--------------+
|    -0.5|           1.0|
|    -0.3|           1.0|
|     0.0|           2.0|
|     0.2|           2.0|
|     1.2|           3.0|
|    -2.8|           0.0|
|    -5.9|           0.0|
|     5.9|           3.0|
|   -0.75|           0.0|
|   -0.45|           1.0|
|   -0.55|           0.0|
+--------+--------------+

when the split value doesnot include 0 ie splits = [-float("inf"), -0.5, 0.5, float("inf")]
+--------+--------------+
|features|bucketfeatures|
+--------+--------------+
|    -0.5|           1.0|
|    -0.3|           1.0|
|     0.0|           1.0|
|     0.2|           1.0|
|     1.2|           2.0|
|    -2.8|           0.0|
|    -5.9|           0.0|
|     5.9|           2.0|
|   -0.75|           0.0|
|   -0.45|           1.0|
|   -0.55|           0.0|
+--------+--------------+
"""
