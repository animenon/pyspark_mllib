from pyspark.ml.feature import DCT
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from pyspark.sql import SQLContext


sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame([(Vectors.dense([-2.0, 2.3, 0.0]),), (Vectors.dense([1.0, 2.0, 3.0]),)],
                                ["features"])
dct = DCT(inputCol="features", outputCol="DCTfeatures", inverse=False)
dctmodel = dct.transform(df)
dctmodel.select("DCTfeatures").show()
"""OUTPUT
+--------------------+
|         DCTfeatures|
+--------------------+
|[0.17320508075688...|
|[3.46410161513775...|
+--------------------+"""