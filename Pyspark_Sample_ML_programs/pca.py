from pyspark.ml.feature import PCA
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
dataDF = sqlContext.createDataFrame(data, ["features"])
pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(dataDF)
result = model.transform(dataDF).select("pcaFeatures")
result.show(truncate=False)

"""OUTPUT k=3
+-----------------------------------------------------------+
|pcaFeatures                                                |
+-----------------------------------------------------------+
|[1.6485728230883807,-4.013282700516296,-5.524543751369388] |
|[-4.645104331781534,-1.1167972663619026,-5.524543751369387]|
|[-6.428880535676489,-5.337951427775355,-5.524543751369389] |
+-----------------------------------------------------------+"""
"""For k =2
+----------------------------------------+
|pcaFeatures                             |
+----------------------------------------+
|[1.6485728230883807,-4.013282700516296] |
|[-4.645104331781534,-1.1167972663619026]|
|[-6.428880535676489,-5.337951427775355] |
+----------------------------------------+
"""
