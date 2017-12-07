from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.stat import Statistics

data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
        (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
        (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
        (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)]
sc = SparkContext("local", "sample")
sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame(data, ["features"])

r1 = Statistics.corr(df, method="features").head()
print("Pearson correlation matrix:\n" + str(r1[0]))

r2 = Statistics.corr(df, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))
