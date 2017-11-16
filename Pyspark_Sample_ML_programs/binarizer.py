from pyspark.ml.feature import Binarizer
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
continuousDF = sqlContext.createDataFrame([(0, 0.1), (1, 0.8), (2, 0.8)], ['label', 'features'])
binarizer = Binarizer(inputCol="features", outputCol="binarized_feature", threshold=0.5)
binarizedDF = binarizer.transform(continuousDF)
for bf in binarizedDF.select("binarized_feature").take(3):
    print bf

"""OUTPUT 
Row(binarized_feature=0.0)
Row(binarized_feature=1.0)
Row(binarized_feature=1.0)
"""