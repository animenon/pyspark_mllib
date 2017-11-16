from pyspark.ml.feature import StopWordsRemover
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
sentenceData = sqlContext.createDataFrame([
    (0, ["I", "saw", "the", "red", "baloon"]),
    (1, ["Mary", "had", "a", "little", "lamb"]),
    (2, ['I', 'am', 'here', 'to', 'verify'])
], ["label", "raw"])

remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
remover.transform(sentenceData).show(truncate=False)
"""OUTPUT
 
+-----+----------------------------+--------------------+
|label|raw                         |filtered            |
+-----+----------------------------+--------------------+
|0    |[I, saw, the, red, baloon]  |[saw, red, baloon]  |
|1    |[Mary, had, a, little, lamb]|[Mary, little, lamb]|
|2    |[I, am, here, to, verify]   |[verify]            |
+-----+----------------------------+--------------------+"""