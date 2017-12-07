from pyspark.ml.feature import RFormula
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
dataset = sqlContext.createDataFrame([(0, "US", 18, 1.0), (167, "IN", 16, 1.0)], ["id", "country", "hour", "clicked"])
rfor = "clicked~country+hour"
formula = RFormula(formula=rfor, featuresCol="features", labelCol="label")
output = formula.fit(dataset).transform(dataset)
output.select("features", "label").show()


"""OUTPUT
 +----------+-----+
|  features|label|
+----------+-----+
|[1.0,18.0]|  1.0|
|[0.0,16.0]|  1.0|
+----------+-----+
"""