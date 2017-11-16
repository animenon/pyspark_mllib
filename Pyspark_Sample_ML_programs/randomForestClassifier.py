from pyspark import SQLContext, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)

data = sqlContext.read.format("libsvm").load("D:\Spark\spark-1.6.1-bin-hadoop2.6\data\mllib\sample_libsvm_data.txt")
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
(trainingData, testData) = data.randomSplit([0.7, 0.3])
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])
model = pipeline.fit(trainingData)
# make predictions
predictions = model.transform(testData)
predictions.select("prediction", "indexedLabel", "features").show(7)

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                              metricName="precision")
accuracy = evaluator.evaluate(predictions)
print("Test Error= %g " % (1.0 - accuracy))
rfModel=model.stages[2]

print(rfModel)
"""OUTPUT
+----------+------------+--------------------+
|prediction|indexedLabel|            features|
+----------+------------+--------------------+
|       0.0|         1.0|(692,[100,101,102...|
|       1.0|         1.0|(692,[122,123,148...|
|       1.0|         1.0|(692,[123,124,125...|
|       1.0|         1.0|(692,[123,124,125...|
|       1.0|         1.0|(692,[124,125,126...|
+----------+------------+--------------------+
only showing top 5 rows
Test Error= 0
RandomForestClassificationModel (uid=rfc_56859ff173d4) with 20 trees"""