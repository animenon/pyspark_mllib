from pyspark import SQLContext, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)

data = sqlContext.read.format("libsvm").load("D:\Spark\spark-1.6.1-bin-hadoop2.6\data\mllib\sample_libsvm_data.txt")
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them
# maxCategories >4 are treated as continuous
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
# split the data into training and test sets
(trainingData, testData) = data.randomSplit([0.7, 0.3])
# train Decision tree model
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
# Cahin indexers & trees in training model
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
model = pipeline.fit(trainingData)
# make predictions
predictions = model.transform(testData)
predictions.select("prediction", "indexedLabel", "features").show(7)

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                              metricName="precision")
accuracy = evaluator.evaluate(predictions)
print("Test Error= %g " % (1.0 - accuracy))
treeModel=model.stages[2]

print(treeModel)
"""OUTPUT
k 0.081736 s
+----------+------------+--------------------+
|prediction|indexedLabel|            features|
+----------+------------+--------------------+
|       1.0|         1.0|(692,[123,124,125...|
|       1.0|         1.0|(692,[126,127,128...|
|       1.0|         1.0|(692,[126,127,128...|
|       1.0|         1.0|(692,[126,127,128...|
|       1.0|         1.0|(692,[126,127,128...|
|       1.0|         1.0|(692,[128,129,130...|
|       1.0|         1.0|(692,[129,130,131...|
+----------+------------+--------------------+
only showing top 7 rows


Test Error= 0.03125
Test Error= 0.0645161
DecisionTreeClassificationModel (uid=DecisionTreeClassifier_4f0b97153aa56c0b5024
) of depth 1 with 3 nodes
"""