from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark import SparkContext
from pyspark.sql import Row, SQLContext

sc = SparkContext("local", "sample")
sqlContext = SQLContext(sc)
LabeledDocument = Row("id", "text", "label")
training = sqlContext.createDataFrame([(0L, 'a b c d e spark', 1.0), (1L, "b d", 0.0),
                                       (2L, "spark f g h", 1.0), (3L, "hadoop mapreduce", 0.0)],
                                      ["id", "text", "label"])
tokenizer = Tokenizer(inputCol="text",outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.01)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
model = pipeline.fit(training)
test = sqlContext.createDataFrame([(4L, "spark i j k"),
                                   (5L, "l m n"),
                                   (6L, "mapreduce spark"),
                                   (7L, "apache hadoop")], ["id", "text"])
prediction=model.transform(test)
selected=prediction.select("id","text","prediction")
for row in selected.collect():
    print row
