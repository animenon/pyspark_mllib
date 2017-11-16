from pyspark.ml.feature import NGram
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
wordDataFrame = sqlContext.createDataFrame([
    (0, ["Hi", "I", "heard", "about", "Spark"]),
    (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
    (2, ["Logistic", "regression", "models", "are", "neat"])
], ["label", "words"])
ngram = NGram(inputCol="words", outputCol="ngrams",n=3)
ngramDataFrame = ngram.transform(wordDataFrame)
for ngrams_label in ngramDataFrame.select("ngrams", "label").take(3):
    print(ngrams_label)

"""OUTPUT,by default n is 2
Row(ngrams=[u'Hi I', u'I heard', u'heard about', u'about Spark'], label=0)
Row(ngrams=[u'I wish', u'wish Java', u'Java could', u'could use', u'use case', u
'case classes'], label=1)
Row(ngrams=[u'Logistic regression', u'regression models', u'models are', u'are n
eat'], label=2)

OUTPUT when n is 3

Row(ngrams=[u'Hi I heard', u'I heard about', u'heard about Spark'], label=0)
Row(ngrams=[u'I wish Java', u'wish Java could', u'Java could use', u'could use c
ase', u'use case classes'], label=1)
Row(ngrams=[u'Logistic regression models', u'regression models are', u'models ar
e neat'], label=2)
"""