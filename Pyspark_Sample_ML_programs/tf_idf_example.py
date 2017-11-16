from pyspark.ml.feature import HashingTF, Tokenizer,IDF
from pyspark import SparkContext
from pyspark.sql import Row, SQLContext

sc=SparkContext('local','sample')
sqlContext=SQLContext(sc)
sentencedata=sqlContext.createDataFrame([(0, "Hi I heard about Spark"),
    (0, "I wish Java could use case classes"),
    (1, "Logistic regression models are neat")],['label','sentence'])
tokenizer=Tokenizer(inputCol="sentence",outputCol="words")
wordsData=tokenizer.transform(sentencedata)
hashTF=HashingTF(inputCol="words",outputCol="rawFeatures",numFeatures=20)
featurisedData=hashTF.transform(wordsData)
idf=IDF(inputCol="rawFeatures",outputCol="features")
idfModel=idf.fit(featurisedData)
rescaleddata=idfModel.transform(featurisedData)
for features_label in rescaleddata.select('features','label').take(3):
    print features_label

"""Row(features=SparseVector(20, {5: 0.0, 6: 0.6931, 9: 1.3863}), label=0)
Row(features=SparseVector(20, {3: 1.3863, 5: 0.0, 12: 0.2877, 14: 0.2877, 18: 0.
2877}), label=0)
Row(features=SparseVector(20, {5: 0.0, 12: 0.5754, 14: 0.2877, 18: 0.2877}), lab
el=1)"""