from pyspark.ml.feature import HashingTF, IDF
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "sample")
sqlContext = SQLContext(sc)

documents = sc.textFile("D:\\Users\\703201013\\Desktop\\proxy1.txt").map(lambda line: line.split(" "))
hashingtf = HashingTF(10)
tf = hashingtf.transform(documents)
tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)
for f in tfidf.collect():
    print f
