from pyspark.ml.feature import RegexTokenizer, Tokenizer
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
sentenceDF = sqlContext.createDataFrame([
    (0, "Hi I heard about Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
], ["label", "sentence"])
# tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
regex_tokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="h+")
#regex_tokenizer_nogaps = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\w+", gaps=False)
wordsDF = regex_tokenizer.transform(sentenceDF)
for words_label in wordsDF.select("words", "label").take(3):
    print words_label

"""OUTPUT for tokenizer


Row(words=[u'hi', u'i', u'heard', u'about', u'spark'], label=0)
Row(words=[u'i', u'wish', u'java', u'could', u'use', u'case', u'classes'], label
=1)
Row(words=[u'logistic,regression,models,are,neat'], label=2)"""
"""OUTPUT FOR REGEX TOKENIZER
Row(words=[u'hi', u'i', u'heard', u'about', u'spark'], label=0)
Row(words=[u'i', u'wish', u'java', u'could', u'use', u'case', u'classes'], label
=1)
Row(words=[u'logistic,regression,models,are,neat'], label=2)
"""
