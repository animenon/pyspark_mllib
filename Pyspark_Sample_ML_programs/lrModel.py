from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)


def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])


data = sc.textFile("D:\Spark\spark-1.6.1-bin-hadoop2.6\data\mllib\sample_svm_data.txt")
parsedData = data.map(parsePoint)
# parsedData.take(3)
model = LogisticRegressionWithLBFGS.train(parsedData)
labelsAndPreds = parsedData.map(lambda x: (x.label, model.predict(x.features)))
trainErr = labelsAndPreds.filter(lambda (x, y): x != y).count() / float(parsedData.count())
print "Training Error =" + str(trainErr)
model.save(sc, "myModelPath")
saveModel = LogisticRegressionModel.load(sc, "myModelPath")
