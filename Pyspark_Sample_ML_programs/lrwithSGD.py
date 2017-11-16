from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark import SQLContext, SparkContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)


def parsePoint(line):
    values = [float(x) for x in line.replace(',', '').split('')]
    return LabeledPoint(values[0], values[1:])


data = sc.textFile("D:/Spark/spark-1.6.1-bin-hadoop2.6/data/mllib/ridge-data/lpsa.data")
parsedData = data.map(parsePoint)
model = LinearRegressionWithSGD.train(parsedData, iterations=10,step=1.0)
valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds.map(lambda (x, y): (x - y) ** 2).reduce(lambda a, b: a + b) / valuesAndPreds.count()
print "Mean Squared error" + str(MSE)
