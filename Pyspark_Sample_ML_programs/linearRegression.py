from pyspark import SQLContext, SparkContext
from pyspark.ml.regression import LinearRegression

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
data = sqlContext.read.format("libsvm").load(
    "D:\Spark\spark-1.6.1-bin-hadoop2.6\data\mllib\sample_linear_regression_data.txt")
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(data)
print "Coefficients" + str(lrModel.coefficients)
print "Intercept" + str(lrModel.intercept)

"""Output
Coefficients[0.0,0.322426805271,-0.34340324676,1.91507271514,0.052431961937,0.76
5582117768,0.0,-0.150663371848,-0.215438568497,0.219779818119]
Intercept0.15991519483
"""