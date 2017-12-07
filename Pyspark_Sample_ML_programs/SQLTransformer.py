from pyspark.ml.feature import SQLTransformer
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "samp")
sqlContext = SQLContext(sc)
df=sqlContext.createDataFrame([(0,1.0,2.0),(1,3.0,4.0),(2,5.0,6.0)],["id","v1","v2"])
sqlTrans=SQLTransformer(statement="SELECT *,(v1+v2) as v3,(v1*v2) as v4 from __THIS__")
sqlTrans.transform(df).show()
"""OUTPUT
+---+---+---+----+----+
| id| v1| v2|  v3|  v4|
+---+---+---+----+----+
|  0|1.0|2.0| 3.0| 2.0|
|  1|3.0|4.0| 7.0|12.0|
|  2|5.0|6.0|11.0|30.0|
+---+---+---+----+----+
"""