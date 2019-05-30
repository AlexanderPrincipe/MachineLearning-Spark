from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

conf = SparkConf().setAppName("RRHH_ML").setMaster("local")
sc  = SparkContext(conf=conf)

sqlContext = SQLContext(sc)

# Preprocesado de data
rdd_empleados = sqlContext.read.csv("core_dataset.csv", header=True).rdd

rdd_empleados = rdd_empleados.filter(
	lambda x : x[5] != None and x[17] != None)

rdd_training = rdd_empleados.map(lambda x : (int(x[5]), float(x[17])))
df = rdd_training.toDF(["Edad", "Pago por hora"])

assembler = VectorAssembler(inputCols=["Edad"], outputCol="Features")
training_df = assembler.transform(df)

lr = LinearRegression(
	maxIter=5, 
	featuresCol="Features", 
	labelCol="Pago por hora",
	regParam=0.001)
lr_model = lr.fit(training_df)

df_predictions = lr_model.transform(training_df)

print(lr_model.coefficients)
print(lr_model.intercept)

df_predictions.show()

evaluator = RegressionEvaluator(
	labelCol="Pago por hora", 
	metricName="rmse", 
	predictionCol="prediction")
metrica_resultante = evaluator.evaluate(df_predictions)
print("RMSE: {}".format(metrica_resultante))









