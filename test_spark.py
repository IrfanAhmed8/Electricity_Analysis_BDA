from pyspark.sql import SparkSession

print("Starting Spark...")

spark = SparkSession.builder \
    .appName("ElectricityConsumptionAnalysis") \
    .master("local[*]") \
    .getOrCreate()

print("Spark started successfully!")
print("Spark version:", spark.version)

data = [
    ("HH001", 1.25),
    ("HH002", 2.10),
    ("HH003", 1.75),
    ("HH004", 2.50),
    ("HH005", 3.10)
]

columns = ["household_id", "consumption_kwh"]

df = spark.createDataFrame(data, columns)

df.show()

spark.stop()

print("Spark stopped successfully!")