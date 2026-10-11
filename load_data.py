from pyspark.sql import SparkSession


# --------------------------------------------------
# 1. CREATE SPARK SESSION
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("ElectricityConsumptionAnalysis") \
    .master("local[*]") \
    .getOrCreate()


# --------------------------------------------------
# 2. LOAD ELECTRICITY DATASET
# --------------------------------------------------

file_path = "data/electricity_consumption.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)


# --------------------------------------------------
# 3. DISPLAY BASIC INFORMATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("ELECTRICITY CONSUMPTION DATASET")
print("=" * 60)

print("\nNumber of rows:", df.count())

print("\nNumber of columns:", len(df.columns))

print("\nColumns:")
print(df.columns)


# --------------------------------------------------
# 4. DISPLAY SCHEMA
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATA SCHEMA")
print("=" * 60)

df.printSchema()


# --------------------------------------------------
# 5. DISPLAY FIRST 10 RECORDS
# --------------------------------------------------

print("\n" + "=" * 60)
print("FIRST 10 RECORDS")
print("=" * 60)

df.show(10, truncate=False)


# --------------------------------------------------
# 6. STOP SPARK
# --------------------------------------------------

spark.stop()