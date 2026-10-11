from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    sum,
    avg,
    min,
    max
)


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("ElectricityDataProcessing") \
    .master("local[*]") \
    .getOrCreate()


# ============================================================
# 2. LOAD DATA
# ============================================================

file_path = "data/electricity_consumption.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Rows:", df.count())
print("Columns:", len(df.columns))

df.printSchema()


# ============================================================
# 4. CHECK NULL VALUES
# ============================================================

print("\n" + "=" * 70)
print("NULL VALUE ANALYSIS")
print("=" * 70)

null_counts = df.select([
    count(
        col(c)
    ).alias(c)
    for c in df.columns
])

print("Non-null values:")
null_counts.show()


# ============================================================
# 5. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

total_rows = df.count()
unique_rows = df.dropDuplicates().count()

print("Total rows:", total_rows)
print("Unique rows:", unique_rows)
print("Duplicate rows:", total_rows - unique_rows)


# ============================================================
# 6. UNIQUE HOUSEHOLDS
# ============================================================

print("\n" + "=" * 70)
print("HOUSEHOLD ANALYSIS")
print("=" * 70)

print("Number of households:")

df.select("household_id") \
    .distinct() \
    .show()


print("Records per household:")

df.groupBy("household_id") \
    .count() \
    .orderBy("household_id") \
    .show()


# ============================================================
# 7. CONSUMPTION STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("CONSUMPTION STATISTICS")
print("=" * 70)

df.select(
    sum("consumption_kwh").alias("total_consumption"),
    avg("consumption_kwh").alias("average_consumption"),
    min("consumption_kwh").alias("minimum_consumption"),
    max("consumption_kwh").alias("maximum_consumption")
).show()


# ============================================================
# 8. TEMPERATURE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("TEMPERATURE STATISTICS")
print("=" * 70)

df.select(
    avg("temperature").alias("average_temperature"),
    min("temperature").alias("minimum_temperature"),
    max("temperature").alias("maximum_temperature")
).show()


# ============================================================
# 9. DATA VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

invalid_hours = df.filter(
    (col("hour") < 0) | (col("hour") > 23)
).count()

invalid_days = df.filter(
    (col("day_of_week") < 0) |
    (col("day_of_week") > 6)
).count()

negative_consumption = df.filter(
    col("consumption_kwh") < 0
).count()

print("Invalid hours:", invalid_hours)
print("Invalid days:", invalid_days)
print("Negative consumption:", negative_consumption)


# ============================================================
# 10. FINAL DATASET PREVIEW
# ============================================================

print("\n" + "=" * 70)
print("CLEAN DATASET PREVIEW")
print("=" * 70)

df.show(10)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()