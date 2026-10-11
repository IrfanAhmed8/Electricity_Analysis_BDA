from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date
from pyspark.sql.functions import (
    col,
    avg,
    sum,
    max,
    min,
    count,
    desc,
    round
)


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("ElectricityConsumptionAnalysis") \
    .master("local[*]") \
    .getOrCreate()


# ============================================================
# 2. LOAD DATA
# ============================================================

df = spark.read.csv(
    "data/electricity_consumption.csv",
    header=True,
    inferSchema=True
)


# ============================================================
# 3. HOURLY CONSUMPTION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("1. HOURLY ELECTRICITY CONSUMPTION")
print("=" * 70)

hourly = df.groupBy("hour") \
    .agg(
        round(avg("consumption_kwh"), 3).alias("average_consumption"),
        round(sum("consumption_kwh"), 2).alias("total_consumption"),
        count("*").alias("records")
    ) \
    .orderBy("hour")

hourly.show(24)


# Peak hour
peak_hour = df.groupBy("hour") \
    .agg(avg("consumption_kwh").alias("average_consumption")) \
    .orderBy(desc("average_consumption")) \
    .first()

print(
    f"Peak consumption hour: {peak_hour['hour']}:00 "
    f"({peak_hour['average_consumption']:.3f} kWh)"
)



# ============================================================
# 2. DAILY ELECTRICITY CONSUMPTION
# ============================================================

print("\n" + "=" * 70)
print("2. DAILY ELECTRICITY CONSUMPTION")
print("=" * 70)

# Extract date from timestamp
df_daily = df.withColumn(
    "date",
    to_date("timestamp")
)

daily = df_daily.groupBy("date") \
    .agg(
        round(sum("consumption_kwh"), 2)
        .alias("total_consumption")
    ) \
    .orderBy("date")


print("Number of days:", daily.count())

print("\nFirst 10 days:")
daily.show(10)


# Highest consumption day
highest_day = daily.orderBy(
    desc("total_consumption")
).first()

print(
    f"\nHighest consumption day: "
    f"{highest_day['date']} → "
    f"{highest_day['total_consumption']:.2f} kWh"
)


# Lowest consumption day
lowest_day = daily.orderBy(
    "total_consumption"
).first()

print(
    f"Lowest consumption day: "
    f"{lowest_day['date']} → "
    f"{lowest_day['total_consumption']:.2f} kWh"
)


# ============================================================
# 5. HOUSEHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("3. HOUSEHOLD CONSUMPTION")
print("=" * 70)

household = df.groupBy("household_id") \
    .agg(
        count("*").alias("records"),
        round(sum("consumption_kwh"), 2).alias("total_consumption"),
        round(avg("consumption_kwh"), 3).alias("average_consumption"),
        round(max("consumption_kwh"), 2).alias("maximum_consumption"),
        round(min("consumption_kwh"), 2).alias("minimum_consumption")
    ) \
    .orderBy(desc("total_consumption"))

household.show()


# ============================================================
# 6. DAY OF WEEK ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("4. DAY OF WEEK ANALYSIS")
print("=" * 70)

day_analysis = df.groupBy("day_of_week") \
    .agg(
        round(avg("consumption_kwh"), 3).alias("average_consumption"),
        round(sum("consumption_kwh"), 2).alias("total_consumption")
    ) \
    .orderBy("day_of_week")

day_analysis.show()


# ============================================================
# 7. TEMPERATURE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("5. TEMPERATURE ANALYSIS")
print("=" * 70)

temperature = df.groupBy(
    "temperature"
).agg(
    round(avg("consumption_kwh"), 3).alias("average_consumption"),
    count("*").alias("records")
).orderBy("temperature")

temperature.show(20)


# ============================================================
# 8. PEAK CONSUMPTION RECORDS
# ============================================================

print("\n" + "=" * 70)
print("6. TOP 20 PEAK CONSUMPTION RECORDS")
print("=" * 70)

df.select(
    "timestamp",
    "household_id",
    "consumption_kwh",
    "temperature",
    "hour",
    "day_of_week"
).orderBy(
    desc("consumption_kwh")
).show(20, truncate=False)


# ============================================================
# 9. OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("7. OVERALL SUMMARY")
print("=" * 70)

summary = df.select(
    round(sum("consumption_kwh"), 2).alias("total_kwh"),
    round(avg("consumption_kwh"), 3).alias("average_kwh"),
    round(min("consumption_kwh"), 2).alias("minimum_kwh"),
    round(max("consumption_kwh"), 2).alias("maximum_kwh")
)

summary.show()


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()