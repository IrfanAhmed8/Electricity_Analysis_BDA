from pyspark.sql import SparkSession
from pyspark.sql.functions import month
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


DATA_PATH = "data/electricity_consumption.csv"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("ElectricityConsumptionPrediction")
        .master("local[*]")
        .getOrCreate()
    )


def load_data(spark):
    return (
        spark.read
        .csv(DATA_PATH, header=True, inferSchema=True)
        .withColumn("month", month("timestamp"))
    )


def build_model():

    household_indexer = StringIndexer(
        inputCol="household_id",
        outputCol="household_index",
        handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=[
            "household_index",
            "hour",
            "day_of_week",
            "temperature",
            "month"
        ],
        outputCol="features"
    )

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="consumption_kwh",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    pipeline = Pipeline(
        stages=[
            household_indexer,
            assembler,
            rf
        ]
    )

    return pipeline


def train_model(df):

    train_df, test_df = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    pipeline = build_model()

    model = pipeline.fit(train_df)

    predictions = model.transform(test_df)

    rmse_evaluator = RegressionEvaluator(
        labelCol="consumption_kwh",
        predictionCol="prediction",
        metricName="rmse"
    )

    r2_evaluator = RegressionEvaluator(
        labelCol="consumption_kwh",
        predictionCol="prediction",
        metricName="r2"
    )

    rmse = rmse_evaluator.evaluate(predictions)
    r2 = r2_evaluator.evaluate(predictions)

    return model, predictions, rmse, r2


if __name__ == "__main__":

    spark = create_spark_session()

    print("\nLoading dataset...")
    df = load_data(spark)

    print(f"Total records: {df.count()}")

    print("\nDataset with month feature:")
    df.select(
        "timestamp",
        "household_id",
        "consumption_kwh",
        "temperature",
        "hour",
        "day_of_week",
        "month"
    ).show(10, truncate=False)

    print("\nTraining Random Forest model...")

    model, predictions, rmse, r2 = train_model(df)

    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")
    print(f"RMSE: {rmse:.4f} kWh")
    print(f"R²:   {r2:.4f}")

    print("\nSample predictions:")
    predictions.select(
        "timestamp",
        "household_id",
        "consumption_kwh",
        "prediction"
    ).show(20, truncate=False)

    spark.stop()