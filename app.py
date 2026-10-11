import streamlit as st
import pandas as pd
from pyspark.sql.functions import month as spark_month
from src.prediction import build_model, train_model
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    avg,
    sum,
    max,
    min,
    count,
    desc,
    round,
    to_date
)

import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Electricity Consumption Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #666666;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    margin-top: 5px;
}

.sidebar-title {
    font-size: 22px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SPARK SESSION
# ============================================================

@st.cache_resource
def create_spark():

    spark = SparkSession.builder \
        .appName("ElectricityConsumptionDashboard") \
        .master("local[*]") \
        .getOrCreate()

    return spark


spark = create_spark()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_data():
    df = spark.read.csv(
        "data/electricity_consumption.csv",
        header=True,
        inferSchema=True
    )

    df = df.withColumn(
        "month",
        spark_month("timestamp")
    )

    return df



df = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-title">⚡ Electricity Analyzer</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Consumption Analysis",
        "Household Analysis",
        "Peak & Anomalies",
        "Prediction",
        "About Project"
    ]
)


# ============================================================
# COMMON STATISTICS
# ============================================================

total_consumption = df.select(
    sum("consumption_kwh")
).collect()[0][0]

average_consumption = df.select(
    avg("consumption_kwh")
).collect()[0][0]

peak_consumption = df.select(
    max("consumption_kwh")
).collect()[0][0]

household_count = df.select(
    "household_id"
).distinct().count()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("⚡ Electricity Consumption Analyzer")

    st.markdown(
        "### Big Data Analytics Dashboard powered by Apache Spark"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Consumption",
            f"{total_consumption:,.2f} kWh"
        )

    with col2:

        st.metric(
            "Average Consumption",
            f"{average_consumption:.3f} kWh"
        )

    with col3:

        st.metric(
            "Peak Consumption",
            f"{peak_consumption:.2f} kWh"
        )

    with col4:

        st.metric(
            "Households",
            household_count
        )

    st.markdown("---")

    # --------------------------------------------------------
    # HOURLY ANALYSIS
    # --------------------------------------------------------

    hourly = df.groupBy("hour") \
        .agg(
            round(
                avg("consumption_kwh"), 3
            ).alias("average_consumption")
        ) \
        .orderBy("hour")

    hourly_pd = hourly.toPandas()

    fig_hourly = px.line(
        hourly_pd,
        x="hour",
        y="average_consumption",
        markers=True,
        title="Average Electricity Consumption by Hour",
        labels={
            "hour": "Hour of Day",
            "average_consumption": "Average Consumption (kWh)"
        }
    )

    fig_hourly.update_layout(
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_hourly,
        use_container_width=True
    )

    # --------------------------------------------------------
    # HOUSEHOLD + MONTHLY
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    # Household
    with col1:

        household = df.groupBy(
            "household_id"
        ).agg(
            round(
                sum("consumption_kwh"), 2
            ).alias("total_consumption")
        ).orderBy("household_id")

        household_pd = household.toPandas()

        fig_household = px.bar(
            household_pd,
            x="household_id",
            y="total_consumption",
            title="Total Consumption by Household",
            labels={
                "household_id": "Household",
                "total_consumption": "Consumption (kWh)"
            },
            text_auto=".0f"
        )

        st.plotly_chart(
            fig_household,
            use_container_width=True
        )

    # Monthly
    with col2:

        df_month = df.withColumn(
            "month",
            col("timestamp").cast("date")
        )

        monthly = df.withColumn(
            "month_num",
            col("timestamp").cast("date")
        )

        monthly_pd = df.toPandas()

        monthly_pd["month"] = pd.to_datetime(
            monthly_pd["timestamp"]
        ).dt.month

        monthly_result = (
            monthly_pd
            .groupby("month")["consumption_kwh"]
            .sum()
            .reset_index()
        )

        monthly_result["month_name"] = pd.to_datetime(
            monthly_result["month"],
            format="%m"
        ).dt.strftime("%b")

        fig_monthly = px.bar(
            monthly_result,
            x="month_name",
            y="consumption_kwh",
            title="Monthly Electricity Consumption",
            labels={
                "month_name": "Month",
                "consumption_kwh": "Consumption (kWh)"
            },
            text_auto=".0f"
        )

        st.plotly_chart(
            fig_monthly,
            use_container_width=True
        )


# ============================================================
# CONSUMPTION ANALYSIS
# ============================================================

elif page == "Consumption Analysis":

    st.title("📊 Consumption Analysis")

    st.markdown(
        "Explore electricity consumption across different time periods."
    )

    # --------------------------------------------------------
    # DAILY CONSUMPTION
    # --------------------------------------------------------

    df_daily = df.withColumn(
        "date",
        to_date("timestamp")
    )

    daily = df_daily.groupBy(
        "date"
    ).agg(
        round(
            sum("consumption_kwh"), 2
        ).alias("total_consumption")
    ).orderBy("date")

    daily_pd = daily.toPandas()

    fig_daily = px.line(
        daily_pd,
        x="date",
        y="total_consumption",
        title="Daily Electricity Consumption",
        labels={
            "date": "Date",
            "total_consumption": "Consumption (kWh)"
        }
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True
    )

    # --------------------------------------------------------
    # DAY OF WEEK
    # --------------------------------------------------------

    weekday = df.groupBy(
        "day_of_week"
    ).agg(
        round(
            avg("consumption_kwh"), 3
        ).alias("average_consumption")
    ).orderBy("day_of_week")

    weekday_pd = weekday.toPandas()

    day_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }

    weekday_pd["day_name"] = weekday_pd[
        "day_of_week"
    ].map(day_names)

    fig_weekday = px.bar(
        weekday_pd,
        x="day_name",
        y="average_consumption",
        title="Average Consumption by Day of Week",
        labels={
            "day_name": "Day",
            "average_consumption": "Average Consumption (kWh)"
        },
        text_auto=".2f"
    )

    st.plotly_chart(
        fig_weekday,
        use_container_width=True
    )


# ============================================================
# HOUSEHOLD ANALYSIS
# ============================================================

elif page == "Household Analysis":

    st.title("🏠 Household Analysis")

    household = df.groupBy(
        "household_id"
    ).agg(
        count("*").alias("records"),
        round(
            sum("consumption_kwh"), 2
        ).alias("total_consumption"),
        round(
            avg("consumption_kwh"), 3
        ).alias("average_consumption"),
        round(
            max("consumption_kwh"), 2
        ).alias("maximum_consumption")
    ).orderBy(
        desc("total_consumption")
    )

    household_pd = household.toPandas()

    # Table
    st.dataframe(
        household_pd,
        use_container_width=True,
        hide_index=True
    )

    # Chart
    fig = px.bar(
        household_pd,
        x="household_id",
        y="total_consumption",
        title="Household Total Consumption",
        text_auto=".0f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PEAK & ANOMALIES
# ============================================================

elif page == "Peak & Anomalies":

    st.title("🔥 Peak Consumption & Anomalies")

    st.markdown(
        "Identify unusually high electricity consumption events."
    )

    # --------------------------------------------------------
    # OUTLIER THRESHOLD
    # --------------------------------------------------------

    quartiles = df.approxQuantile(
        "consumption_kwh",
        [0.25, 0.75],
        0.01
    )

    q1 = quartiles[0]
    q3 = quartiles[1]

    iqr = q3 - q1

    upper_limit = q3 + 1.5 * iqr

    st.metric(
        "Anomaly Threshold",
        f"{upper_limit:.2f} kWh"
    )

    anomalies = df.filter(
        col("consumption_kwh") > upper_limit
    )

    anomaly_count = anomalies.count()

    st.metric(
        "Detected Anomalies",
        f"{anomaly_count:,}"
    )

    st.markdown("### Highest Consumption Events")

    peak_df = anomalies.orderBy(
        desc("consumption_kwh")
    ).limit(20)

    peak_pd = peak_df.toPandas()

    st.dataframe(
        peak_pd,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT
# ============================================================
elif page == "Prediction":

    st.title("⚡ Electricity Consumption Prediction")

    st.write(
        "Predict electricity consumption using a Random Forest "
        "Regression model powered by Apache Spark MLlib."
    )

    st.divider()

    # Train and cache the model
    @st.cache_resource
    def get_prediction_model():
        model, predictions, rmse, r2 = train_model(df)
        return model, rmse, r2

    with st.spinner("Training Spark ML model..."):
        model, rmse, r2 = get_prediction_model()

    st.subheader("Enter Prediction Parameters")

    col1, col2 = st.columns(2)

    with col1:
        household = st.selectbox(
            "Household",
            ["HH001", "HH002", "HH003", "HH004", "HH005"]
        )

        hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=19
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=25.0,
            step=0.5
        )

    with col2:
        day_of_week = st.selectbox(
            "Day of Week",
            [
                ("Monday", 0),
                ("Tuesday", 1),
                ("Wednesday", 2),
                ("Thursday", 3),
                ("Friday", 4),
                ("Saturday", 5),
                ("Sunday", 6)
            ],
            format_func=lambda x: x[0]
        )

        month_value = st.selectbox(
            "Month",
            list(range(1, 13)),
            format_func=lambda x: pd.Timestamp(
                month=x,
                day=1,
                year=2023
            ).strftime("%B")
        )

    st.divider()

    if st.button("🔮 Predict Consumption", type="primary"):

        input_data = spark.createDataFrame(
            [
                (
                    household,
                    hour,
                    day_of_week[1],
                    temperature,
                    month_value
                )
            ],
            schema=[
                "household_id",
                "hour",
                "day_of_week",
                "temperature",
                "month"
            ]
        )

        # Verify the features sent to the model
        input_data = input_data.select(
            "household_id",
            "hour",
            "day_of_week",
            "temperature",
            "month"
        )

        prediction = model.transform(input_data)

        predicted_value = (
            prediction
            .select("prediction")
            .first()["prediction"]
        )

        st.success(
            f"Predicted Electricity Consumption: "
            f"**{predicted_value:.2f} kWh**"
        )
    st.divider()

    st.subheader("Model Performance")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "RMSE",
            f"{rmse:.4f} kWh"
        )

    with metric2:
        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )

    st.caption(
        "The model uses household, hour, day of week, "
        "temperature, and month as prediction features."
    )
elif page == "About Project":

    st.title("ℹ️ About the Project")

    st.markdown("""
    ## Electricity Consumption Analysis

    This project analyzes electricity consumption data using
    **Apache Spark / PySpark**.

    ### Technologies Used

    - Python
    - Apache Spark
    - PySpark
    - Streamlit
    - Plotly
    - Pandas

    ### Dataset

    The dataset contains:

    - 43,800 records
    - 5 households
    - Hourly observations
    - One year of data
    - Electricity consumption
    - Temperature
    - Hour
    - Day of week

    ### Objectives

    - Analyze electricity consumption patterns
    - Identify peak consumption periods
    - Compare household consumption
    - Analyze daily and monthly trends
    - Detect unusual consumption
    - Build a scalable Big Data analytics pipeline

    ### Big Data Technology

    Apache Spark is used for data processing and aggregation,
    demonstrating how the solution can scale to larger
    electricity-consumption datasets.
    """)


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Electricity Consumption Analysis | BDA Mini Project"
)