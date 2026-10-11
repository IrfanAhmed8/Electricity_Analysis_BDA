import plotly.express as px


# ============================================================
# HOURLY CONSUMPTION CHART
# ============================================================

def hourly_consumption_chart(hourly_df):

    fig = px.line(
        hourly_df,
        x="hour",
        y="average_consumption",
        markers=True,
        title="Average Electricity Consumption by Hour",
        labels={
            "hour": "Hour of Day",
            "average_consumption": "Average Consumption (kWh)"
        }
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
            dtick=1
        ),
        hovermode="x unified"
    )

    return fig


# ============================================================
# MONTHLY CONSUMPTION CHART
# ============================================================

def monthly_consumption_chart(monthly_df):

    fig = px.bar(
        monthly_df,
        x="month_name",
        y="total_consumption",
        title="Monthly Electricity Consumption",
        labels={
            "month_name": "Month",
            "total_consumption": "Total Consumption (kWh)"
        },
        text_auto=".0f"
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Consumption (kWh)"
    )

    return fig


# ============================================================
# HOUSEHOLD CONSUMPTION CHART
# ============================================================

def household_consumption_chart(household_df):

    fig = px.bar(
        household_df,
        x="household_id",
        y="total_consumption",
        title="Total Consumption by Household",
        labels={
            "household_id": "Household",
            "total_consumption": "Total Consumption (kWh)"
        },
        text_auto=".0f"
    )

    return fig


# ============================================================
# DAILY CONSUMPTION CHART
# ============================================================

def daily_consumption_chart(daily_df):

    fig = px.line(
        daily_df,
        x="date",
        y="total_consumption",
        title="Daily Electricity Consumption",
        labels={
            "date": "Date",
            "total_consumption": "Total Consumption (kWh)"
        }
    )

    fig.update_layout(
        hovermode="x unified"
    )

    return fig


# ============================================================
# DAY OF WEEK CHART
# ============================================================

def weekday_consumption_chart(weekday_df):

    day_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }

    weekday_df["day_name"] = weekday_df["day_of_week"].map(day_names)

    fig = px.bar(
        weekday_df,
        x="day_name",
        y="average_consumption",
        title="Average Consumption by Day of Week",
        labels={
            "day_name": "Day",
            "average_consumption": "Average Consumption (kWh)"
        },
        text_auto=".2f"
    )

    return fig


# ============================================================
# TEMPERATURE VS CONSUMPTION
# ============================================================

def temperature_consumption_chart(df):

    fig = px.scatter(
        df,
        x="temperature",
        y="consumption_kwh",
        color="household_id",
        title="Temperature vs Electricity Consumption",
        labels={
            "temperature": "Temperature (°C)",
            "consumption_kwh": "Consumption (kWh)"
        },
        opacity=0.6
    )

    return fig


# ============================================================
# HOUSEHOLD × HOUR HEATMAP
# ============================================================

def household_hour_heatmap(df):

    pivot_df = df.pivot_table(
        values="consumption_kwh",
        index="household_id",
        columns="hour",
        aggfunc="mean"
    )

    fig = px.imshow(
        pivot_df,
        labels={
            "x": "Hour",
            "y": "Household",
            "color": "Average Consumption (kWh)"
        },
        title="Household × Hour Consumption Heatmap",
        aspect="auto"
    )

    return fig