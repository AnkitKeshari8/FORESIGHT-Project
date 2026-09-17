import streamlit as st
import pandas as pd
import plotly.express as px

from forecasting import forecast_all_products
from inventory import analyze_inventory


st.set_page_config(
    page_title="FORESIGHT",
    page_icon="📊",
    layout="wide"
)

st.title("📊 FORESIGHT")
st.subheader("AI-Powered Demand & Inventory Intelligence Platform")

st.markdown("""
Analyze historical sales and forecast future demand
using ARIMA and SARIMA time-series models.
""")

# Read CSV
df = pd.read_csv("sales_data.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

st.sidebar.header("Forecast Settings")

days = st.sidebar.slider(
    "Forecast Period (Days)",
    min_value=7,
    max_value=90,
    value=30
)

st.sidebar.write("Forecast period:", days, "days")


# Show dataset
st.header("📋 Sales Data")

st.dataframe(df)


# Forecast
st.header("🔮 Demand Forecast")

if st.button("Generate Forecast"):

    with st.spinner("Running ARIMA and SARIMA models..."):

        forecast = forecast_all_products(df, days)

    st.success("Forecast generated successfully!")

    st.dataframe(forecast)


    # Comparison chart

    chart_data = forecast.melt(
        id_vars="Product",
        value_vars=[
            "ARIMA Forecast",
            "SARIMA Forecast"
        ],
        var_name="Model",
        value_name="Forecast"
    )

    fig = px.bar(
        chart_data,
        x="Product",
        y="Forecast",
        color="Model",
        barmode="group",
        title="ARIMA vs SARIMA Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)
