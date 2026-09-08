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

st.markdown(
    """
    Analyze historical sales, forecast future demand using Machine Learning,
    and identify stockout and overstock risks.
    """
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Sales Dataset (CSV)",
    type=["csv"]
)

forecast_days = st.sidebar.slider(
    "Forecast Period (Days)",
    min_value=7,
    max_value=60,
    value=30
)

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    required_columns = {"Date", "Product", "Sales", "Inventory"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "CSV must contain: Date, Product, Sales, Inventory"
        )

    data["Date"] = pd.to_datetime(data["Date"])
    return data


if uploaded_file is None:
    st.info("Using the included sample dataset. Upload your own CSV from the sidebar to replace it.")
    df = pd.read_csv("data/sales_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
else:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(str(e))
        st.stop()


# ---------------- Summary metrics ----------------
latest_inventory = df.sort_values("Date").groupby("Product")["Inventory"].last()

total_sales = int(df["Sales"].sum())
avg_daily_sales = round(df["Sales"].mean(), 1)
total_inventory = int(latest_inventory.sum())
products_count = df["Product"].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Historical Sales", f"{total_sales:,}")
col2.metric("Average Daily Demand", avg_daily_sales)
col3.metric("Current Inventory", f"{total_inventory:,}")
col4.metric("Products Analyzed", products_count)

st.divider()

# ---------------- Sales analytics ----------------
st.header("📈 Historical Sales Analytics")

selected_product = st.selectbox(
    "Select Product",
    sorted(df["Product"].unique())
)

product_history = df[df["Product"] == selected_product].sort_values("Date")

fig_history = px.line(
    product_history,
    x="Date",
    y="Sales",
    title=f"Historical Daily Sales – {selected_product}",
    markers=True
)
st.plotly_chart(fig_history, use_container_width=True)

# ---------------- Forecasting ----------------
st.header("🤖 AI Demand Forecast")

with st.spinner("Training machine learning model and generating forecasts..."):
    forecast_df = forecast_all_products(df, forecast_days)

selected_forecast = forecast_df[
    forecast_df["Product"] == selected_product
]

fig_forecast = px.line(
    selected_forecast,
    x="Date",
    y="Predicted_Demand",
    title=f"Predicted Demand for Next {forecast_days} Days – {selected_product}",
    markers=True
)
st.plotly_chart(fig_forecast, use_container_width=True)

forecast_summary = (
    forecast_df.groupby("Product", as_index=False)["Predicted_Demand"]
    .sum()
)
forecast_summary["Predicted_Demand"] = forecast_summary[
    "Predicted_Demand"
].round()

st.subheader(f"{forecast_days}-Day Demand Forecast")
st.dataframe(
    forecast_summary.rename(
        columns={"Predicted_Demand": "Forecasted Demand (Units)"}
    ),
    use_container_width=True,
    hide_index=True
)

# ---------------- Inventory intelligence ----------------
st.divider()
st.header("📦 Inventory Intelligence & Risk Detection")

inventory_result = analyze_inventory(df, forecast_df)

status_counts = inventory_result["Status"].value_counts()
stockout_count = int(
    status_counts.get("HIGH STOCKOUT RISK", 0)
    + status_counts.get("STOCKOUT WARNING", 0)
)
overstock_count = int(status_counts.get("OVERSTOCK RISK", 0))
optimal_count = int(status_counts.get("OPTIMAL", 0))

a, b, c = st.columns(3)
a.metric("Stockout Alerts", stockout_count)
b.metric("Overstock Alerts", overstock_count)
c.metric("Optimal Products", optimal_count)

st.dataframe(
    inventory_result,
    use_container_width=True,
    hide_index=True
)

# ---------------- Recommendations ----------------
st.header("💡 AI Inventory Recommendations")

for _, row in inventory_result.iterrows():
    if row["Status"] == "OPTIMAL":
        st.success(f"**{row['Product']}** — {row['Recommendation']}")
    elif "STOCKOUT" in row["Status"]:
        st.warning(
            f"⚠️ **{row['Product']} — {row['Status']}**: "
            f"{row['Recommendation']}"
        )
    else:
        st.error(
            f"📦 **{row['Product']} — {row['Status']}**: "
            f"{row['Recommendation']}"
        )

# ---------------- Downloads ----------------
st.divider()
st.header("⬇️ Download Results")

csv = inventory_result.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Inventory Analysis CSV",
    csv,
    "foresight_inventory_analysis.csv",
    "text/csv"
)

st.caption(
    "FORESIGHT uses historical sales patterns, lag features, "
    "and a Random Forest machine-learning model for demand forecasting."
)
