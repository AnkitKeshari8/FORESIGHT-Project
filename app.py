import streamlit as st
import pandas as pd
import plotly.express as px

from forecasting import forecast_all_products


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FORESIGHT",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("📊 FORESIGHT")

st.subheader(
    "AI-Powered Demand & Inventory Intelligence Platform"
)

st.markdown(
    """
    **FORESIGHT** analyzes historical sales and forecasts
    future demand using **ARIMA and SARIMA** time-series
    models.

    Users can upload their own CSV dataset and generate
    demand forecasts and inventory insights.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Settings")


uploaded_file = st.sidebar.file_uploader(
    "📁 Upload your own CSV dataset",
    type=["csv"]
)


forecast_days = st.sidebar.slider(
    "🔮 Forecast Period",
    min_value=7,
    max_value=90,
    value=30
)


# =========================================================
# LOAD DATA
# =========================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        st.sidebar.success(
            "✅ Your dataset has been uploaded!"
        )

    except Exception as e:

        st.error(
            f"Unable to read your CSV: {e}"
        )

        st.stop()

else:

    try:

        df = pd.read_csv(
            "sales_data.csv"
        )

        st.sidebar.info(
            "📄 Using sample dataset"
        )

    except FileNotFoundError:

        st.error(
            "Sample dataset not found. "
            "Please upload your own CSV."
        )

        st.stop()


# =========================================================
# DATA VALIDATION
# =========================================================

required_columns = [
    "Date",
    "Product",
    "Sales"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "❌ Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.info(
        """
        Your CSV should contain:

        Date | Product | Sales

        Example:

        2026-01-01 | Laptop | 50
        """
    )

    st.stop()


# =========================================================
# DATA CLEANING
# =========================================================

try:

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

except Exception:

    st.error(
        "❌ Could not convert the Date column."
    )

    st.stop()


df["Sales"] = pd.to_numeric(
    df["Sales"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "Date",
        "Product",
        "Sales"
    ]
)


df = df.sort_values(
    "Date"
)


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.header("📋 Dataset Overview")


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total Records",
    f"{len(df):,}"
)


col2.metric(
    "Products",
    df["Product"].nunique()
)


col3.metric(
    "Total Sales",
    f"{df['Sales'].sum():,.0f}"
)


col4.metric(
    "Average Sales",
    f"{df['Sales'].mean():,.2f}"
)


with st.expander(
    "👀 View Dataset"
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PRODUCT SELECTION
# =========================================================

st.header("📈 Sales Analysis")


products = sorted(
    df["Product"].astype(str).unique()
)


selected_product = st.selectbox(
    "Select Product",
    products
)


product_data = df[
    df["Product"].astype(str)
    == selected_product
].copy()


product_data = (
    product_data
    .groupby("Date", as_index=False)["Sales"]
    .sum()
)


# =========================================================
# HISTORICAL SALES GRAPH
# =========================================================

fig_history = px.line(
    product_data,
    x="Date",
    y="Sales",
    markers=True,
    title=(
        f"Historical Sales - "
        f"{selected_product}"
    )
)


fig_history.update_layout(
    hovermode="x unified"
)


st.plotly_chart(
    fig_history,
    use_container_width=True
)


# =========================================================
# FORECAST BUTTON
# =========================================================

st.divider()

st.header(
    "🔮 ARIMA & SARIMA Demand Forecast"
)


st.write(
    f"Forecasting the next "
    f"**{forecast_days} days**."
)


if st.button(
    "🚀 Generate Forecast",
    type="primary"
):

    with st.spinner(
        "Running ARIMA and SARIMA models..."
    ):

        try:

            forecast_df = forecast_all_products(
                df,
                forecast_days
            )

        except Exception as e:

            st.error(
                f"Forecasting error: {e}"
            )

            st.stop()


    st.success(
        "✅ Forecast generated successfully!"
    )


    # =====================================================
    # SELECTED PRODUCT FORECAST
    # =====================================================

    selected_forecast = forecast_df[
        forecast_df["Product"].astype(str)
        == selected_product
    ].copy()


    if selected_forecast.empty:

        st.warning(
            "No forecast was generated for "
            "the selected product."
        )

    else:

        st.subheader(
            f"🔮 Forecast - {selected_product}"
        )


        chart_data = selected_forecast[
            [
                "Date",
                "ARIMA_Prediction",
                "SARIMA_Prediction"
            ]
        ].melt(
            id_vars="Date",
            var_name="Model",
            value_name="Predicted Demand"
        )


        chart_data["Model"] = (
            chart_data["Model"]
            .replace(
                {
                    "ARIMA_Prediction": "ARIMA",
                    "SARIMA_Prediction": "SARIMA"
                }
            )
        )


        fig_forecast = px.line(
            chart_data,
            x="Date",
            y="Predicted Demand",
            color="Model",
            markers=True,
            title=(
                f"ARIMA vs SARIMA "
                f"- {selected_product}"
            )
        )


        fig_forecast.update_layout(
            hovermode="x unified"
        )


        st.plotly_chart(
            fig_forecast,
            use_container_width=True
        )


        # =================================================
        # FORECAST TOTALS
        # =================================================

        arima_total = (
            selected_forecast[
                "ARIMA_Prediction"
            ].sum()
        )


        sarima_total = (
            selected_forecast[
                "SARIMA_Prediction"
            ].sum()
        )


        col1, col2 = st.columns(2)


        col1.metric(
            "ARIMA Forecast",
            f"{arima_total:,.0f} units"
        )


        col2.metric(
            "SARIMA Forecast",
            f"{sarima_total:,.0f} units"
        )


        # =================================================
        # FORECAST TABLE
        # =================================================

        st.subheader(
            "📊 Detailed Forecast"
        )


        st.dataframe(
            selected_forecast,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # DOWNLOAD FORECAST
        # =================================================

        forecast_csv = (
            selected_forecast
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            "⬇️ Download Forecast CSV",
            forecast_csv,
            "foresight_forecast.csv",
            "text/csv"
        )


    # =====================================================
    # ALL PRODUCT FORECAST
    # =====================================================

    st.divider()

    st.subheader(
        "📊 Forecast Summary - All Products"
    )


    summary = (
        forecast_df
        .groupby("Product")
        .agg(
            ARIMA_Forecast=(
                "ARIMA_Prediction",
                "sum"
            ),
            SARIMA_Forecast=(
                "SARIMA_Prediction",
                "sum"
            )
        )
        .reset_index()
    )


    summary["ARIMA_Forecast"] = (
        summary["ARIMA_Forecast"]
        .round(0)
    )


    summary["SARIMA_Forecast"] = (
        summary["SARIMA_Forecast"]
        .round(0)
    )


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # MODEL COMPARISON BAR CHART
    # =====================================================

    comparison = summary.melt(
        id_vars="Product",
        value_vars=[
            "ARIMA_Forecast",
            "SARIMA_Forecast"
        ],
        var_name="Model",
        value_name="Forecast"
    )


    comparison["Model"] = (
        comparison["Model"]
        .replace(
            {
                "ARIMA_Forecast": "ARIMA",
                "SARIMA_Forecast": "SARIMA"
            }
        )
    )


    fig_comparison = px.bar(
        comparison,
        x="Product",
        y="Forecast",
        color="Model",
        barmode="group",
        title="ARIMA vs SARIMA Forecast Comparison"
    )


    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )


    # =====================================================
    # INVENTORY ANALYSIS
    # =====================================================

    st.divider()

    st.header(
        "📦 Inventory Intelligence"
    )


    if "Inventory" not in df.columns:

        st.info(
            """
            ℹ️ Inventory analysis is unavailable
            because your dataset does not contain an
            **Inventory** column.

            Add an Inventory column to your CSV to
            enable stockout and overstock analysis.
            """
        )

    else:

        latest_inventory = (
            df.sort_values("Date")
            .groupby("Product")
            .last()
            .reset_index()
        )


        inventory_table = latest_inventory[
            [
                "Product",
                "Inventory"
            ]
        ].copy()


        forecast_totals = (
            forecast_df
            .groupby("Product")
            .agg(
                SARIMA_Demand=(
                    "SARIMA_Prediction",
                    "sum"
                )
            )
            .reset_index()
        )


        inventory_table = inventory_table.merge(
            forecast_totals,
            on="Product"
        )


        inventory_table[
            "Recommended Stock"
        ] = (
            inventory_table[
                "SARIMA_Demand"
            ] * 1.15
        ).round(0)


        def calculate_status(row):

            inventory = row["Inventory"]
            demand = row["SARIMA_Demand"]
            recommended = row["Recommended Stock"]


            if inventory < demand * 0.80:

                return "🔴 HIGH STOCKOUT RISK"


            elif inventory < recommended:

                return "🟠 STOCKOUT WARNING"


            elif inventory > recommended * 1.50:

                return "🔵 OVERSTOCK RISK"


            else:

                return "🟢 OPTIMAL"


        inventory_table[
            "Status"
        ] = inventory_table.apply(
            calculate_status,
            axis=1
        )


        def make_recommendation(row):

            inventory = row["Inventory"]
            recommended = row["Recommended Stock"]


            if "STOCKOUT" in row["Status"]:

                amount = max(
                    0,
                    recommended - inventory
                )


                return (
                    f"Increase inventory by "
                    f"{int(amount)} units"
                )


            elif "OVERSTOCK" in row["Status"]:

                amount = max(
                    0,
                    inventory - recommended
                )


                return (
                    f"Reduce inventory by "
                    f"{int(amount)} units"
                )


            return (
                "Maintain current inventory level"
            )


        inventory_table[
            "Recommendation"
        ] = inventory_table.apply(
            make_recommendation,
            axis=1
        )


        st.dataframe(
            inventory_table,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # RECOMMENDATIONS
        # =================================================

        st.subheader(
            "💡 Inventory Recommendations"
        )


        for _, row in inventory_table.iterrows():

            product = row["Product"]
            status = row["Status"]
            recommendation = row[
                "Recommendation"
            ]


            if "HIGH STOCKOUT" in status:

                st.error(
                    f"🔴 **{product}** — "
                    f"{recommendation}"
                )


            elif "STOCKOUT WARNING" in status:

                st.warning(
                    f"🟠 **{product}** — "
                    f"{recommendation}"
                )


            elif "OVERSTOCK" in status:

                st.info(
                    f"🔵 **{product}** — "
                    f"{recommendation}"
                )


            else:

                st.success(
                    f"🟢 **{product}** — "
                    f"{recommendation}"
                )


        # =================================================
        # DOWNLOAD INVENTORY
        # =================================================

        inventory_csv = (
            inventory_table
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            "⬇️ Download Inventory Analysis",
            inventory_csv,
            "foresight_inventory_analysis.csv",
            "text/csv"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "FORESIGHT | Demand Intelligence | "
    "ARIMA + SARIMA"
)
