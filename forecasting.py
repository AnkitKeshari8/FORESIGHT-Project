import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_arima(data, days=30):
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=days)

    return forecast


def forecast_sarima(data, days=30):
    model = SARIMAX(
        data,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7)
    )

    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=days)

    return forecast


def forecast_all_products(df, days=30):
    results = []

    products = df["Product"].unique()

    for product in products:

        product_data = df[df["Product"] == product].copy()

        product_data["Date"] = pd.to_datetime(product_data["Date"])

        product_data = product_data.sort_values("Date")

        sales = product_data.set_index("Date")["Sales"]

        # ARIMA
        arima_forecast = forecast_arima(sales, days)

        # SARIMA
        sarima_forecast = forecast_sarima(sales, days)

        results.append({
            "Product": product,
            "ARIMA Forecast": round(arima_forecast.sum(), 2),
            "SARIMA Forecast": round(sarima_forecast.sum(), 2)
        })

    return pd.DataFrame(results)
