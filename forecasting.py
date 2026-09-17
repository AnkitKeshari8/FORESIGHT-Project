import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def prepare_series(df, product):
    """Prepare daily sales time series for a product."""

    data = df[df["Product"].astype(str) == str(product)].copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")

    data = data.dropna(subset=["Date", "Sales"])
    data = data.sort_values("Date")

    # Combine duplicate dates
    data = (
        data.groupby("Date")["Sales"]
        .sum()
        .sort_index()
    )

    # Fill missing dates
    data = data.asfreq("D")

    # Fill missing sales values
    data = data.interpolate()
    data = data.bfill().ffill()

    return data


def arima_forecast(series, days):
    """Generate ARIMA forecast."""

    model = ARIMA(
        series,
        order=(1, 1, 1)
    )

    fitted_model = model.fit()

    forecast = fitted_model.forecast(
        steps=days
    )

    forecast = np.maximum(
        forecast,
        0
    )

    return forecast


def sarima_forecast(series, days):
    """Generate SARIMA forecast with weekly seasonality."""

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(
        disp=False
    )

    forecast = fitted_model.forecast(
        steps=days
    )

    forecast = np.maximum(
        forecast,
        0
    )

    return forecast


def forecast_product(df, product, days=30):
    """
    Generate ARIMA and SARIMA forecasts
    for one product.
    """

    series = prepare_series(
        df,
        product
    )

    if len(series) < 14:
        raise ValueError(
            f"{product} does not have enough "
            "historical data. Please provide "
            "at least 14 days of sales data."
        )

    arima_values = arima_forecast(
        series,
        days
    )

    sarima_values = sarima_forecast(
        series,
        days
    )

    last_date = series.index.max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days,
        freq="D"
    )

    result = pd.DataFrame({
        "Date": future_dates,
        "Product": product,
        "ARIMA_Prediction": np.round(
            arima_values,
            2
        ),
        "SARIMA_Prediction": np.round(
            sarima_values,
            2
        )
    })

    return result


def forecast_all_products(df, days=30):
    """
    Generate ARIMA and SARIMA forecasts
    for all products.
    """

    results = []

    products = df["Product"].dropna().unique()

    for product in products:

        try:

            result = forecast_product(
                df,
                product,
                days
            )

            results.append(result)

        except Exception as e:

            print(
                f"Forecast failed for "
                f"{product}: {e}"
            )

    if not results:
        raise ValueError(
            "Unable to generate forecasts. "
            "Check your dataset."
        )

    return pd.concat(
        results,
        ignore_index=True
    )
