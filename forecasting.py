import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def prepare_features(df):
    """Create date and lag features for demand forecasting."""
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Product", "Date"])

    data["dayofweek"] = data["Date"].dt.dayofweek
    data["month"] = data["Date"].dt.month
    data["day"] = data["Date"].dt.day
    data["lag_1"] = data.groupby("Product")["Sales"].shift(1)
    data["lag_7"] = data.groupby("Product")["Sales"].shift(7)
    data["rolling_mean_7"] = (
        data.groupby("Product")["Sales"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )
    return data.dropna()


def train_model(df):
    """Train a Random Forest demand forecasting model."""
    data = prepare_features(df)

    features = [
        "dayofweek", "month", "day",
        "lag_1", "lag_7", "rolling_mean_7"
    ]

    X = data[features]
    y = data["Sales"]

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=2
    )
    model.fit(X, y)
    return model


def forecast_product(df, product, days=30):
    """
    Forecast future daily demand for one product.
    Forecasting is recursive: each predicted value becomes
    part of the history for the next prediction.
    """
    product_df = df[df["Product"] == product].copy()
    product_df["Date"] = pd.to_datetime(product_df["Date"])
    product_df = product_df.sort_values("Date")

    model = train_model(df)
    history = product_df["Sales"].astype(float).tolist()
    last_date = product_df["Date"].max()

    predictions = []
    dates = []

    for i in range(days):
        future_date = last_date + pd.Timedelta(days=i + 1)

        lag_1 = history[-1]
        lag_7 = history[-7] if len(history) >= 7 else np.mean(history)
        rolling_mean_7 = np.mean(history[-7:])

        X_future = pd.DataFrame([{
            "dayofweek": future_date.dayofweek,
            "month": future_date.month,
            "day": future_date.day,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_mean_7": rolling_mean_7
        }])

        prediction = max(0, float(model.predict(X_future)[0]))
        history.append(prediction)
        predictions.append(round(prediction, 2))
        dates.append(future_date)

    return pd.DataFrame({
        "Date": dates,
        "Predicted_Demand": predictions
    })


def forecast_all_products(df, days=30):
    results = []
    for product in df["Product"].unique():
        forecast = forecast_product(df, product, days)
        forecast["Product"] = product
        results.append(forecast)
    return pd.concat(results, ignore_index=True)
