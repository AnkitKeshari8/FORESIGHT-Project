import pandas as pd


def analyze_inventory(df, forecast_df):
    """
    Compare available inventory with predicted demand and classify risk.
    """
    current_inventory = (
        df.sort_values("Date")
        .groupby("Product", as_index=False)["Inventory"]
        .last()
    )

    predicted_demand = (
        forecast_df.groupby("Product", as_index=False)["Predicted_Demand"]
        .sum()
        .rename(columns={"Predicted_Demand": "Forecast_Demand"})
    )

    result = current_inventory.merge(predicted_demand, on="Product")
    result["Recommended_Stock"] = (result["Forecast_Demand"] * 1.15).round()
    result["Stock_Difference"] = (
        result["Inventory"] - result["Recommended_Stock"]
    ).round()

    def risk(row):
        if row["Inventory"] < row["Forecast_Demand"] * 0.80:
            return "HIGH STOCKOUT RISK"
        elif row["Inventory"] < row["Recommended_Stock"]:
            return "STOCKOUT WARNING"
        elif row["Inventory"] > row["Recommended_Stock"] * 1.50:
            return "OVERSTOCK RISK"
        return "OPTIMAL"

    result["Status"] = result.apply(risk, axis=1)

    def recommendation(row):
        if "STOCKOUT" in row["Status"]:
            units = max(0, row["Recommended_Stock"] - row["Inventory"])
            return f"Increase inventory by {int(round(units))} units"
        elif row["Status"] == "OVERSTOCK RISK":
            units = max(0, row["Inventory"] - row["Recommended_Stock"])
            return f"Reduce inventory by approximately {int(round(units))} units"
        return "Maintain current inventory level"

    result["Recommendation"] = result.apply(recommendation, axis=1)

    return result.sort_values("Product")
