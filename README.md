# FORESIGHT – AI-Powered Demand & Inventory Intelligence Platform

## Overview
FORESIGHT is a data science and analytics project that forecasts product demand
and provides inventory intelligence for businesses.

## Features
- Historical sales analytics
- Machine-learning demand forecasting
- 7 to 60 day forecast period
- Stockout risk detection
- Overstock risk detection
- Recommended inventory levels
- Interactive Streamlit dashboard
- Downloadable inventory analysis

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit

## Dataset Format
Your CSV should contain these columns:

Date, Product, Sales, Inventory

Example:

2026-01-01,Laptop,50,300

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
FORESIGHT_Project/
├── app.py
├── forecasting.py
├── inventory.py
├── requirements.txt
├── README.md
└── data/
    └── sales_data.csv
```

## How It Works
1. Historical sales data is loaded.
2. Date features and lag features are created.
3. A Random Forest model learns demand patterns.
4. Future demand is forecasted recursively.
5. Available inventory is compared with forecast demand.
6. The system identifies stockout or overstock risks.
7. Inventory recommendations are displayed.

## Disclaimer
This project is an educational prototype developed for internship submission.
