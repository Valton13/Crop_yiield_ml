# moong_production_forecast.py
# Predict Moong Production for 2023–2025 for Each District

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print("🌱 Moong (Pulses) Production Prediction for 2023–2025 (Punjab, India)")

# --- Step 1: Load Data ---
df = pd.read_csv('Table_4.7_Production_Moong_3.csv')
print(f"Loaded data with {len(df)} districts and {len(df.columns)-1} years of data.")

# Replace non-numeric values like 'NA', '(d)', '(a)', '.' with NaN
df.replace(['NA', '(d)', '(a)', '.', '(b)'], np.nan, inplace=True)

# Set District as index
df.set_index('District/Year', inplace=True)

# Convert all values to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Transpose so years are rows and districts are columns
data_t = df.T  # rows = years, cols = districts
data_t.index = pd.to_numeric(data_t.index, errors='coerce')  # Convert year to int

# --- Step 2: Prepare Forecast Years ---
forecast_years = np.array([2023, 2024, 2025]).reshape(-1, 1)
predictions = {}

# --- Step 3: Predict for Each District ---
for district in data_t.columns:
    series = data_t[district]
    available_years = series.dropna().index.values
    productions = series.dropna().values

    # Need at least 2 data points
    if len(available_years) < 2:
        print(f"⚠️ Not enough data for {district}")
        predictions[district] = [np.nan] * 3
        continue

    # Fit Linear Regression
    X_train = available_years.reshape(-1, 1)
    y_train = productions
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict next 3 years
    future_preds = model.predict(forecast_years)
    future_preds = np.clip(future_preds, 0, None)  # No negative production
    predictions[district] = future_preds

    print(f"✅ Predicted for {district}: {future_preds[0]:.2f}, {future_preds[1]:.2f}, {future_preds[2]:.2f} (thousand tonnes)")

# --- Step 4: Save Predictions ---
pred_df = pd.DataFrame(predictions, index=['2023', '2024', '2025'])
pred_df.to_csv('Moong_Production_Predictions_2023_2025.csv')

print(f"\n📊 Predictions saved to 'Moong_Production_Predictions_2023_2025.csv'")
print("\nTop 10 Districts - 2023 Prediction:")
print(pred_df.loc['2023'].sort_values(ascending=False).head(10).round(2))

# --- Optional: Plot Example ---
import matplotlib.pyplot as plt

sample_district = 'Ferozepur'
if sample_district in pred_df.columns:
    plt.figure(figsize=(10, 5))
    full_series = df.loc[sample_district]
    years_historical = pd.to_numeric(full_series.index, errors='coerce')
    prods_historical = pd.to_numeric(full_series.values, errors='coerce')

    mask = ~np.isnan(prods_historical) & (years_historical <= 2022)
    years_historical = years_historical[mask]
    prods_historical = prods_historical[mask]

    # Plot
    plt.plot(years_historical, prods_historical, 'o-', label='Historical Production', color='green')
    pred_vals = pred_df[sample_district].values.astype(float)
    plt.plot([2023, 2024, 2025], pred_vals, 's--', label='Predicted', color='red')

    plt.title(f"Moong Production Forecast: {sample_district}")
    plt.xlabel("Year")
    plt.ylabel("Production (thousand tonnes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()