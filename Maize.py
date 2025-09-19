# maize_yield_forecast.py
# Predict Maize Yield for 2023, 2024, 2025 for Each District

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print("🌽 Maize Yield Prediction for 2023–2025 (Punjab, India)")

# --- Step 1: Load Data ---
df = pd.read_csv('Table_4.7_Yield_Maize.csv')
print(f"Loaded data with {len(df)} districts and {len(df.columns)-1} years of data.")

# Set District as index
df.set_index('District/Year', inplace=True)

# Transpose so years become rows and districts become columns
data_t = df.T  # Now: rows = years, columns = districts
data_t.index = pd.to_numeric(data_t.index, errors='coerce')  # Convert year strings to numbers

# Convert all values to numeric, 'NA' → NaN
data_t = data_t.apply(pd.to_numeric, errors='coerce')

# --- Step 2: Prepare Forecast Years ---
forecast_years = np.array([2023, 2024, 2025]).reshape(-1, 1)
predictions = {}

# --- Step 3: Predict for Each District ---
for district in data_t.columns:
    # Get available data (remove NaNs)
    series = data_t[district]
    available_years = series.dropna().index.values.reshape(-1, 1)
    yields = series.dropna().values

    # Need at least 2 points to fit a line
    if len(yields) < 2:
        print(f"⚠️ Not enough data for {district}")
        predictions[district] = [np.nan] * 3
        continue

    # Fit Linear Model
    model = LinearRegression()
    model.fit(available_years, yields)

    # Predict next 3 years
    future_preds = model.predict(forecast_years)
    future_preds = np.clip(future_preds, 0, None)  # No negative yields
    predictions[district] = future_preds

    print(f"✅ Predicted for {district}: {future_preds[0]:.0f}, {future_preds[1]:.0f}, {future_preds[2]:.0f} kg/ha")

# --- Step 4: Save Predictions ---
pred_df = pd.DataFrame(predictions, index=['2023', '2024', '2025'])
pred_df.to_csv('Maize_Yield_Predictions_2023_2025.csv')

print(f"\n📊 Predictions saved to 'Maize_Yield_Predictions_2023_2025.csv'")
print(pred_df.round(0).head(10))  # Show first 10 districts

# Optional: Full historical + predicted plot for one district
import matplotlib.pyplot as plt

sample_district = 'Ludhiana'
if sample_district in pred_df.columns:
    plt.figure(figsize=(10, 5))
    full_series = df.loc[sample_district].apply(pd.to_numeric, errors='coerce')
    years_historical = pd.to_numeric(full_series.index, errors='coerce')
    yields_historical = full_series.values

    # Remove NaNs
    mask = ~np.isnan(yields_historical)
    years_historical = years_historical[mask]
    yields_historical = yields_historical[mask]

    # Plot historical
    plt.plot(years_historical, yields_historical, 'o-', color='blue', label='Historical Yield')

    # Plot predictions
    pred_vals = pred_df[sample_district].values.astype(float)
    plt.plot([2023, 2024, 2025], pred_vals, 's--', color='red', label='Predicted')

    plt.title(f"Maize Yield Trend & Forecast: {sample_district}")
    plt.xlabel("Year")
    plt.ylabel("Yield (kg/ha)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()