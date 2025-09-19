# msp_price_forecast.py
# Predict Crop MSP for 2025-26, 2026-27, 2027-28

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print("💰 Crop MSP Forecast (2025–28) | Using data from table-ad7b58ca-215e-4352-b0f2-49760098e987-11.csv")

# --- Step 1: Load Data ---
df = pd.read_csv('table-ad7b58ca-215e-4352-b0f2-49760098e987-11.csv')
print(f"Loaded data for {len(df)} crops and {len(df.columns)-1} years.")

# Extract year part (e.g., '2015-16' → 2015)
year_cols = [col for col in df.columns if col != 'Commodity']
years = np.array([int(col.split('-')[0]) for col in year_cols]).reshape(-1, 1)

# Prepare predictions
forecast_years = np.array([2025, 2026, 2027]).reshape(-1, 1)
predictions = {}

# --- Step 2: Predict for Each Crop ---
for _, row in df.iterrows():
    crop = row['Commodity']
    prices = row[year_cols].dropna().values.astype(float)
    available_years = years[:len(prices)]

    if len(prices) < 2:
        print(f"⚠️ Not enough data for {crop}")
        predictions[crop] = [np.nan] * 3
        continue

    # Fit Linear Regression
    model = LinearRegression()
    model.fit(available_years, prices)

    # Predict next 3 years
    future_preds = model.predict(forecast_years)
    future_preds = np.clip(future_preds, 0, None)  # No negative prices
    predictions[crop] = future_preds

    print(f"✅ Predicted for {crop}: ₹{future_preds[0]:.0f}, ₹{future_preds[1]:.0f}, ₹{future_preds[2]:.0f}")

# --- Step 3: Save Predictions ---
pred_df = pd.DataFrame(predictions, index=['2025-26', '2026-27', '2027-28'])
pred_df.to_csv('Crop_MSP_Predictions_2025_2028.csv')

print(f"\n📊 Predictions saved to 'Crop_MSP_Predictions_2025_2028.csv'")
print("\nPredicted MSP (₹/quintal):")
print(pred_df.round(0))

# --- Optional: Plot Example ---
sample_crop = 'Wheat'
if sample_crop in df['Commodity'].values:
    plt.figure(figsize=(10, 5))
    row = df[df['Commodity'] == sample_crop].iloc[0]
    prices = row[year_cols].astype(float)
    historical_years = [int(y.split('-')[0]) for y in year_cols if pd.notna(row[y])]
    historical_prices = [row[y] for y in year_cols if pd.notna(row[y])]

    plt.plot(historical_years, historical_prices, 'o-', label='Historical MSP', color='green')

    pred_vals = pred_df[sample_crop].values.astype(float)
    plt.plot([2025, 2026, 2027], pred_vals, 's--', color='red', label='Predicted')

    plt.title(f"MSP Forecast: {sample_crop}")
    plt.xlabel("Year")
    plt.ylabel("MSP (₹ per quintal)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()