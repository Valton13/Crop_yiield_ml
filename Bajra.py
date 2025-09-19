# bajra_yield_forecast.py
# Predict Bajra (Pearl Millet) Yield (kg/ha) for 2019–2021 for Each District in Punjab

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print("🌾 Bajra (Pearl Millet) Yield Prediction for 2019–2021 (Punjab, India)")

# --- Step 1: Load Data ---
df = pd.read_csv('Table_4.7_Yield_Bajra_1.csv')
print(f"Loaded data with {len(df)} districts and {len(df.columns)-1} years of yield data.")

# Set District as index
df.set_index('District/Year', inplace=True)

# Replace non-numeric values like 'NA' with NaN
df.replace(['NA', '(c)', '(a)', '(b)', '.'], np.nan, inplace=True)

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Transpose: years become rows, districts become columns
data_t = df.T
data_t.index = pd.to_numeric(data_t.index, errors='coerce')  # Convert year strings to numbers
data_t = data_t.sort_index()  # Sort by year

# --- Step 2: Forecast Years ---
forecast_years = np.array([2019, 2020, 2021]).reshape(-1, 1)
predictions = {}

# --- Step 3: Predict for Each District ---
for district in data_t.columns:
    series = data_t[district]
    available_years = series.dropna().index.values
    yields = series.dropna().values

    if len(yields) < 2:
        print(f"⚠️ Not enough data for {district}")
        predictions[district] = [np.nan] * 3
        continue

    # Fit Linear Regression Model
    X_train = available_years.reshape(-1, 1)
    y_train = yields
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict next 3 years
    future_preds = model.predict(forecast_years)
    future_preds = np.clip(future_preds, 0, None)  # No negative yield
    predictions[district] = future_preds

    print(f"✅ Predicted for {district}: {future_preds[0]:.0f}, {future_preds[1]:.0f}, {future_preds[2]:.0f} kg/ha")

# --- Step 4: Save Predictions ---
pred_df = pd.DataFrame(predictions, index=['2019', '2020', '2021'])
pred_df.to_csv('Bajra_Yield_Predictions_2019_2021.csv')

print(f"\n📊 Predictions saved to 'Bajra_Yield_Predictions_2019_2021.csv'")
print("\nTop 10 Districts - Predicted 2019 Yield:")
print(pred_df.loc['2019'].sort_values(ascending=False).dropna().head(10).round(0))

# --- Optional: Plot Example ---
sample_district = 'Bathinda'
if sample_district in pred_df.columns:
    plt.figure(figsize=(10, 5))
    full_series = df.loc[sample_district]
    years_historical = pd.to_numeric(full_series.index, errors='coerce')
    yields_historical = pd.to_numeric(full_series.values, errors='coerce')

    # Remove NaNs
    mask = ~np.isnan(yields_historical)
    years_historical = years_historical[mask]
    yields_historical = yields_historical[mask]

    # Plot historical
    plt.plot(years_historical, yields_historical, 'o-', color='blue', label='Historical Yield')

    # Plot predictions
    pred_vals = pred_df[sample_district].values.astype(float)
    plt.plot([2019, 2020, 2021], pred_vals, 's--', color='red', label='Predicted')

    plt.title(f"Bajra Yield Trend & Forecast: {sample_district}")
    plt.xlabel("Year")
    plt.ylabel("Yield (kg/ha)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()