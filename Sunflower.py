# sunflower_area_forecast.py
# Predict Area under Sunflower Cultivation (2023–2025)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print("🌻 Sunflower Area Forecast (Punjab, India)")

# --- Step 1: Load Data ---
df = pd.read_csv('Table_4.7_Area_Sunflower.csv')
print(f"Loaded data with {len(df)} districts and {len(df.columns)-1} years of area data.")

# Clean and set index
df.replace(['NA', '(c)', '(a)', '(b)', '0.8.'], np.nan, inplace=True)
df.set_index('District/Year', inplace=True)

# Convert to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Transpose: years as rows, districts as columns
data_t = df.T
data_t.index = pd.to_numeric(data_t.index, errors='coerce')  # Year as int
data_t = data_t[data_t.index.notna()]
data_t.sort_index(inplace=True)

# --- Step 2: Forecast Years ---
forecast_years = np.array([2023, 2024, 2025]).reshape(-1, 1)
area_predictions = {}

# --- Step 3: Predict Area for Each District ---
for district in data_t.columns:
    series = data_t[district]
    available_years = series.dropna().index.values
    areas = series.dropna().values  # in thousand hectares

    if len(available_years) < 2:
        print(f"⚠️ Not enough data for {district}")
        area_predictions[district] = [np.nan] * 3
        continue

    # Fit Linear Model
    model = LinearRegression()
    X_train = available_years.reshape(-1, 1)
    y_train = areas
    model.fit(X_train, y_train)

    # Predict next 3 years
    future_area = model.predict(forecast_years)
    future_area = np.clip(future_area, 0, None)  # No negative area
    area_predictions[district] = future_area

    print(f"✅ Predicted area for {district}: {future_area[0]:.2f}, {future_area[1]:.2f}, {future_area[2]:.2f} K ha")

# --- Step 4: Save Predictions ---
pred_df = pd.DataFrame(area_predictions, index=['2023', '2024', '2025'])
pred_df.to_csv('Sunflower_Area_Predictions_2023_2025.csv')

print(f"\n📊 Predictions saved to 'Sunflower_Area_Predictions_2023_2025.csv'")
print("\nTop 5 Districts by Predicted Area in 2023:")
print(pred_df.loc['2023'].dropna().sort_values(ascending=False).head(5).round(2))

# --- Optional: Plot Example ---
sample_district = 'Hoshiarpur'
if sample_district in pred_df.columns:
    plt.figure(figsize=(10, 5))
    full_series = df.loc[sample_district]
    years = pd.to_numeric(full_series.index, errors='coerce')
    areas = pd.to_numeric(full_series.values, errors='coerce')
    mask = ~np.isnan(areas) & (years <= 2022)
    years = years[mask]
    areas = areas[mask]

    plt.plot(years, areas, 'o-', label='Historical Area (K ha)', color='orange')
    pred_vals = pred_df[sample_district].values.astype(float)
    plt.plot([2023, 2024, 2025], pred_vals, 's--', color='red', label='Predicted')

    plt.title(f"Sunflower Cultivation Area Forecast: {sample_district}")
    plt.xlabel("Year")
    plt.ylabel("Area (thousand hectares)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()