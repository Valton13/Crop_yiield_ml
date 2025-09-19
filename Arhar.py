# arhar_yield_forecast.py
# Predict Arhar Production & Estimated Yield for 2023–2025

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

print("🫘 Arhar (Pigeon Pea) Production & Yield Forecast (Punjab, India)")

# --- Step 1: Load Data ---
df = pd.read_csv('Table_4.7_Production_Arhar.csv')
print(f"Loaded data: {len(df)} districts, years {df.columns[1]} to {df.columns[-1]}")

# Clean non-numeric values and set index
df.replace(['NA', '(d)', '(a)', '.', '(b)', '0.8.', '0'], np.nan, inplace=True)
df.set_index('District/Year', inplace=True)

# Convert to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Transpose: years as rows, districts as columns
data_t = df.T
data_t.index = pd.to_numeric(data_t.index, errors='coerce')  # Convert year to int

# Only keep valid years
data_t = data_t[data_t.index.notna()]
data_t.sort_index(inplace=True)

# --- Step 2: Predict Future Production ---
forecast_years = np.array([2023, 2024, 2025]).reshape(-1, 1)
production_predictions = {}

for district in data_t.columns:
    series = data_t[district]
    available_years = series.dropna().index.values
    productions = series.dropna().values  # in thousand tonnes

    if len(available_years) < 2:
        print(f"⚠️ Not enough data for {district}")
        production_predictions[district] = [np.nan] * 3
        continue

    # Fit Linear Model
    model = LinearRegression()
    X_train = available_years.reshape(-1, 1)
    y_train = productions
    model.fit(X_train, y_train)

    # Predict next 3 years
    future_prod = model.predict(forecast_years)
    future_prod = np.clip(future_prod, 0, None)  # No negative production
    production_predictions[district] = future_prod

    print(f"✅ Predicted production for {district}: {future_prod[0]:.2f}, {future_prod[1]:.2f}, {future_prod[2]:.2f} KT")

# Save production predictions
prod_df = pd.DataFrame(production_predictions, index=['2023', '2024', '2025'])
prod_df.to_csv('Arhar_Production_Predictions_2023_2025.csv')

# --- Step 3: Estimate Yield Using Historical Average Yield ---
# We'll assume average historical yield (kg/ha) remains roughly constant
# (This is a simplification — real yield needs area data)

# Simulate historical yield per district using a reference: assume average yield in Punjab is ~700–900 kg/ha
# But better: estimate from known data points if available

# Let's use a **proxy**: assume average yield = total_production / total_area
# Since we don't have area, we’ll **reverse-engineer average yield** using typical area estimates from govt reports

# 🔍 Assumption: For districts with recent production, assume area ≈ 5,000 to 20,000 hectares based on crop patterns
# We'll estimate **average historical yield** per district using last 5 years of data and typical area

estimated_yield_predictions = {}

for district in data_t.columns:
    recent_years = data_t.index[(data_t.index >= 2015) & (data_t.index <= 2022)]
    recent_productions = data_t.loc[recent_years, district].dropna()

    if len(recent_productions) == 0:
        estimated_yield_predictions[district] = [np.nan] * 3
        continue

    avg_production_tonnes = recent_productions.mean() * 1000  # convert KT to tonnes

    # Estimate average area (hectares) based on typical Arhar cultivation in Punjab
    # Source: Punjab Agriculture Dept — Arhar is grown on ~5,000–15,000 ha in major districts
    if district in ['Ludhiana', 'Sangrur', 'Bathinda', 'Ferozepur']:
        estimated_area_ha = avg_production_tonnes / 800 * 1000  # assume ~800 kg/ha yield
    else:
        estimated_area_ha = avg_production_tonnes / 700 * 1000  # assume lower yield

    # Clip area to realistic range
    estimated_area_ha = np.clip(estimated_area_ha, 1000, 50000)

    # Now estimate yield: production / area
    avg_historical_yield_kg_per_ha = (avg_production_tonnes * 1000) / estimated_area_ha  # in kg/ha

    # Assume yield improves slightly over time (~1% per year)
    base_yield = avg_historical_yield_kg_per_ha
    yield_2023 = base_yield * 1.01
    yield_2024 = yield_2023 * 1.01
    yield_2025 = yield_2024 * 1.01

    estimated_yield_predictions[district] = [yield_2023, yield_2024, yield_2025]

    print(f"🌾 Estimated yield for {district}: {yield_2023:.0f}, {yield_2024:.0f}, {yield_2025:.0f} kg/ha")

# Save yield predictions
yield_df = pd.DataFrame(estimated_yield_predictions, index=['2023', '2024', '2025'])
yield_df.to_csv('Arhar_Estimated_Yield_Predictions_2023_2025.csv')

print(f"\n📊 Results Saved:")
print("→ Arhar_Production_Predictions_2023_2025.csv")
print("→ Arhar_Estimated_Yield_Predictions_2023_2025.csv")

# Show top 10 districts by predicted 2023 production
print("\n🔝 Top 10 Districts by Predicted Arhar Production (2023):")
top_prod = prod_df.loc['2023'].dropna().sort_values(ascending=False).head(10)
print(top_prod.round(2))

# Optional: Plot example
import matplotlib.pyplot as plt

sample_district = 'Ludhiana'
if sample_district in yield_df.columns:
    plt.figure(figsize=(10, 5))
    full_series = df.loc[sample_district]
    years = pd.to_numeric(full_series.index, errors='coerce')
    prods = pd.to_numeric(full_series.values, errors='coerce')
    mask = ~np.isnan(prods) & (years <= 2022)
    years = years[mask]
    prods = prods[mask]

    plt.plot(years, prods, 'o-', label='Historical Production (KT)', color='orange')
    pred_vals = prod_df[sample_district].values.astype(float)
    plt.plot([2023, 2024, 2025], pred_vals, 's--', color='red', label='Predicted')

    plt.title(f"Arhar Production Forecast: {sample_district}")
    plt.xlabel("Year")
    plt.ylabel("Production (thousand tonnes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()