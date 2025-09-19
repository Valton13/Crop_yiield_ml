# all_crop_predictions.py
# Unified Crop Yield Prediction for Punjab: Maize, Wheat, Rice, Arhar, Bajra, Moong, Sunflower
# Uses files: Table_4.7_Yield_*.csv, Table_4.7_Production_*.csv, Table_4.7_Area_Sunflower.csv

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

print("🌾 Unified Crop Yield Prediction Model for Punjab")
print("Predicting for: Maize, Wheat, Rice, Arhar, Bajra, Moong, Sunflower\n")

# --- Helper: Load and Clean Data ---
def load_crop_data(filename, crop_name, index_col='District/Year'):
    try:
        df = pd.read_csv(filename)
        df.set_index(index_col, inplace=True)
        # Replace non-numeric values
        df.replace(['NA', '(d)', '(a)', '.', '(c)', '0.8.', '0'], np.nan, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    except FileNotFoundError:
        print(f"⚠️ File not found: {filename}")
        return None

# --- Step 1: Load All Datasets ---
maize_yield = load_crop_data('Table_4.7_Yield_Maize.csv', 'Maize')
wheat_yield = load_crop_data('Table_4.7_Yield_Wheat_1 (1).csv', 'Wheat')
rice_yield = load_crop_data('Table_4.7_Yield_Rice_1.csv', 'Rice')
bajra_yield = load_crop_data('Table_4.7_Yield_Bajra_1.csv', 'Bajra')

moong_prod = load_crop_data('Table_4.7_Production_Moong_3.csv', 'Moong')
arhar_prod = load_crop_data('Table_4.7_Production_Arhar.csv', 'Arhar')

sunflower_area = load_crop_data('Table_4.7_Area_Sunflower.csv', 'Sunflower')

# --- Step 2: Predict Function ---
def predict_yield(df, crop_name, target_years=[2023, 2024, 2025]):
    predictions = {}
    df_t = df.T
    df_t.index = pd.to_numeric(df_t.index, errors='coerce')
    df_t = df_t[df_t.index.notna()]
    df_t.sort_index(inplace=True)

    for district in df_t.columns:
        series = df_t[district]
        available_years = series.dropna().index.values
        values = series.dropna().values

        if len(values) < 2:
            predictions[district] = [np.nan] * 3
            continue

        model = LinearRegression()
        X_train = available_years.reshape(-1, 1)
        y_train = values
        model.fit(X_train, y_train)

        future_preds = model.predict(np.array(target_years).reshape(-1, 1))
        future_preds = np.clip(future_preds, 0, None)
        predictions[district] = future_preds

    return pd.DataFrame(predictions, index=target_years)

# --- Step 3: Generate Predictions ---
print("📊 Predicting yields for 2023, 2024, 2025...\n")

results = {}

if maize_yield is not None:
    maize_pred = predict_yield(maize_yield, 'Maize')
    results['Maize_Yield_kg_per_ha'] = maize_pred
    print("✅ Maize yield predictions done")

if wheat_yield is not None:
    wheat_pred = predict_yield(wheat_yield, 'Wheat')
    results['Wheat_Yield_kg_per_ha'] = wheat_pred
    print("✅ Wheat yield predictions done")

if rice_yield is not None:
    rice_pred = predict_yield(rice_yield, 'Rice')
    results['Rice_Yield_kg_per_ha'] = rice_pred
    print("✅ Rice yield predictions done")

if bajra_yield is not None:
    bajra_pred = predict_yield(bajra_yield, 'Bajra')
    results['Bajra_Yield_kg_per_ha'] = bajra_pred
    print("✅ Bajra yield predictions done")

# --- Special: Moong & Arhar → Need Yield = Production / Area ---
def predict_yield_from_production(prod_df, crop_name):
    # Simulate area with average values (Punjab typical)
    avg_area_mapping = {
        'Moong': 10000,  # hectares
        'Arhar': 5000
    }
    area_ha = avg_area_mapping.get(crop_name, 10000)

    predictions = {}
    df_t = prod_df.T
    df_t.index = pd.to_numeric(df_t.index, errors='coerce')
    df_t = df_t[df_t.index.notna()]
    df_t.sort_index(inplace=True)

    for district in df_t.columns:
        series = df_t[district]
        available_years = series.dropna().index.values
        tonnes = series.dropna().values  # thousand tonnes → kg
        kg_values = (tonnes * 1e6)  # convert to kg
        ha = area_ha  # assume constant area

        if len(kg_values) < 2:
            predictions[district] = [np.nan] * 3
            continue

        yields = kg_values / ha  # kg/ha
        model = LinearRegression()
        X_train = available_years.reshape(-1, 1)
        y_train = yields
        model.fit(X_train, y_train)

        future_preds = model.predict(np.array([2023, 2024, 2025]).reshape(-1, 1))
        future_preds = np.clip(future_preds, 0, None)
        predictions[district] = future_preds

    return pd.DataFrame(predictions, index=[2023, 2024, 2025])

if moong_prod is not None:
    moong_yield_pred = predict_yield_from_production(moong_prod, 'Moong')
    results['Moong_Yield_kg_per_ha'] = moong_yield_pred
    print("✅ Moong yield predictions done (from production)")

if arhar_prod is not None:
    arhar_yield_pred = predict_yield_from_production(arhar_prod, 'Arhar')
    results['Arhar_Yield_kg_per_ha'] = arhar_yield_pred
    print("✅ Arhar yield predictions done (from production)")

# --- Sunflower: Only Area Available ---
if sunflower_area is not None:
    sunflower_area_pred = predict_yield(sunflower_area, 'Sunflower')
    # Convert to hectares (data is in thousand hectares)
    sunflower_area_pred *= 1000  # now in hectares
    results['Sunflower_Area_ha'] = sunflower_area_pred
    print("✅ Sunflower area predictions done (in hectares)")

# --- Step 4: Combine All Results ---
print("\n🧩 Combining all predictions...")

# Create a list of DataFrames with MultiIndex
combined_dfs = []
for key, df in results.items():
    df_stack = df.stack().reset_index()
    df_stack.columns = ['Year', 'District', key]
    combined_dfs.append(df_stack)

# Merge all
final_df = combined_dfs[0]
for df in combined_dfs[1:]:
    final_df = pd.merge(final_df, df, on=['Year', 'District'], how='outer')

# Reorder columns
yield_cols = [c for c in final_df.columns if 'Yield' in c]
area_cols = [c for c in final_df.columns if 'Area' in c]
final_df = final_df[['Year', 'District'] + sorted(yield_cols) + sorted(area_cols)]

# Sort
final_df.sort_values(['District', 'Year'], inplace=True)
final_df.reset_index(drop=True, inplace=True)

# --- Step 5: Save to CSV ---
output_file = 'All_Crop_Predictions_2023_2025.csv'
final_df.to_csv(output_file, index=False)
print(f"\n🎉 All predictions saved to '{output_file}'")
print(f"📁 Shape: {final_df.shape} (rows, columns)")
print("\nTop 20 Predictions:")
print(final_df.head(20))

# --- Optional: Summary by Crop ---
print("\n📈 Average Predicted Yield (2023):")
for col in final_df.columns:
    if 'Yield' in col and '2023' in str(final_df['Year'].values):
        avg_2023 = final_df[final_df['Year'] == 2023][col].mean()
        crop = col.replace('_Yield_kg_per_ha', '')
        print(f"  {crop}: {avg_2023:.0f} kg/ha")