# crop_recommendation_system.py
# Best Crop Recommendation Based on Soil & District (Punjab)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

print("🌱 Punjab Crop Recommendation System")
print("Using soil data and historical yield to recommend best crop\n")

# --- Step 1: Load Soil Data ---
try:
    soil_df = pd.read_csv('Qwen_csv_20250905_bpqico7po.csv')
    print(f"✅ Loaded soil data for {len(soil_df)} districts")
except FileNotFoundError:
    raise FileNotFoundError("Soil data file 'Qwen_csv_20250905_bpqico7po.txt' not found. Make sure it's in the same folder.")

# Clean district names
soil_df['District'] = soil_df['District'].str.replace('_', ' ')

# --- Step 2: Function to Load Yield Data ---
def load_yield_data(filename, crop_name):
    try:
        df = pd.read_csv(filename)
        df.set_index('District/Year', inplace=True)
        df.replace(['NA', '(d)', '(a)', '.', '(c)'], pd.NA, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Use most recent year with data
        recent_years = [col for col in df.columns if col.isdigit()]
        recent_years = sorted(recent_years, reverse=True)
        
        df[crop_name] = df[recent_years[0]]  # e.g., 2018
        df = df[[crop_name]].reset_index()
        df.rename(columns={'District/Year': 'District'}, inplace=True)
        return df
    except FileNotFoundError:
        print(f"⚠️ File '{filename}' not found. Skipping {crop_name}.")
        return None

# Load all yield datasets
wheat = load_yield_data('Table_4.7_Yield_Wheat_1 (1).csv', 'Wheat')
rice = load_yield_data('Table_4.7_Yield_Rice_1.csv', 'Rice')
bajra = load_yield_data('Table_4.7_Yield_Bajra_1.csv', 'Bajra')
maize = load_yield_data('Table_4.7_Yield_Maize.csv', 'Maize')

# Only keep crops that were successfully loaded
yield_dfs = [df for df in [wheat, rice, bajra, maize] if df is not None]
if not yield_dfs:
    raise Exception("No yield data files found. Please check filenames.")

# Merge all yield data
yield_df = yield_dfs[0]
for df in yield_dfs[1:]:
    yield_df = yield_df.merge(df, on='District', how='outer')

# --- Step 3: Merge with Soil Data ---
merged_df = soil_df.merge(yield_df, on='District', how='inner')
print(f"✅ Merged dataset: {len(merged_df)} districts with soil and yield data")

# Define crops
crop_columns = [col for col in ['Wheat', 'Rice', 'Bajra', 'Maize'] if col in merged_df.columns]

# --- Step 4: Create Target: Best Crop (Highest Yield)
def get_best_crop(row):
    crops = row[crop_columns]
    if crops.isna().all():
        return None
    return crops.idxmax()

merged_df['Best_Crop'] = merged_df.apply(get_best_crop, axis=1)
merged_df.dropna(subset=['Best_Crop'], inplace=True)

print(f"🎯 Final training dataset: {len(merged_df)} districts")

# --- Step 5: Train Crop Recommendation Model ---
# Features (soil parameters)
features = [
    'pH', 'EC_dS_m', 'Organic_Carbon_pct',
    'Nitrogen_kg_ha', 'Phosphorus_kg_ha', 'Potassium_kg_ha', 'Zinc_ppm'
]

X = merged_df[features]
y = merged_df['Best_Crop']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Accuracy: {accuracy:.2f}")

# Save model and encoder
joblib.dump(model, 'crop_recommendation_model.pkl')
joblib.dump(le, 'label_encoder_crop.pkl')
print("💾 Model saved as 'crop_recommendation_model.pkl'")

# --- Step 6: Prediction Function ---
def recommend_crop(pH, ec, oc, n, p, k, zn):
    input_data = np.array([[pH, ec, oc, n, p, k, zn]])
    pred = model.predict(input_data)[0]
    crop = le.inverse_transform([pred])[0]
    proba = model.predict_proba(input_data)[0]
    confidence = max(proba) * 100
    return f"🌱 Recommended Crop: {crop} | Confidence: {confidence:.1f}%"

# --- Step 7: Example Predictions ---
print("\n🔍 Example Recommendations:")
print(recommend_crop(pH=7.9, ec=0.8, oc=0.55, n=280, p=25, k=220, zn=1.0))  # Ludhiana-like
print(recommend_crop(pH=8.3, ec=1.2, oc=0.25, n=160, p=8,  k=100, zn=0.3))  # Mansa-like
print(recommend_crop(pH=6.8, ec=0.5, oc=0.70, n=350, p=40, k=300, zn=1.5))  # Hoshiarpur-like

# --- Step 8: Interactive Mode (Optional) ---
print("\n📋 Enter your soil test values:")
try:
    pH = float(input("pH (e.g., 7.9): "))
    ec = float(input("EC (dS/m) (e.g., 0.8): "))
    oc = float(input("Organic Carbon (%) (e.g., 0.55): "))
    n = float(input("Nitrogen (kg/ha) (e.g., 280): "))
    p = float(input("Phosphorus (kg/ha) (e.g., 25): "))
    k = float(input("Potassium (kg/ha) (e.g., 220): "))
    zn = float(input("Zinc (ppm) (e.g., 1.0): "))
    
    print("\n" + recommend_crop(pH, ec, oc, n, p, k, zn))
except ValueError:
    print("Invalid input. Please enter numbers.")