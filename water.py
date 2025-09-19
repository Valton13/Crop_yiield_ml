# climate_risk_score.py
# Climate Risk Score for Punjab Districts

import pandas as pd
import numpy as np

print("🌍 CLIMATE RISK SCORE FOR PUNJAB DISTRICTS\n")

# --- Step 1: Load Soil Data ---
try:
    soil_df = pd.read_csv('Qwen_csv_20250905_bpqico7po.csv')
    soil_df['District'] = soil_df['District'].str.replace('_', ' ')
    print(f"✅ Loaded soil data for {len(soil_df)} districts")
except FileNotFoundError:
    raise FileNotFoundError("Soil data file 'Qwen_csv_20250905_bpqico7po.csv' not found.")

# --- Step 2: Define Climate Risk Factors ---

# 1. Rainfall Risk (Assumed from district type)
# Punjab rainfall: 400–700 mm (Kharif) → Low rainfall = high risk for Rice
rainfall_risk = {
    'Amritsar': 3, 'Bathinda': 5, 'Firozpur': 4, 'Faridkot': 5,
    'Ludhiana': 3, 'Moga': 3, 'Sangrur': 4, 'Patiala': 3,
    'Jalandhar': 3, 'Hoshiarpur': 2, 'Kapurthala': 2, 'Mansa': 6,
    'Barnala': 5, 'Tarn Taran': 3, 'SBS Nagar': 3, 'Rupnagar': 2,
    'Pathankot': 1, 'Gurdaspur': 1, 'Fazilka': 6, 'Sri Muktsar Sahib': 5,
    'Fatehgarh Sahib': 4
}
# Scale: 1 = Low risk (good rain), 6 = High risk (arid)

# 2. Groundwater Risk (CGWB Data: Over-exploited = High Risk)
# Source: Central Ground Water Board (Punjab: 109% of annual recharge used)
gw_risk = {
    'Amritsar': 5, 'Bathinda': 8, 'Firozpur': 6, 'Faridkot': 7,
    'Ludhiana': 7, 'Moga': 8, 'Sangrur': 9, 'Patiala': 7,
    'Jalandhar': 8, 'Hoshiarpur': 4, 'Kapurthala': 5, 'Mansa': 9,
    'Barnala': 8, 'Tarn Taran': 6, 'SBS Nagar': 5, 'Rupnagar': 4,
    'Pathankot': 2, 'Gurdaspur': 2, 'Fazilka': 8, 'Sri Muktsar Sahib': 7,
    'Fatehgarh Sahib': 6
}
# Scale: 1–10 (10 = critically over-exploited)

# 3. Temperature Risk (Heat Stress)
# Western Punjab faces higher temps
temp_risk = {
    'Mansa': 8, 'Bathinda': 8, 'Fazilka': 9, 'Faridkot': 7,
    'Sangrur': 7, 'Moga': 6, 'Ludhiana': 6, 'Firozpur': 7,
    'Amritsar': 5, 'Patiala': 5, 'Barnala': 7, 'Sri Muktsar Sahib': 8,
    'Others': 4
}

# 4. Crop Risk (Rice-Wheat = High Climate Risk)
# Rice = high water, high emissions
crop_risk = {
    'Rice': 8, 'Wheat': 5, 'Bajra': 2, 'Maize': 4, 'Moong': 1, 'Arhar': 2
}

# --- Step 3: Climate Risk Score Calculator ---
def calculate_climate_risk(district, current_crop='Rice'):
    # Get base risks
    rr = rainfall_risk.get(district, 4)
    gr = gw_risk.get(district, 5)
    tr = temp_risk.get(district, temp_risk['Others'])
    cr = crop_risk.get(current_crop, 5)

    # Normalize (0–10)
    score = (rr + gr + tr + cr) / 4.0
    score = round(score, 1)

    # Adjust based on soil
    row = soil_df[soil_df['District'] == district]
    if not row.empty:
        oc = row['Organic_Carbon_pct'].values[0]
        zn = row['Zinc_ppm'].values[0]
        if oc < 0.5:
            score += 0.5  # Low OC → less resilient
        if zn < 0.6:
            score += 0.3  # Zn deficiency → weak plants

    return min(10.0, round(score, 1))

# --- Step 4: Generate Risk Scores ---
results = []
for district in soil_df['District']:
    risk = calculate_climate_risk(district, current_crop='Rice')  # Default: Rice
    recommendation = "Switch to Bajra/Moong" if risk > 7 else "Continue with rotation" if risk > 5 else "Low risk — sustainable farming"
    
    results.append({
        'District': district,
        'Rainfall_Risk': rainfall_risk.get(district, 4),
        'GW_Risk': gw_risk.get(district, 5),
        'Temp_Risk': temp_risk.get(district, 4),
        'Crop_Risk': crop_risk['Rice'],
        'Climate_Risk_Score': risk,
        'Recommendation': recommendation
    })

# Create DataFrame
results_df = pd.DataFrame(results)
print(results_df[['District', 'Climate_Risk_Score', 'Recommendation']].to_string(index=False))

# Save to CSV
results_df.to_csv('Climate_Risk_Score_Punjab.csv', index=False)
print(f"\n✅ All risk scores saved to 'Climate_Risk_Score_Punjab.csv'")

# --- Step 5: Summary ---
high_risk = results_df[results_df['Climate_Risk_Score'] >= 7]
moderate_risk = results_df[(results_df['Climate_Risk_Score'] >= 5) & (results_df['Climate_Risk_Score'] < 7)]
low_risk = results_df[results_df['Climate_Risk_Score'] < 5]

print(f"\n📊 SUMMARY")
print(f"High Risk (≥7): {len(high_risk)} districts")
print(f"Moderate Risk (5–6.9): {len(moderate_risk)} districts")
print(f"Low Risk (<5): {len(low_risk)} districts")

print(f"\n⚠️ HIGH RISK DISTRICTS (Score ≥7):")
for _, row in high_risk.iterrows():
    print(f"  {row['District']}: {row['Climate_Risk_Score']}/10 → {row['Recommendation']}")