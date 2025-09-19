# crop_rotation_planner.py
# Smart Crop Rotation Planner for Punjab Districts

import pandas as pd
import numpy as np

print("🔄 CROP ROTATION PLANNER FOR PUNJAB DISTRICTS\n")

# --- Step 1: Load Soil Data ---
try:
    soil_df = pd.read_csv('Qwen_csv_20250905_bpqico7po.csv')  # or .csv
    print(f"✅ Loaded soil data for {len(soil_df)} districts\n")
except FileNotFoundError:
    raise FileNotFoundError("Soil data file 'Qwen_csv_20250905_bpqico7po.csv' not found.")

# Clean district names
soil_df['District'] = soil_df['District'].str.replace('_', ' ')
soil_df.reset_index(drop=True, inplace=True)

# --- Step 2: Define Crop Types ---
cereals = ['Wheat', 'Rice', 'Maize', 'Bajra', 'Jowar']
pulses = ['Moong', 'Urad', 'Arhar', 'Gram', 'Masur']
oilseeds = ['Mustard', 'Sunflower', 'Sesamum', 'Soyabean']
fiber = ['Cotton']
sugarcane = ['Sugarcane']

# --- Step 3: Rotation Logic Based on Soil & Current Crop ---
def suggest_rotation(district, current_crop='Rice'):
    row = soil_df[soil_df['District'] == district]
    if row.empty:
        return {"Error": "District not found"}

    # Extract soil values
    n_soil = row['Nitrogen_kg_ha'].values[0]
    p_soil = row['Phosphorus_kg_ha'].values[0]
    k_soil = row['Potassium_kg_ha'].values[0]
    zn_soil = row['Zinc_ppm'].values[0]
    oc = row['Organic_Carbon_pct'].values[0]

    # Initialize rotation plan
    rotation = [current_crop]

    # --- Rotation Logic ---
    recommendations = []

    # 1. After Rice (Kharif) → Suggested: Pulse (Rabi)
    if current_crop == 'Rice':
        if oc < 0.5:
            next_crop = 'Moong' if zn_soil >= 0.6 else 'Urad'
            recommendations.append(f"🔁 After Rice, grow {next_crop} to fix nitrogen and improve soil health.")
            recommendations.append("💡 Moong/Urad are short-duration pulses ideal for Rabi after Rice.")
        else:
            next_crop = 'Gram'
            recommendations.append(f"🔁 After Rice, grow Gram (Chickpea) — suits medium-high OC soils.")
        rotation.append(next_crop)
        rotation.append('Wheat')

    # 2. After Wheat (Rabi) → Suggested: Maize/Bajra/Moong (Kharif)
    elif current_crop == 'Wheat':
        if n_soil > 280:
            next_crop = 'Bajra'  # Low N demand
            recommendations.append(f"🔁 After Wheat, grow {next_crop} — low nitrogen requirement, drought-tolerant.")
        else:
            next_crop = 'Moong'
            recommendations.append(f"🔁 After Wheat, grow {next_crop} — improves soil fertility.")
        rotation.append(next_crop)
        rotation.append('Wheat')

    # 3. After Cotton → Suggested: Mustard or Pulse
    elif current_crop in fiber:
        next_crop = 'Mustard'
        recommendations.append(f"🔁 After {current_crop}, grow {next_crop} — breaks pest cycle and improves soil.")
        rotation.append(next_crop)
        rotation.append('Cotton')

    # 4. After Sugarcane → Fallow or Pulse
    elif current_crop in sugarcane:
        next_crop = 'Moong or Fallow'
        recommendations.append(f"🔁 After {current_crop}, grow Moong or keep fallow — sugarcane depletes soil.")
        rotation.append(next_crop)

    # 5. Avoid Continuous Cereals
    if current_crop in cereals:
        if len(rotation) == 1:
            recommendations.append("⚠️ Avoid continuous cereal cropping (e.g., Rice-Wheat). It depletes soil and increases pests.")

    # 6. Zinc Deficiency → Avoid Bajra if severe
    if current_crop == 'Bajra' and zn_soil < 0.4:
        recommendations.append("⚠️ Bajra sensitive to Zn deficiency. Apply ZnSO₄ or rotate with Wheat.")

    return {
        'District': district,
        'Rotation': ' → '.join(rotation),
        'Recommendations': recommendations
    }

# --- Step 4: Generate Rotation Plans for All Districts ---
results = []
districts = soil_df['District'].unique()

print("🔄 SUGGESTED CROP ROTATIONS\n")
for district in sorted(districts):
    plan = suggest_rotation(district, current_crop='Rice')  # Default: Rice-based system
    results.append(plan)

    print(f"📍 {district}")
    print(f"   🔄 Rotation: {plan['Rotation']}")
    for rec in plan['Recommendations']:
        print(f"   {rec}")
    print()

# --- Step 5: Save to CSV ---
rotation_df = pd.DataFrame([
    {
        'District': r['District'],
        'Rotation': r['Rotation'],
        'Notes': ' | '.join(r['Recommendations']) if isinstance(r['Recommendations'], list) else r['Recommendations']
    } for r in results
])
rotation_df.to_csv('Crop_Rotation_Plan_Punjab.csv', index=False)
print(f"✅ All rotation plans saved to 'Crop_Rotation_Plan_Punjab.csv'")