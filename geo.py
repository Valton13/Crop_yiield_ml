# crop_suitability_map.py
# District-Level Crop Suitability Map for Punjab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

print("🗺️ DISTRICT-LEVEL CROP SUITABILITY MAP FOR PUNJAB\n")

# --- Step 1: Load Soil Data ---
try:
    soil_df = pd.read_csv('Qwen_csv_20250905_bpqico7po.csv')
    print(f"✅ Loaded soil data for {len(soil_df)} districts")
except FileNotFoundError:
    raise FileNotFoundError("Soil data file 'Qwen_csv_20250905_bpqico7po.csv' not found.")

# Clean district names
soil_df['District'] = soil_df['District'].str.replace('_', ' ')
soil_df.reset_index(drop=True, inplace=True)

# --- Step 2: Define Crop Suitability Criteria (Source: PAU & ICAR) ---
crop_criteria = {
    'Wheat': {
        'pH_low': 6.5, 'pH_high': 8.0,
        'N_demand': 'medium', 'P_demand': 'medium', 'K_demand': 'medium',
        'Zn_critical': 0.6,
        'best_in': 'fertile_loam'
    },
    'Rice': {
        'pH_low': 5.5, 'pH_high': 8.0,
        'N_demand': 'high', 'P_demand': 'high', 'K_demand': 'high',
        'Zn_critical': 0.6,
        'best_in': 'clay_loam'
    },
    'Bajra': {
        'pH_low': 7.5, 'pH_high': 9.0,
        'N_demand': 'low', 'P_demand': 'low', 'K_demand': 'low',
        'Zn_critical': 0.6,
        'best_in': 'sandy_loam'
    },
    'Maize': {
        'pH_low': 6.0, 'pH_high': 7.5,
        'N_demand': 'high', 'P_demand': 'medium', 'K_demand': 'medium',
        'Zn_critical': 0.6,
        'best_in': 'loamy'
    }
}

# Nutrient scoring
def get_nutrient_score(value, demand):
    if demand == 'low':
        threshold = 180
    elif demand == 'medium':
        threshold = 240
    else:  # high
        threshold = 300
    return 1.0 if value >= threshold else value / threshold

# --- Step 3: Score Each District for Each Crop ---
results = []

for _, row in soil_df.iterrows():
    district = row['District']
    ph = row['pH']
    zn = row['Zinc_ppm']
    n = row['Nitrogen_kg_ha']
    p = row['Phosphorus_kg_ha']
    k = row['Potassium_kg_ha']
    texture = row['Texture'].lower()

    scores = {'District': district}

    for crop, criteria in crop_criteria.items():
        score = 0

        # pH match
        if criteria['pH_low'] <= ph <= criteria['pH_high']:
            score += 3
        elif abs(ph - criteria['pH_low']) < 0.5 or abs(ph - criteria['pH_high']) < 0.5:
            score += 2
        else:
            score += 1

        # Nutrients
        score += get_nutrient_score(n, criteria['N_demand']) * 2
        score += get_nutrient_score(p, criteria['P_demand']) * 2
        score += get_nutrient_score(k, criteria['K_demand']) * 2

        # Zinc
        score += 2 if zn >= criteria['Zn_critical'] else 1

        # Texture preference
        if (criteria['best_in'] in texture or
            ('loam' in criteria['best_in'] and 'loam' in texture) or
            ('sandy' in criteria['best_in'] and 'sandy' in texture)):
            score += 2
        else:
            score += 1

        scores[crop] = round(score, 2)

    # Best crop
    best_crop = max(crop_criteria.keys(), key=lambda x: scores[x])
    scores['Best_Crop'] = best_crop

    results.append(scores)

# Create DataFrame
suitability_df = pd.DataFrame(results)
print(f"\n📊 Suitability scores computed for {len(suitability_df)} districts")

# --- Step 4: Save to CSV ---
suitability_df.to_csv('Crop_Suitability_Scores_Punjab.csv', index=False)
print("✅ Results saved to 'Crop_Suitability_Scores_Punjab.csv'")

# --- Step 5: Create a Visual Suitability Map (Bar Chart) ---
plt.figure(figsize=(14, 8))

# Sort by district
suitability_df = suitability_df.sort_values('District')

# Melt for plotting
melted = suitability_df.melt(id_vars='District', value_vars=['Wheat', 'Rice', 'Bajra', 'Maize'],
                             var_name='Crop', value_name='Suitability_Score')

# Define colors for crops
crop_colors = {'Wheat': '#DAA520', 'Rice': '#8FBC8F', 'Bajra': '#CD853F', 'Maize': '#FFD700'}
colors = [crop_colors[crop] for crop in melted['Crop']]

# Bar chart
bars = plt.barh(melted['District'] + ' - ' + melted['Crop'],
                melted['Suitability_Score'], color=colors, edgecolor='gray', alpha=0.8)

# Highlight best crop per district
for i, district in enumerate(suitability_df['District']):
    best_crop = suitability_df[suitability_df['District'] == district]['Best_Crop'].values[0]
    if melted.iloc[i*4]['Crop'] == best_crop:
        bars[i*4].set_edgecolor('red')
        bars[i*4].set_linewidth(2)

plt.title('🌾 District-Level Crop Suitability Map for Punjab\n(Red Border = Recommended Crop)', fontsize=16, pad=20)
plt.xlabel('Suitability Score (Higher = Better Fit)')
plt.ylabel('District - Crop')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()

# Custom legend
legend_elements = [Patch(facecolor=crop_colors[c], label=c) for c in crop_colors]
plt.legend(handles=legend_elements, title="Crops", loc='lower right')

plt.show()

# --- Step 6: Print Summary ---
print("\n✅ RECOMMENDED CROPS BY DISTRICT")
print("-" * 50)
for _, row in suitability_df.iterrows():
    print(f"{row['District']:<18} → 🌾 {row['Best_Crop']}")