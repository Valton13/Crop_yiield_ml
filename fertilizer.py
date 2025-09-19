# fertilizer_recommendation_bags.py
# Fertilizer recommendation in bags per acre (DAP, Urea, Zinc Sulfate)

import pandas as pd
import numpy as np

# --- Step 1: Load Soil Data ---
try:
    soil_df = pd.read_csv('Qwen_csv_20250905_bpqico7po.csv')  # or .csv
    print(f"✅ Loaded soil data for {len(soil_df)} districts\n")
except FileNotFoundError:
    raise FileNotFoundError('Qwen_csv_20250905_bpqico7po.csv')

# Clean district names
soil_df['District'] = soil_df['District'].str.replace('_', ' ')
soil_df.reset_index(drop=True, inplace=True)

# --- Step 2: Base Fertilizer Doses (kg/ha) - Source: PAU Guidelines ---
base_fert = {
    'Wheat': {'N': 120, 'P': 60, 'K': 40},
    'Rice': {'N': 120, 'P': 60, 'K': 40},
    'Maize': {'N': 150, 'P': 60, 'K': 40},
    'Bajra': {'N': 60, 'P': 30, 'K': 20},
    'Moong': {'N': 20, 'P': 40, 'K': 20},
    'Arhar': {'N': 20, 'P': 60, 'K': 40}
}

# Thresholds for soil nutrient levels
n_thresholds = (200, 280)   # Low <200, Medium <280, High >=280 (kg/ha)
p_thresholds = (20, 40)
k_thresholds = (150, 250)
zn_threshold = 0.6  # ppm

# --- Step 3: Fertilizer Recommendation Function with Bags per Acre ---
def recommend_fertilizer_bags(district, crop):
    row = soil_df[soil_df['District'] == district]
    if row.empty:
        return None

    # Extract soil values
    n_soil = row['Nitrogen_kg_ha'].values[0]
    p_soil = row['Phosphorus_kg_ha'].values[0]
    k_soil = row['Potassium_kg_ha'].values[0]
    zn_soil = row['Zinc_ppm'].values[0]
    oc = row['Organic_Carbon_pct'].values[0]

    if crop not in base_fert:
        return None

    base = base_fert[crop]

    # Helper: Adjust dose based on soil
    def adjust(nutrient, soil_val, low, med, name):
        if soil_val < low:
            return nutrient * 1.0, f"{name} is low ({soil_val:.0f}) → Apply full dose"
        elif soil_val < med:
            return nutrient * 0.75, f"{name} is medium ({soil_val:.0f}) → Apply 75% dose"
        else:
            return nutrient * 0.5, f"{name} is high ({soil_val:.0f}) → Apply 50% dose"

    n_rec_kg_ha, n_exp = adjust(base['N'], n_soil, n_thresholds[0], n_thresholds[1], "Nitrogen")
    p_rec_kg_ha, p_exp = adjust(base['P'], p_soil, p_thresholds[0], p_thresholds[1], "Phosphorus")
    k_rec_kg_ha, k_exp = adjust(base['K'], k_soil, k_thresholds[0], k_thresholds[1], "Potassium")

    # Zinc recommendation
    if zn_soil < zn_threshold:
        zn_rec_kg_ha = 5.0
        zn_exp = f"Zinc is deficient ({zn_soil:.1f} ppm < 0.6) → Apply 5 kg/ha Zinc Sulfate"
    else:
        zn_rec_kg_ha = 0.0
        zn_exp = f"Zinc is sufficient ({zn_soil:.1f} ppm) → No Zn needed"

    # Convert kg/ha → kg/acre → bags/acre
    def kg_ha_to_bags_acre(kg_ha):
        kg_acre = kg_ha * 0.4047
        bags_acre = kg_acre / 50
        return round(bags_acre, 2)

    # DAP provides both N and P
    # 1 bag DAP ≈ 18.5 kg N + 26 kg P₂O₅ ≈ 18.5 kg N + 11.8 kg P (elemental)
    # We assume 1 bag DAP = 11.8 kg P and 18.5 kg N

    p_from_dap_per_bag = 11.8  # kg P per bag of DAP
    n_from_dap_per_bag = 18.5  # kg N per bag of DAP

    # How many bags of DAP needed to meet P requirement?
    dap_bags_acre = kg_ha_to_bags_acre(p_rec_kg_ha) * (60 / 11.8)  # Adjusted for P content
    dap_bags_acre = round(dap_bags_acre, 2)

    # N supplied by DAP
    n_from_dap = dap_bags_acre * 18.5 / 0.4047  # convert back to kg/ha

    # Remaining N to be supplied by Urea
    urea_n_kg_ha = max(0, n_rec_kg_ha - n_from_dap)
    urea_bags_acre = kg_ha_to_bags_acre(urea_n_kg_ha * (100 / 46))  # Urea is 46% N
    urea_bags_acre = round(urea_bags_acre, 2)

    # K: Use MOP (Muriate of Potash), 1 bag = 60 kg K₂O ≈ 50 kg K
    k_bags_acre = kg_ha_to_bags_acre(k_rec_kg_ha * (100 / 50))
    k_bags_acre = round(k_bags_acre, 2)

    # Zinc Sulfate: 1 bag = 50 kg
    zn_bags_acre = kg_ha_to_bags_acre(zn_rec_kg_ha)
    zn_bags_acre = round(zn_bags_acre, 2)

    return {
        'District': district,
        'Crop': crop,
        'DAP_bags_per_acre': dap_bags_acre,
        'Urea_bags_per_acre': urea_bags_acre,
        'MOP_bags_per_acre': k_bags_acre,
        'Zinc_Sulfate_bags_per_acre': zn_bags_acre,
        'Soil_OC_%': round(oc, 2),
        'N_Explanation': n_exp,
        'P_Explanation': p_exp,
        'K_Explanation': k_exp,
        'Zn_Explanation': zn_exp
    }

# --- Step 4: Generate Recommendations for All Districts & Crops ---
results = []
crops = ['Wheat', 'Rice', 'Maize', 'Bajra', 'Moong', 'Arhar']

print("🌾 FERTILIZER RECOMMENDATIONS IN BAGS PER ACRE\n")
print(f"{'Dist':<12} {'Crop':<8} {'DAP':<6} {'Urea':<6} {'MOP':<6} {'Zn':<6}")
print("-" * 50)

for district in sorted(soil_df['District'].unique()):
    for crop in crops:
        rec = recommend_fertilizer_bags(district, crop)
        if rec:
            results.append(rec)
            print(f"{rec['District'][:10]:<12} {rec['Crop']:<8} "
                  f"{rec['DAP_bags_per_acre']:<6} {rec['Urea_bags_per_acre']:<6} "
                  f"{rec['MOP_bags_per_acre']:<6} {rec['Zinc_Sulfate_bags_per_acre']:<6}")

            # Print explanations for Wheat only
            if crop == 'Wheat':
                print(f"  📌 DAP: {rec['P_Explanation']}")
                print(f"  📌 Urea: {rec['N_Explanation']}")
                print(f"  📌 MOP: {rec['K_Explanation']}")
                print(f"  📌 Zn: {rec['Zn_Explanation']}")
                print()

# --- Step 5: Save to CSV ---
results_df = pd.DataFrame(results)
results_df.to_csv('Fertilizer_Recommendations_Bags_Per_Acre.csv', index=False)
print(f"✅ All recommendations saved to 'Fertilizer_Recommendations_Bags_Per_Acre.csv'")
print(f"📊 Total: {len(results)} recommendations generated")