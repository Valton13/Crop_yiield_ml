# crop_recommendation_engine.py
# Smart Crop Recommendation Engine for Punjab

import pandas as pd
import numpy as np

print("🎯 CROP RECOMMENDATION ENGINE FOR PUNJAB DISTRICTS\n")

# --- Step 1: Load All Data ---

# Soil Data
try:
    soil_df = pd.read_csv('Qwen_csv_20250905_bpqico7po.csv')
    soil_df['District'] = soil_df['District'].str.replace('_', ' ')
    print("✅ Loaded soil data")
except:
    raise FileNotFoundError("Soil data file 'Qwen_csv_20250905_bpqico7po.csv' not found.")

# MSP Data
try:
    msp_df = pd.read_csv('table-ad7b58ca-215e-4352-b0f2-49760098e987-11.csv')
    if 'Parameter/Year' in msp_df.columns:
        msp_df.rename(columns={'Parameter/Year': 'Commodity'}, inplace=True)
    msp_df.set_index('Commodity', inplace=True)
    print("✅ Loaded MSP data")
except:
    raise FileNotFoundError("MSP file 'table-ad7b58ca-215e-4352-b0f2-49760098e987-11.csv' not found.")

# Yield Data
def load_yield(filename, crop_name):
    try:
        df = pd.read_csv(filename)
        df.set_index('District/Year', inplace=True)
        df.replace(['NA', '(d)', '(a)', '.', '0'], pd.NA, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        if 'Punjab' in df.index:
            yield_kg_per_ha = df.loc['Punjab', '2018']
        else:
            recent_years = [col for col in df.columns if col.isdigit()]
            if recent_years:
                yield_kg_per_ha = df[recent_years[-1]].mean()
            else:
                yield_kg_per_ha = df.mean().mean()
        return yield_kg_per_ha if pd.notna(yield_kg_per_ha) else 0
    except:
        return 0

yields = {
    'Wheat': load_yield('Table_4.7_Yield_Wheat_1 (1).csv', 'Wheat'),
    'Rice': load_yield('Table_4.7_Yield_Rice_1.csv', 'Rice'),
    'Maize': load_yield('Table_4.7_Yield_Maize.csv', 'Maize'),
    'Bajra': load_yield('Table_4.7_Yield_Bajra_1.csv', 'Bajra'),
}

# Moong & Arhar from Production
try:
    moong_df = pd.read_csv('Table_4.7_Production_Moong_3.csv')
    arhar_df = pd.read_csv('Table_4.7_Production_Arhar.csv')
    moong_tonnes = moong_df['2018'].sum() * 1000
    arhar_tonnes = arhar_df['2018'].sum() * 1000
    yields['Moong'] = (moong_tonnes * 1000) / 10000  # kg/ha
    yields['Arhar'] = (arhar_tonnes * 1000) / 5000
except:
    yields['Moong'] = 600
    yields['Arhar'] = 700

# Input Costs (₹/acre)
input_costs = {
    'Wheat': 9200,
    'Rice': 13200,
    'Maize': 9200,
    'Bajra': 5800,
    'Moong': 5400,
    'Arhar': 6500
}

# MSP 2023-24
msp_2023 = {
    'Wheat': msp_df.loc['Wheat', '2023-24'],
    'Paddy (Common)': msp_df.loc['Paddy (Common)', '2023-24'],
    'Maize': msp_df.loc['Maize', '2023-24'],
    'Bajra': msp_df.loc['Bajra', '2023-24'],
    'Arhar (Tur)': msp_df.loc['Arhar (Tur)', '2023-24'],
    'Moong': msp_df.loc['Moong', '2023-24']
}
msp_2023['Rice'] = msp_2023.pop('Paddy (Common)')
msp_2023['Arhar'] = msp_2023.pop('Arhar (Tur)')

# --- Step 2: Crop Criteria ---
crop_criteria = {
    'Wheat': {'pH_low': 6.5, 'pH_high': 8.0, 'N_demand': 'medium', 'texture': ['Loam', 'Clay Loam']},
    'Rice': {'pH_low': 5.5, 'pH_high': 8.0, 'N_demand': 'high', 'texture': ['Clay Loam']},
    'Bajra': {'pH_low': 7.5, 'pH_high': 9.0, 'N_demand': 'low', 'texture': ['Sandy Loam', 'Loam']},
    'Maize': {'pH_low': 6.0, 'pH_high': 7.5, 'N_demand': 'high', 'texture': ['Loam']},
    'Moong': {'pH_low': 6.0, 'pH_high': 8.5, 'N_demand': 'low', 'texture': ['Loam', 'Sandy Loam']},
    'Arhar': {'pH_low': 6.5, 'pH_high': 8.0, 'N_demand': 'medium', 'texture': ['Loam', 'Clay Loam']}
}

# --- Step 3: Recommendation Engine ---
def recommend_best_crop(district):
    row = soil_df[soil_df['District'] == district]
    if row.empty:
        return f"❌ {district}: Not found"

    row = row.iloc[0]
    ph = row['pH']
    oc = row['Organic_Carbon_pct']
    zn = row['Zinc_ppm']
    texture = row['Texture']

    scores = {}

    for crop in yields:
        if crop not in crop_criteria:
            continue

        crit = crop_criteria[crop]
        score = 0

        # 1. pH Match
        if crit['pH_low'] <= ph <= crit['pH_high']:
            score += 3
        elif abs(ph - crit['pH_low']) < 0.5 or abs(ph - crit['pH_high']) < 0.5:
            score += 2
        else:
            score += 1

        # 2. Texture Match
        if texture in crit['texture']:
            score += 2
        else:
            score += 1

        # 3. Nitrogen Match
        n_soil = row['Nitrogen_kg_ha']
        if crit['N_demand'] == 'low' and n_soil > 280:
            score += 2  # High N → good for low-demand crops
        elif crit['N_demand'] == 'high' and n_soil < 200:
            score += 1  # Low N → bad for high-demand
        else:
            score += 2

        # 4. Zinc (Critical for Bajra/Maize)
        if crop in ['Bajra', 'Maize'] and zn < 0.6:
            score -= 1

        # 5. Profitability
        msp = msp_2023.get(crop, 0)
        if msp == 0:
            continue

        yield_kg_per_acre = yields[crop] * 0.4047
        revenue = (yield_kg_per_acre / 100) * msp  # quintal/acre × ₹
        cost = input_costs[crop]
        profit = revenue - cost

        # Normalize profit (0–3 scale)
        profit_score = min(3, max(1, profit / 5000))
        score += profit_score

        scores[crop] = score

    if not scores:
        return f"⚠️ {district}: No suitable crop"

    best_crop = max(scores, key=scores.get)
    msp = msp_2023[best_crop]
    yield_kg_per_ha = yields[best_crop]
    cost = input_costs[best_crop]
    revenue = (yield_kg_per_ha * 0.4047 / 100) * msp
    profit = revenue - cost

    return {
        'District': district,
        'Best_Crop': best_crop,
        'Yield_kg_per_ha': int(yield_kg_per_ha),
        'MSP_Rs_per_quintal': int(msp),
        'Profit_Rs_per_acre': int(profit),
        'Reason': get_reason(best_crop, ph, zn, oc, texture)
    }

def get_reason(crop, ph, zn, oc, texture):
    reasons = {
        'Wheat': "Stable MSP, suits neutral pH, high-yield zones",
        'Rice': "High revenue, but high water use — best in clay loam",
        'Bajra': "Drought-tolerant, ideal for alkaline soils (pH > 8)",
        'Maize': "High-yield crop, but needs Zn > 0.6 ppm",
        'Moong': "Fixes nitrogen, high MSP (₹7000/q), low input cost",
        'Arhar': "High MSP, improves soil structure, good after Maize"
    }
    return reasons.get(crop, "Good fit based on soil and profitability")

# --- Step 4: Generate Recommendations ---
results = []
for district in soil_df['District']:
    rec = recommend_best_crop(district)
    if isinstance(rec, dict):
        results.append(rec)

# Create DataFrame
results_df = pd.DataFrame(results)
print(results_df[['District', 'Best_Crop', 'Yield_kg_per_ha', 'MSP_Rs_per_quintal', 'Profit_Rs_per_acre']].to_string(index=False))

# Save to CSV
results_df.to_csv('Best_Crop_Recommendations_Punjab.csv', index=False)
print(f"\n✅ All recommendations saved to 'Best_Crop_Recommendations_Punjab.csv'")