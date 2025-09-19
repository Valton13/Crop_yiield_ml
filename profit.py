# revenue_analysis.py
# Farmer Input Cost vs Output Revenue Analysis (Punjab)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("📊 Farmer Input Cost vs Output Revenue Analysis (Punjab)")

# --- Step 1: Load MSP Data ---
try:
    msp_df = pd.read_csv('table-ad7b58ca-215e-4352-b0f2-49760098e987-11.csv')
    print("✅ Loaded MSP data")
except FileNotFoundError:
    raise FileNotFoundError("File 'table-ad7b58ca-215e-4352-b0f2-49760098e987-11.csv' not found.")

# Fix column name if needed
if 'Parameter/Year' in msp_df.columns:
    msp_df.rename(columns={'Parameter/Year': 'Commodity'}, inplace=True)
elif 'Commodity ' in msp_df.columns:
    msp_df.rename(columns={'Commodity ': 'Commodity'}, inplace=True)

if 'Commodity' not in msp_df.columns:
    raise KeyError("No 'Commodity' column found. Check your MSP file.")

msp_df.set_index('Commodity', inplace=True)

# Extract 2023-24 MSP
msp_2023 = {}
for crop in ['Wheat', 'Paddy (Common)', 'Maize', 'Bajra', 'Arhar (Tur)', 'Moong']:
    if crop in msp_df.index and '2023-24' in msp_df.columns:
        val = msp_df.loc[crop, '2023-24']
        msp_2023[crop] = float(val) if pd.notna(val) else 0
    else:
        print(f"⚠️ MSP not found for {crop}")

# Rename for consistency
msp_2023['Rice'] = msp_2023.pop('Paddy (Common)')

# --- Step 2: Load Yield Data (Use 2018 or average) ---
def load_yield(filename, crop_name):
    try:
        df = pd.read_csv(filename)
        df.set_index('District/Year', inplace=True)
        df.replace(['NA', '(d)', '(a)', '.', '0'], pd.NA, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')

        # Use Punjab average or last available year
        if 'Punjab' in df.index:
            yield_kg_per_ha = df.loc['Punjab', '2018']
        else:
            # Take average of all districts for 2018
            if '2018' in df.columns:
                yield_kg_per_ha = df['2018'].mean()
            else:
                # Use most recent year
                recent_years = [col for col in df.columns if col.isdigit()]
                if recent_years:
                    yield_kg_per_ha = df[recent_years[-1]].mean()
                else:
                    yield_kg_per_ha = df.mean().mean()
        return yield_kg_per_ha if pd.notna(yield_kg_per_ha) else 0
    except Exception as e:
        print(f"Error loading {crop_name} yield: {e}")
        return 0

# Load yields
yields = {
    'Wheat': load_yield('Table_4.7_Yield_Wheat_1 (1).csv', 'Wheat'),
    'Rice': load_yield('Table_4.7_Yield_Rice_1.csv', 'Rice'),
    'Maize': load_yield('Table_4.7_Yield_Maize.csv', 'Maize'),
    'Bajra': load_yield('Table_4.7_Yield_Bajra_1.csv', 'Bajra'),
}

# --- Moong & Arhar: Use Production and Assumed Area ---
try:
    moong_prod = pd.read_csv('Table_4.7_Production_Moong_3.csv')
    arhar_prod = pd.read_csv('Table_4.7_Production_Arhar.csv')

    # Use 2018 production (thousand tonnes)
    if '2018' in moong_prod.columns:
        moong_tonnes = moong_prod['2018'].sum() * 1000  # tonnes
    else:
        moong_tonnes = moong_prod.iloc[:, -1].sum() * 1000

    if '2018' in arhar_prod.columns:
        arhar_tonnes = arhar_prod['2018'].sum() * 1000
    else:
        arhar_tonnes = arhar_prod.iloc[:, -1].sum() * 1000

    # Assumed area (hectares)
    moong_area_ha = 10000
    arhar_area_ha = 5000

    # Yield = Production / Area
    yields['Moong'] = (moong_tonnes * 1000) / moong_area_ha  # kg/ha
    yields['Arhar'] = (arhar_tonnes * 1000) / arhar_area_ha
except:
    print("⚠️ Production data not found for Moong/Arhar")
    yields['Moong'] = 600
    yields['Arhar'] = 700

# --- Step 3: Input Costs (₹/acre) ---
input_costs = {
    'Wheat': 9200,
    'Rice': 13200,
    'Maize': 9200,
    'Bajra': 5800,
    'Moong': 5400,
    'Arhar': 6500
}

# Crop names for plotting
crops = list(yields.keys())

# --- Step 4: Calculate Revenue & Net Profit ---
results = []

for crop in crops:
    yield_kg_per_ha = yields[crop]
    msp = msp_2023.get(crop, 0)

    if msp == 0:
        
        continue

    # Convert yield to per acre
    yield_kg_per_acre = yield_kg_per_ha * 0.4047
    yield_quintal_per_acre = yield_kg_per_acre / 100  # 100 kg = 1 quintal

    revenue = yield_quintal_per_acre * msp
    cost = input_costs[crop]
    net_profit = revenue - cost

    results.append({
        'Crop': crop,
        'Yield_kg_per_ha': yield_kg_per_ha,
        'MSP_Rs_per_quintal': msp,
        'Cost_Rs_per_acre': cost,
        'Revenue_Rs_per_acre': revenue,
        'Net_Profit_Rs_per_acre': net_profit
    })

# Create DataFrame
results_df = pd.DataFrame(results)
print("\n📈 Revenue & Profit Summary (per acre):")
print(results_df[['Crop', 'Yield_kg_per_ha', 'MSP_Rs_per_quintal',
                  'Cost_Rs_per_acre', 'Revenue_Rs_per_acre', 'Net_Profit_Rs_per_acre']].round(0))

# Save to CSV
results_df.to_csv('Farmer_Revenue_Analysis_Punjab.csv', index=False)
print(f"\n✅ Results saved to 'Farmer_Revenue_Analysis_Punjab.csv'")

# --- Step 5: Plot Stacked Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 6))

crops_plot = results_df['Crop']
costs = results_df['Cost_Rs_per_acre']
profits = results_df['Net_Profit_Rs_per_acre'].apply(lambda x: max(x, 0))  # Only positive profit

# Stacked bar: cost + profit
bars1 = ax.bar(crops_plot, costs, label='Input Cost', color='indianred')
bars2 = ax.bar(crops_plot, profits, bottom=costs, label='Net Profit', color='seagreen')

# Add value labels
for i, (cost, profit) in enumerate(zip(costs, profits)):
    ax.text(i, cost / 2, f"₹{int(cost)}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    if profit > 1000:
        ax.text(i, cost + profit / 2, f"₹{int(profit)}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)

ax.set_title("🌾 Input Cost vs Net Profit per Acre (Punjab, 2023–24)", fontsize=16, pad=20)
ax.set_ylabel("Amount (₹ per acre)")
ax.set_xlabel("Crop")
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Rotate x-axis labels
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()