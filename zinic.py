# zinc_deficiency_alert_system.py
# Zinc Deficiency Detection & Recommendation System for Punjab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("⚠️ ZINC DEFICIENCY ALERT SYSTEM FOR PUNJAB DISTRICTS\n")

# --- Step 1: Load Soil Data ---
try:
    soil_df = pd.read_csv('Qwen_csv_20250905_bpqico7po.csv')  # or .csv
    print(f"✅ Loaded soil data for {len(soil_df)} districts\n")
except FileNotFoundError:
    raise FileNotFoundError("Soil data file 'Qwen_csv_20250905_bpqico7po.csv' not found.")

# Clean district names
soil_df['District'] = soil_df['District'].str.replace('_', ' ')
soil_df.reset_index(drop=True, inplace=True)

# --- Step 2: Define Zinc Thresholds ---
ZINC_CRITICAL_THRESHOLD = 0.6  # ppm (mg/kg) - PAU & ICAR standard
ZINC_SEVERE = 0.4
ZINC_MODERATE = 0.6
ZINC_SUFFICIENT = 0.8

def classify_zinc(zn):
    if zn < ZINC_SEVERE:
        return "Severe Deficiency"
    elif zn < ZINC_MODERATE:
        return "Moderate Deficiency"
    elif zn < ZINC_SUFFICIENT:
        return "Low (Near Deficient)"
    else:
        return "Sufficient"

def get_recommendation(zn, district):
    if zn < ZINC_CRITICAL_THRESHOLD:
        return f"Apply 25 kg Zinc Sulfate/ha (5 bags/acre) once every 3 years. Critical for rice & wheat in {district}."
    else:
        return "No Zn application needed. Soil Zn is sufficient."

# --- Step 3: Add Zinc Status ---
soil_df['Zinc_Status'] = soil_df['Zinc_ppm'].apply(classify_zinc)
soil_df['Zn_Recommendation'] = soil_df.apply(lambda row: get_recommendation(row['Zinc_ppm'], row['District']), axis=1)

# Flag deficient districts
soil_df['Is_Zn_Deficient'] = soil_df['Zinc_ppm'] < ZINC_CRITICAL_THRESHOLD

# --- Step 4: Yield Impact Estimation ---
# Source: PAU studies show 15–25% yield increase after Zn correction
def estimate_yield_gain(crop, zn):
    base_gain = 0
    if zn < 0.4:
        base_gain = 25  # Severe deficiency → high gain
    elif zn < 0.6:
        base_gain = 15  # Moderate → medium gain
    else:
        base_gain = 0   # No gain needed

    # Crop-specific response
    crop_multiplier = {
        'Rice': 1.2,
        'Wheat': 1.1,
        'Maize': 1.0,
        'Bajra': 0.8,
        'Sugarcane': 1.3
    }
    return base_gain

soil_df['Estimated_Yield_Gain_Pct'] = soil_df['Zinc_ppm'].apply(lambda zn: estimate_yield_gain('Wheat', zn))

# --- Step 5: Generate Alerts ---
print("🔴 ZINC DEFICIENCY ALERTS\n")
print(f"{'District':<15} {'Zn (ppm)':<8} {'Status':<18} {'Yield Gain (%)':<15}")
print("-" * 60)

alerts = []

for _, row in soil_df.iterrows():
    zn = row['Zinc_ppm']
    status = row['Zinc_Status']
    gain = row['Estimated_Yield_Gain_Pct']
    
    print(f"{row['District']:<15} {zn:<8.1f} {status:<18} {gain:<15.1f}")
    
    if zn < ZINC_CRITICAL_THRESHOLD:
        alerts.append({
            'District': row['District'],
            'Zn_ppm': zn,
            'Status': status,
            'Recommendation': row['Zn_Recommendation'],
            'Yield_Gain_Pct': gain
        })

# --- Step 6: Summary Statistics ---
total_districts = len(soil_df)
deficient_count = len(alerts)
sufficient_count = total_districts - deficient_count

print(f"\n📊 SUMMARY")
print(f"Total Districts: {total_districts}")
print(f"Zinc Deficient (<0.6 ppm): {deficient_count}")
print(f"Zinc Sufficient: {sufficient_count}")
print(f"Deficiency Rate: {deficient_count / total_districts * 100:.1f}%")

# --- Step 7: Save Alerts to CSV ---
alerts_df = pd.DataFrame(alerts)
alerts_df.to_csv('Zinc_Deficiency_Alerts_Punjab.csv', index=False)
print(f"\n✅ Full alert report saved to 'Zinc_Deficiency_Alerts_Punjab.csv'")

# --- Step 8: Visualization ---
plt.figure(figsize=(12, 6))
sns.barplot(
    data=soil_df,
    x='District',
    y='Zinc_ppm',
    hue='Zinc_Status',
    dodge=False,
    palette={'Sufficient': 'green', 'Low (Near Deficient)': 'orange', 'Moderate Deficiency': 'red', 'Severe Deficiency': 'darkred'}
)
plt.axhline(y=ZINC_CRITICAL_THRESHOLD, color='blue', linestyle='--', label='Critical Threshold (0.6 ppm)')
plt.title('Zinc Status in Punjab Districts')
plt.ylabel('Zinc (ppm)')
plt.xlabel('District')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# --- Step 9: Farmer Advisory Message ---
print(f"\n📢 FARMER ADVISORY")
print("Zinc deficiency is invisible but costly! Even if crops look healthy, low Zn causes:")
print("  • Poor root growth")
print("  • Delayed maturity")
print("  • Lodging in wheat")
print("  • 'Khaira' disease in rice")
print("\n✅ Solution: Apply 25 kg Zinc Sulfate (ZnSO₄) per hectare once every 3 years.")
print("💡 Best time: At sowing, mixed with fertilizer.")