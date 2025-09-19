# sowing_calendar.py
# AI-Powered Sowing Calendar with Multiple Crop Selection (Now includes Rice)

import requests
import pandas as pd
import numpy as np
from datetime import datetime

print("📅 AI-POWERED SOWING CALENDAR (Multiple Crops + RICE)\n")

# 🔑 Your WeatherAPI Key
API_KEY = "08d2c6272dff488181e160916250609"  # Replace with your key from https://www.weatherapi.com/my/

# --- Step 1: Load Soil Data ---
try:
    soil_df = pd.read_csv('Qwen_csv_20250905_bpqico7po.csv')
    soil_df['District'] = soil_df['District'].str.replace('_', ' ')
    districts = soil_df['District'].tolist()
    print(f"✅ Loaded soil data for {len(districts)} districts")
except:
    raise FileNotFoundError("Soil data file 'Qwen_csv_20250905_bpqico7po.csv' not found.")

# --- Step 2: Crop Sowing Guidelines (PAU Rules) ---
crop_sowing_rules = {
    'Wheat': {
        'ideal_month': 'November',
        'ideal_dates': 'Nov 10–30',
        'min_temp': 20,
        'max_temp': 35,
        'avoid_rain': True,
        'soil_ph_low': 6.5,
        'soil_ph_high': 8.0,
        'n_demand': 'medium'
    },
    'Rice': {
        'ideal_month': 'June',
        'ideal_dates': 'Jun 10–25',
        'min_temp': 25,
        'max_temp': 40,
        'avoid_rain': False,  # Rice needs water, but heavy rain can damage transplanting
        'soil_ph_low': 5.5,
        'soil_ph_high': 8.0,
        'n_demand': 'high'
    },
    'Bajra': {
        'ideal_month': 'June',
        'ideal_dates': 'Jun 15–30',
        'min_temp': 30,
        'max_temp': 45,
        'avoid_rain': False,
        'soil_ph_low': 7.5,
        'soil_ph_high': 9.0,
        'n_demand': 'low'
    },
    'Maize': {
        'ideal_month': 'June',
        'ideal_dates': 'Jun 5–20',
        'min_temp': 25,
        'max_temp': 40,
        'avoid_rain': True,
        'soil_ph_low': 6.0,
        'soil_ph_high': 7.5,
        'n_demand': 'high'
    },
    'Moong': {
        'ideal_month': 'June',
        'ideal_dates': 'Jun 10–25',
        'min_temp': 25,
        'max_temp': 45,
        'avoid_rain': False,
        'soil_ph_low': 6.0,
        'soil_ph_high': 8.5,
        'n_demand': 'low'
    },
    'Arhar': {
        'ideal_month': 'June',
        'ideal_dates': 'Jun 15–30',
        'min_temp': 28,
        'max_temp': 40,
        'avoid_rain': True,
        'soil_ph_low': 6.5,
        'soil_ph_high': 8.0,
        'n_demand': 'medium'
    }
}

# --- Step 3: Map Districts to Cities for WeatherAPI ---
district_to_city = {
    'Amritsar': 'Amritsar, India',
    'Bathinda': 'Bathinda, India',
    'Ludhiana': 'Ludhiana, India',
    'Jalandhar': 'Jalandhar, India',
    'Patiala': 'Patiala, India',
    'Mansa': 'Mansa, India',
    'Fazilka': 'Fazilka, India',
    'Hoshiarpur': 'Hoshiarpur, India',
    'Sangrur': 'Sangrur, India',
    'Moga': 'Moga, India',
    'Firozpur': 'Firozpur, India',
    'Gurdaspur': 'Gurdaspur, India',
    'Rupnagar': 'Rupnagar, India',
    'Kapurthala': 'Kapurthala, India',
    'Barnala': 'Barnala, India'
}

# --- Step 4: Fetch 7-Day Weather Forecast ---
def get_forecast(city):
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        'key': API_KEY,
        'q': city,
        'days': 7,
        'aqi': 'no'
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()['forecast']['forecastday']
        else:
            error = response.json().get('error', {}).get('message', 'Unknown error')
            print(f"❌ {city}: {response.status_code} — {error}")
            return None
    except Exception as e:
        print(f"⚠️ {city}: {e}")
        return None

# --- Step 5: Generate Sowing Advice for Multiple Crops ---
def get_sowing_advice(district, crops):
    row = soil_df[soil_df['District'] == district]
    if row.empty:
        return [{ 'Crop': crop, 'Advice': f"{district}: Not found" } for crop in crops]
    row = row.iloc[0]

    city = district_to_city.get(district)
    if not city:
        return [{ 'Crop': crop, 'Advice': f"{district}: City not mapped" } for crop in crops]

    forecast = get_forecast(city)
    if not forecast:
        return [{ 'Crop': crop, 'Advice': f"{district}: No weather data" } for crop in crops]

    current_month = datetime.now().strftime('%B')
    avg_temp = sum(day['day']['avgtemp_c'] for day in forecast[:3]) / 3
    total_rain = sum(day['day']['totalprecip_mm'] for day in forecast[:3])

    advice_list = []

    for crop in crops:
        rules = crop_sowing_rules.get(crop)
        if not rules:
            advice_list.append({
                'Crop': crop,
                'Sowing_Window': 'N/A',
                'Avg_Temp_3D_C': avg_temp,
                'Current_Rain_3D_mm': total_rain,
                'Advice': f"{crop}: Rules not defined"
            })
            continue

        # Check conditions
        ph_match = rules['soil_ph_low'] <= row['pH'] <= rules['soil_ph_high']
        temp_match = rules['min_temp'] <= avg_temp <= rules['max_temp']
        rain_risk = rules['avoid_rain'] and total_rain > 10
        in_month = rules['ideal_month'] == current_month

        advice = []

        if not in_month:
            advice.append(f"📅 Not ideal month")
        if not ph_match:
            advice.append(f"⚠️ pH {row['pH']} not ideal")
        if not temp_match:
            advice.append(f"🌡️ Temp ({avg_temp:.1f}°C) outside range")
        if rain_risk:
            advice.append(f"🌧️ Rain ({total_rain:.1f}mm)")

        if not advice:
            advice.append("✅ Ideal to sow now")
        else:
            advice.append("⏳ Delay sowing")

        advice_list.append({
            'Crop': crop,
            'Sowing_Window': rules['ideal_dates'],
            'Avg_Temp_3D_C': round(avg_temp, 1),
            'Current_Rain_3D_mm': round(total_rain, 1),
            'Advice': " | ".join(advice)
        })

    return advice_list

# --- Step 6: Generate Calendar for All Districts ---
all_results = []

for district in districts:
    if district not in district_to_city:
        continue

    # Include Rice in comparison
    crops_to_compare = ['Wheat', 'Rice', 'Bajra', 'Moong']  # Rice added!
    advice_for_district = get_sowing_advice(district, crops_to_compare)

    for advice in advice_for_district:
        all_results.append({
            'District': district,
            **advice
        })

# --- Step 7: Display & Save ---
results_df = pd.DataFrame(all_results)
print("🌾 AI-POWERED SOWING CALENDAR (With Rice)")
print("=" * 100)
for _, row in results_df.iterrows():
    print(f"{row['District']:15} | {row['Crop']:8} | {row['Sowing_Window']:12} | {row['Current_Rain_3D_mm']:6}mm | {row['Advice']}")

# Save to CSV
results_df.to_csv('AI_Sowing_Calendar_Punjab_MultiCrop_with_Rice.csv', index=False)
print(f"\n✅ Data saved to 'AI_Sowing_Calendar_Punjab_MultiCrop_with_Rice.csv'")