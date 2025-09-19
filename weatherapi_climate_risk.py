# weatherapi_climate_risk.py
# Real-Time Climate Risk with WeatherAPI Integration

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

print("🌦️ REAL-TIME CLIMATE RISK WITH WEATHERAPI\n")

# --- Step 1: Load Soil Data ---
try:
    soil_df = pd.read_csv('Qwen_csv_20250905_bpqico7po.csv')
    soil_df['District'] = soil_df['District'].str.replace('_', ' ')
    print(f"✅ Loaded soil data for {len(soil_df)} districts")
except FileNotFoundError:
    raise FileNotFoundError("Soil data file 'Qwen_csv_20250905_bpqico7po.csv' not found.")

# --- Step 2: Simulated Historical Rainfall (Kharif Season Avg in mm) ---
historical_rainfall = {
    'Amritsar': 780, 'Bathinda': 620, 'Firozpur': 750, 'Faridkot': 600,
    'Ludhiana': 720, 'Moga': 760, 'Sangrur': 680, 'Patiala': 700,
    'Jalandhar': 740, 'Hoshiarpur': 920, 'Kapurthala': 880, 'Mansa': 580,
    'Barnala': 610, 'Tarn Taran': 770, 'SBS Nagar': 730, 'Rupnagar': 850,
    'Pathankot': 950, 'Gurdaspur': 900, 'Fazilka': 550,
    'Sri Muktsar Sahib': 600, 'Fatehgarh Sahib': 710
}

# --- Step 3: Map Districts to Major Cities (for WeatherAPI) ---
# WeatherAPI uses city names — we'll map districts to nearest city
district_to_city = {
    'Amritsar': 'Amritsar',
    'Bathinda': 'Bathinda',
    'Ludhiana': 'Ludhiana',
    'Jalandhar': 'Jalandhar',
    'Patiala': 'Patiala',
    'Mansa': 'Mansa',
    'Fazilka': 'Fazilka',
    'Hoshiarpur': 'Hoshiarpur',
    'Sangrur': 'Sangrur',
    'Moga': 'Moga',
    'Firozpur': 'Firozpur',
    'Gurdaspur': 'Gurdaspur',
    'Rupnagar': 'Rupnagar',
    'Kapurthala': 'Kapurthala',
    'Barnala': 'Barnala'
}

# --- Step 4: Fetch Real-Time Weather from WeatherAPI ---
def get_weather(city, api_key):
    url = "http://api.weatherapi.com/v1/current.json"
    params = {
        'key': api_key,
        'q': city,
        'aqi': 'no'
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            temp_c = data['current']['temp_c']
            rain_mm = data['current'].get('precip_mm', 0)
            humidity = data['current']['humidity']
            last_updated = data['current']['last_updated']
            return {
                'temp_c': temp_c,
                'rain_mm': rain_mm,
                'humidity': humidity,
                'last_updated': last_updated
            }
        else:
            print(f"❌ {city}: {response.status_code} - {response.json().get('error', {}).get('message', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"⚠️ Error fetching data for {city}: {e}")
        return None

# 🔑 Your WeatherAPI Key
API_KEY = "08d2c6272dff488181e160916250609"

# --- Step 5: Generate Climate Risk Report ---
results = []

for district in soil_df['District']:
    if district not in district_to_city:
        continue

    city = district_to_city[district]
    hist_rain = historical_rainfall.get(district, 700)

    # Get real-time weather
    weather = get_weather(city, API_KEY)
    if not weather:
        continue

    current_rain = weather['rain_mm']
    temp = weather['temp_c']
    humidity = weather['humidity']

    # Rainfall deviation
    rain_deviation = current_rain - (hist_rain / 365 * 7)  # Weekly avg
    if rain_deviation > 20:
        rain_alert = "⚠️ Heavy rain — delay sowing"
    elif rain_deviation < -20:
        rain_alert = "⚠️ Rain deficit — irrigate soon"
    else:
        rain_alert = "✅ Rainfall normal"

    # Reuse previous risk factors
    gw_risk = {'Mansa': 9, 'Bathinda': 8, 'Fazilka': 8}.get(district, 5)
    temp_risk = 2 if temp < 35 else 5 if temp < 40 else 8
    crop_risk = 8  # Rice-Wheat
    oc = soil_df[soil_df['District'] == district]['Organic_Carbon_pct'].values[0]
    oc_risk = 1 if oc < 0.5 else 0

    # Climate Risk Score (0–10)
    base_score = (gw_risk * 0.3 + temp_risk * 0.2 + crop_risk * 0.2 + oc_risk * 0.1)
    rain_impact = min(2.0, current_rain / 25)  # 25mm = +1 risk
    final_score = min(10.0, round(base_score + rain_impact, 1))

    results.append({
        'District': district,
        'City': city,
        'Current_Temp_C': temp,
        'Current_Rain_24h_mm': current_rain,
        'Humidity_%': humidity,
        'Rain_Alert': rain_alert,
        'Climate_Risk_Score': final_score,
        'Recommendation': "Switch to Bajra/Moong" if final_score > 7 else "Monitor weather"
    })

    time.sleep(0.5)  # Respect API rate limits

# --- Step 6: Create DataFrame & Save ---
results_df = pd.DataFrame(results)
print("\n🌦️ REAL-TIME WEATHER & CLIMATE RISK")
print(results_df[[
    'District', 'Current_Temp_C', 'Current_Rain_24h_mm',
    'Humidity_%', 'Rain_Alert', 'Climate_Risk_Score'
]].to_string(index=False))

# Save to CSV
results_df.to_csv('Realtime_Weather_Climate_Risk.csv', index=False)
print(f"\n✅ Data saved to 'Realtime_Weather_Climate_Risk.csv'")

# --- Step 7: Summary ---
high_risk = results_df[results_df['Climate_Risk_Score'] >= 7]
print(f"\n🚨 HIGH RISK DISTRICTS (Score ≥7): {len(high_risk)}")
for _, row in high_risk.iterrows():
    print(f"  {row['District']}: {row['Climate_Risk_Score']}/10 → {row['Rain_Alert']}")