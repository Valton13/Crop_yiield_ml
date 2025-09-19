# weather_forecast.py
# 7-Day Weather Forecast for Punjab Districts

import requests
import pandas as pd
from datetime import datetime

print("📅 7-DAY WEATHER FORECAST FOR PUNJAB DISTRICTS\n")

# 🔑 Your WeatherAPI Key
API_KEY = "08d2c6272dff488181e160916250609"

# --- Map Districts to Cities ---
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

# --- Fetch 7-Day Forecast ---
def get_forecast(city, api_key):
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        'key': api_key,
        'q': city,
        'days': 7,
        'aqi': 'no',
        'alerts': 'no'
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            forecasts = []
            for day in data['forecast']['forecastday']:
                forecasts.append({
                    'Date': day['date'],
                    'Avg_Temp_C': day['day']['avgtemp_c'],
                    'Max_Temp_C': day['day']['maxtemp_c'],
                    'Min_Temp_C': day['day']['mintemp_c'],
                    'Rainfall_mm': day['day']['totalprecip_mm'],
                    'Humidity': day['day']['avghumidity'],
                    'Condition': day['day']['condition']['text'],
                    'Wind_kph': day['day']['maxwind_kph']
                })
            return forecasts
        else:
            error = response.json().get('error', {}).get('message', 'Unknown error')
            print(f"❌ {city}: {response.status_code} — {error}")
            return None
    except Exception as e:
        print(f"⚠️ Error fetching forecast for {city}: {e}")
        return None

# --- Generate Forecast Report ---
results = []

for district in district_to_city:
    city = district_to_city[district]
    print(f"📡 Fetching 7-day forecast for {district} ({city})...")
    
    forecast = get_forecast(city, API_KEY)
    if not forecast:
        continue

    # Take next 3 days for alerting
    next_3_days = forecast[:3]
    total_rain = sum(day['Rainfall_mm'] for day in next_3_days)
    max_temp = max(day['Max_Temp_C'] for day in next_3_days)

    # Alert logic
    if total_rain > 20:
        alert = "⚠️ Heavy rain expected — delay sowing"
    elif total_rain == 0:
        alert = "✅ No rain — irrigate if needed"
    else:
        alert = "🌧️ Light rain expected — safe to sow"

    # Store first day as summary
    first_day = forecast[0]
    results.append({
        'District': district,
        'Date': first_day['Date'],
        'Avg_Temp_C': first_day['Avg_Temp_C'],
        'Rainfall_mm': first_day['Rainfall_mm'],
        'Humidity': first_day['Humidity'],
        'Condition': first_day['Condition'],
        '3-Day_Rain_Total_mm': round(total_rain, 1),
        'Alert': alert
    })

# --- Save to CSV ---
if results:
    forecast_df = pd.DataFrame(results)
    print("\n📅 7-DAY FORECAST SUMMARY")
    print(forecast_df[['District', 'Date', 'Avg_Temp_C', 'Rainfall_mm', '3-Day_Rain_Total_mm', 'Alert']].to_string(index=False))

    # Save full 7-day forecast
    all_forecasts = []
    for district in district_to_city:
        city = district_to_city[district]
        forecast = get_forecast(city, API_KEY)
        if forecast:
            for day in forecast:
                all_forecasts.append({'District': district, **day})

    full_df = pd.DataFrame(all_forecasts)
    full_df.to_csv('Punjab_7Day_Weather_Forecast.csv', index=False)
    print(f"\n✅ Full 7-day forecast saved to 'Punjab_7Day_Weather_Forecast.csv'")
else:
    print("❌ No forecast data retrieved. Check API key and internet connection.")