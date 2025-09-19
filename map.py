# gee_satellite_integration.py
# Connect to Google Earth Engine for Punjab Crop Monitoring

import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print("🛰️ GOOGLE EARTH ENGINE SATELLITE DATA INTEGRATION\n")

# --- Step 1: Initialize Earth Engine ---
try:
    ee.Initialize()
    print("✅ Earth Engine initialized")
except Exception as e:
    print("⚠️ Failed to initialize EE. Run `ee.Authenticate()` first.")
    raise e

# --- Step 2: Define Punjab Region ---
# You can use your district list or draw a boundary
punjab_geometry = ee.Geometry.Polygon(
    [[[73.0, 32.5],
      [73.0, 30.0],
      [76.5, 30.0],
      [76.5, 32.5]]],
    None, False
)

# Or use district centroids from your data
districts = {
    'Ludhiana': [30.9010, 75.8573],
    'Bathinda': [29.9888, 75.0834],
    'Ferozepur': [30.9509, 74.6119],
    'Mansa': [29.8167, 75.3333],
    'Hoshiarpur': [31.1048, 75.9058],
    'Amritsar': [31.6339, 74.8722]
}

# Function to create point
def create_point(lat, lon):
    return ee.Geometry.Point([lon, lat])

# --- Step 3: Get MODIS NDVI (Vegetation Index) ---
def get_ndvi_stats(start_date, end_date, geometry):
    collection = ee.ImageCollection('MODIS/006/MOD13Q1') \
        .filterDate(start_date, end_date) \
        .select('NDVI') \
        .filterBounds(geometry)

    # Reduce to mean NDVI
    mean_image = collection.mean()
    stats = mean_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=250,
        maxPixels=1e9
    )
    return stats.get('NDVI').getInfo()

# --- Step 4: Get CHIRPS Rainfall ---
def get_rainfall(start_date, end_date, geometry):
    rainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry)

    total_rainfall = rainfall.sum().reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=5000,
        maxPixels=1e9
    )
    return total_rainfall.get('precipitation').getInfo()

# --- Step 5: Get Land Surface Temperature (LST) ---
def get_lst(start_date, end_date, geometry):
    lst_collection = ee.ImageCollection('MODIS/006/MOD11A2') \
        .filterDate(start_date, end_date) \
        .select('LST_Day_1km') \
        .filterBounds(geometry)

    mean_lst = lst_collection.mean().multiply(0.02).subtract(273.15)  # Kelvin to °C
    stats = mean_lst.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=1000,
        maxPixels=1e9
    )
    return stats.get('LST_Day_1km').getInfo()

# --- Step 6: Get Soil Moisture (SMAP) ---
def get_soil_moisture(start_date, end_date, geometry):
    smap = ee.ImageCollection('NASA_USDA/HSL/SMAP100M_ERA5L_MEMPHIS_V1') \
        .filterDate(start_date, end_date) \
        .select('ssm') \
        .filterBounds(geometry)

    mean_sm = smap.mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30,
        maxPixels=1e9
    )
    return mean_sm.get('ssm').getInfo()

# --- Step 7: Analyze All Districts ---
results = []
current_year = datetime.now().year
start_date = f'{current_year}-01-01'
end_date = f'{current_year}-12-31'

print("📡 Fetching satellite data for Punjab districts...\n")

for district, (lat, lon) in districts.items():
    point = create_point(lat, lon)

    try:
        ndvi = get_ndvi_stats(start_date, end_date, point)
        rainfall = get_rainfall(start_date, end_date, point)
        lst = get_lst(start_date, end_date, point)
        ssm = get_soil_moisture(start_date, end_date, point)

        results.append({
            'District': district,
            'NDVI': round(ndvi, 3) if ndvi else None,
            'Rainfall_mm': round(rainfall, 1) if rainfall else None,
            'Temp_C': round(lst, 1) if lst else None,
            'Soil_Moisture_m3_m3': round(ssm, 3) if ssm else None
        })

        print(f"📍 {district}")
        print(f"  🌿 NDVI: {ndvi:.3f} (0.3–0.6 = healthy crop)")
        print(f"  🌧️ Rainfall: {rainfall:.1f} mm")
        print(f"  🌡️ Temp: {lst:.1f}°C")
        print(f"  💧 Soil Moisture: {ssm:.3f} m³/m³\n")

    except Exception as e:
        print(f"❌ {district}: {str(e)}")

# --- Step 8: Save to CSV ---
satellite_df = pd.DataFrame(results)
satellite_df.to_csv('Punjab_Satellite_Data.csv', index=False)
print("✅ Satellite data saved to 'Punjab_Satellite_Data.csv'")

# --- Step 9: Plot NDVI vs Yield Correlation (Example) ---
try:
    # Load your yield data
    wheat_yield = pd.read_csv('Table_4.7_Yield_Wheat_1 (1).csv')
    wheat_yield.set_index('District/Year', inplace=True)
    wheat_yield = wheat_yield['2018'].dropna().astype(float)

    # Merge with satellite data
    merged = satellite_df.merge(
        wheat_yield.reset_index().rename(columns={'District/Year': 'District', '2018': 'Yield'}),
        on='District'
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(merged['NDVI'], merged['Yield'], color='green', s=100)
    plt.title('🌾 NDVI vs Wheat Yield (2018)')
    plt.xlabel('NDVI (Satellite Vegetation Index)')
    plt.ylabel('Yield (kg/ha)')
    for i, row in merged.iterrows():
        plt.text(row['NDVI'], row['Yield'], row['District'], fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"📊 Plotting error: {e}")