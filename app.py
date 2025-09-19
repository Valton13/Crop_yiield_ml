# app.py
# Punjab Agricultural Decision Support System
# Comprehensive Dashboard with All Crops

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

# Set wide layout
st.set_page_config(page_title="🌾 Punjab Agri Dashboard", layout="wide")
st.title("🌾 Punjab Agricultural Decision Support System")

# --- Helper: Safe CSV Loader ---
def load_csv(filename, **kwargs):
    try:
        if os.path.exists(filename):
            return pd.read_csv(filename, **kwargs)
        else:
            st.warning(f"📁 File not found: {filename}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading {filename}: {e}")
        return None

# --- Load Soil Data ---
@st.cache_data
def get_soil_data():
    df = load_csv('Qwen_csv_20250905_bpqico7po.csv')
    if df is not None:
        df['District'] = df['District'].str.replace('_', ' ')
        return df
    return pd.DataFrame()

soil_df = get_soil_data()

# Stop if no soil data
if soil_df.empty:
    st.error("⚠️ Soil data not loaded. Upload 'Qwen_csv_20250905_bpqico7po.csv'")
    st.stop()

# Get list of districts
districts = soil_df['District'].tolist()

# --- Tab Setup ---
# --- Tab Setup (7 tabs) ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🏠 Home", "📈 Yield Prediction",
    "💰 MSP Forecast", "🌱 Fertilizer Advice",
    "⚠️ Zinc Alert", "🔄 Crop Rotation",
    "🎯 Best Crop Recommendation",
    "📅 Weather Forecast",
    "📅 Sowing Calendar"  # ← New tab
])
# =================== TAB 1: Crop Suitability ===================
with tab1:
    st.header("🌾 Best Crop by District")

    # Crop criteria (pH, N, P, Zn, texture)
    crop_criteria = {
        'Wheat': {'pH_low': 6.5, 'pH_high': 8.0, 'N_demand': 'medium'},
        'Rice': {'pH_low': 5.5, 'pH_high': 8.0, 'N_demand': 'high'},
        'Bajra': {'pH_low': 7.5, 'pH_high': 9.0, 'N_demand': 'low'},
        'Maize': {'pH_low': 6.0, 'pH_high': 7.5, 'N_demand': 'high'},
        'Moong': {'pH_low': 6.0, 'pH_high': 8.5, 'N_demand': 'low'},
        'Arhar': {'pH_low': 6.5, 'pH_high': 8.0, 'N_demand': 'medium'}
    }

    def recommend_crop(row):
        scores = {}
        for crop, crit in crop_criteria.items():
            score = 0
            # pH match
            if crit['pH_low'] <= row['pH'] <= crit['pH_high']: score += 3
            # Nitrogen
            n_score = 2 if row['Nitrogen_kg_ha'] >= (300 if crit['N_demand']=='high' else 240) else 1
            # Zinc
            zn_score = 2 if row['Zinc_ppm'] >= 0.6 else 1
            score += n_score + zn_score
            scores[crop] = score
        return max(scores, key=scores.get)

    soil_df['Best_Crop'] = soil_df.apply(recommend_crop, axis=1)

    st.dataframe(soil_df[['District', 'Best_Crop', 'pH', 'Nitrogen_kg_ha', 'Zinc_ppm']], width="stretch")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=soil_df, y='Best_Crop', ax=ax, hue='Best_Crop', palette='Set2', legend=False)
    ax.set_title("Crop Suitability in Punjab")
    st.pyplot(fig)

# =================== TAB 2: Yield Prediction ===================
with tab2:
    st.header("📈 Crop Yield Prediction (2023–2025)")

    def predict_yield(filename, crop_name, use_production=False, assumed_area_ha=None):
        try:
            df = pd.read_csv(filename)
            df.set_index('District/Year', inplace=True)
            df.replace(['NA', '(d)', '(a)', '.', '0'], pd.NA, inplace=True)
            df = df.apply(pd.to_numeric, errors='coerce')

            predictions = {}
            for district in df.index:
                series = df.loc[district].dropna()
                if len(series) < 2: continue

                years = pd.to_numeric(series.index).values.reshape(-1, 1)
                values = series.values

                if use_production and assumed_area_ha:
                    # Convert production (thousand tonnes) to yield (kg/ha)
                    tonnes_per_ha = (values * 1000) / assumed_area_ha  # tonnes → kg
                    y_train = tonnes_per_ha
                else:
                    y_train = values  # already in kg/ha

                model = LinearRegression().fit(years, y_train)
                future = model.predict([[2023], [2024], [2025]])
                future = np.clip(future, 0, None)
                predictions[district] = future

            return predictions
        except Exception as e:
            st.warning(f"Error in {crop_name}: {e}")
            return {}

    # Crop Config
    crops_config = [
        {"name": "Wheat", "file": "Table_4.7_Yield_Wheat_1 (1).csv", "use_production": False},
        {"name": "Rice", "file": "Table_4.7_Yield_Rice_1.csv", "use_production": False},
        {"name": "Maize", "file": "Table_4.7_Yield_Maize.csv", "use_production": False},
        {"name": "Bajra", "file": "Table_4.7_Yield_Bajra_1.csv", "use_production": False},
        {"name": "Moong", "file": "Table_4.7_Production_Moong_3.csv", "use_production": True, "assumed_area_ha": 10000},
        {"name": "Arhar", "file": "Table_4.7_Production_Arhar.csv", "use_production": True, "assumed_area_ha": 5000}
    ]

    selected_crop = st.selectbox("Select Crop", [c["name"] for c in crops_config], key="yield_crop")

    config = next(c for c in crops_config if c["name"] == selected_crop)
    preds = predict_yield(
        config["file"],
        config["name"],
        use_production=config.get("use_production"),
        assumed_area_ha=config.get("assumed_area_ha")
    )

    if preds:
        pred_df = pd.DataFrame(preds).T
        pred_df.columns = ['2023', '2024', '2025']
        pred_df = pred_df.round(0)
        st.dataframe(pred_df, width="stretch")
        if config.get("use_production"):
            st.caption(f"💡 Yield estimated using assumed area of {config['assumed_area_ha']} hectares")
    else:
        st.warning(f"No data available for {selected_crop}")

# =================== TAB 3: MSP Forecast ===================
with tab3:
    st.header("💰 MSP Forecast (2025–2027)")

    msp_df = load_csv('table-ad7b58ca-215e-4352-b0f2-49760098e987-11.csv')
    if msp_df is not None:
        commodity = st.selectbox("Select Commodity", msp_df['Commodity'].dropna().unique(), key="msp_select")
        row = msp_df[msp_df['Commodity'] == commodity].iloc[0, 1:].dropna()

        # Extract years and prices
        years = [int(y.split('-')[0]) for y in row.index]
        prices = row.values.astype(float)

        if len(years) > 1:
            X = np.array(years).reshape(-1, 1)
            y = prices
            model = LinearRegression().fit(X, y)
            future = model.predict([[2025], [2026], [2027]])
            for i, year in enumerate([2025, 2026, 2027]):
                st.metric(f"Predicted MSP {year}", f"₹{future[i]:.0f}/quintal")
        else:
            st.warning("Not enough data for prediction")
    else:
        st.info("Upload 'table-ad7b58ca-215e-4352-b0f2-49760098e987-11.csv' for MSP forecasting")

# =================== TAB 4: Fertilizer Advice ===================
with tab4:
    st.header("🌱 Fertilizer Recommendation (Bags/Acre)")

    base_doses = {
        'Wheat': {'N': 120, 'P': 60},
        'Rice': {'N': 120, 'P': 60},
        'Maize': {'N': 150, 'P': 60},
        'Bajra': {'N': 60, 'P': 30},
        'Moong': {'N': 20, 'P': 40},
        'Arhar': {'N': 20, 'P': 60}
    }

    district = st.selectbox("Select District", districts, key="fert_district")
    crop = st.selectbox("Select Crop", list(base_doses.keys()), key="fert_crop")

    row = soil_df[soil_df['District'] == district].iloc[0]
    n_soil = row['Nitrogen_kg_ha']
    p_soil = row['Phosphorus_kg_ha']
    zn_soil = row['Zinc_ppm']

    base = base_doses[crop]
    n_dose = base['N'] * (0.5 if n_soil > 280 else 0.75 if n_soil > 200 else 1.0)
    p_dose = base['P'] * (0.5 if p_soil < 20 else 0.75 if p_soil < 40 else 1.0)

    # Convert kg/ha → bags/acre (1 bag = 50 kg, 1 acre = 0.4047 ha)
    def kg_to_bags(kg_ha):
        kg_acre = kg_ha * 0.4047
        return round(kg_acre / 50, 1)

    dap_bags = kg_to_bags(p_dose * (60 / 11.8))  # Adjusted for P content in DAP
    urea_bags = kg_to_bags(n_dose * (100 / 46))  # 46% N in Urea
    zn_bags = 5.0 if zn_soil < 0.6 else 0.0

    st.write(f"**Recommended Fertilizer for {crop} in {district}:**")
    st.write(f"- DAP: {dap_bags:.1f} bags/acre")
    st.write(f"- Urea: {urea_bags:.1f} bags/acre")
    st.write(f"- Zinc Sulfate: {'5 bags/acre' if zn_bags else 'Not needed'}")

# =================== TAB 5: Zinc Alert ===================
with tab5:
    st.header("⚠️ Zinc Deficiency Alert")

    zn_df = soil_df[['District', 'Zinc_ppm']].copy()
    zn_df['Status'] = zn_df['Zinc_ppm'].apply(
        lambda x: 'Severe Deficiency' if x < 0.4 else
                  'Moderate Deficiency' if x < 0.6 else
                  'Low' if x < 0.8 else 'Sufficient'
    )
    zn_df['Action'] = zn_df['Zinc_ppm'].apply(
        lambda x: 'Apply 25 kg ZnSO₄/ha' if x < 0.6 else 'No action needed'
    )

    st.dataframe(zn_df, width="stretch")

    fig, ax = plt.subplots()
    zn_df['Status'].value_counts().plot(kind='bar', ax=ax, color=['red', 'orange', 'yellow', 'green'])
    ax.set_title("Zinc Status in Punjab Districts")
    st.pyplot(fig)

# =================== TAB 6: Profitable Crop Rotation Planner (No Graph, With Explanation) ===================
# =================== TAB 6: Crop Rotation Planner (With Arhar) ===================
with tab6:
    st.header("🔄 Crop Rotation Planner")

    # All crops including Arhar
    all_crops = [
        'Rice', 'Wheat', 'Bajra', 'Maize', 'Moong', 'Arhar',
        'Cotton', 'Sugarcane', 'Mustard', 'Gram'
    ]

    current_crop = st.selectbox("Current Crop", all_crops, key="rot_current_crop")
    district = st.selectbox("District", districts, key="rot_district")

    # Get soil data for the district
    try:
        row = soil_df[soil_df['District'] == district].iloc[0]
        ph = row['pH']
        oc = row['Organic_Carbon_pct']
        zn = row['Zinc_ppm']
    except:
        ph = 7.5
        oc = 0.5
        zn = 0.6
        st.warning(f"⚠️ Soil data not found for {district}")

    # Rotation logic with Arhar
    if current_crop == 'Rice':
        if oc < 0.5:
            next_crop = 'Moong'  # Low fertility → legume
        else:
            next_crop = 'Gram'
        st.success(f"🔁 After Rice, grow **{next_crop}** to fix nitrogen and improve soil health")

    elif current_crop == 'Wheat':
        if ph > 8.0:
            next_crop = 'Bajra'  # Alkaline soil → drought-tolerant
        else:
            next_crop = 'Moong'
        st.success(f"🔁 After Wheat, grow **{next_crop}** — suits rotation and soil")

    elif current_crop == 'Bajra':
        next_crop = 'Wheat'
        st.success(f"🔁 After Bajra, grow **{next_crop}** — common rotation in arid zones")

    elif current_crop == 'Maize':
        next_crop = 'Moong' if zn >= 0.6 else 'Arhar'
        st.success(f"🔁 After Maize, grow **{next_crop}** — improves soil fertility")

    elif current_crop == 'Moong':
        next_crop = 'Wheat'
        st.success(f"🔁 After Moong, grow **{next_crop}** — legume-wheat is a sustainable cycle")

    elif current_crop == 'Arhar':
        next_crop = 'Wheat'
        st.success(f"🔁 After Arhar, grow **{next_crop}** — deep-rooted pulse followed by cereal")

    elif current_crop == 'Cotton':
        next_crop = 'Mustard' if ph <= 8.0 else 'Moong'
        st.success(f"🔁 After Cotton, grow **{next_crop}** — breaks pest cycle")

    elif current_crop == 'Sugarcane':
        next_crop = 'Moong or Fallow'
        st.warning(f"🔁 After Sugarcane, grow **{next_crop}** — sugarcane depletes soil; rest or add green manure")

    elif current_crop == 'Mustard':
        next_crop = 'Rice'
        st.success(f"🔁 After Mustard, grow **{next_crop}** — oilseed-cereal rotation")

    elif current_crop == 'Gram':
        next_crop = 'Rice'
        st.success(f"🔁 After Gram, grow **{next_crop}** — Rabi pulse followed by Kharif cereal")

    # General advice
    st.info("""
    💡 **Crop Rotation Benefits**:
    - Reduces pests & diseases
    - Improves soil fertility
    - Prevents nutrient depletion
    - Increases long-term yield
    """)

    # Optional: Show Arhar production trend
  

# =================== TAB 7: Profit Analysis ===================
with tab7:
    st.header("📊 Input Cost vs Output Revenue (per Acre)")

    try:
        # --- Load MSP Data ---
        msp_df = pd.read_csv('table-ad7b58ca-215e-4352-b0f2-49760098e987-11.csv')
        
        # Fix column name if needed
        if 'Parameter/Year' in msp_df.columns:
            msp_df.rename(columns={'Parameter/Year': 'Commodity'}, inplace=True)
        if 'Commodity' not in msp_df.columns:
            st.error("❌ 'Commodity' column not found in MSP file.")
            st.stop()
        
        msp_df.set_index('Commodity', inplace=True)

        # Extract 2023-24 MSP
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

        # --- Load Yield Data ---
        def load_yield(filename):
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

        yields = {
            'Wheat': load_yield('Table_4.7_Yield_Wheat_1 (1).csv'),
            'Rice': load_yield('Table_4.7_Yield_Rice_1.csv'),
            'Maize': load_yield('Table_4.7_Yield_Maize.csv'),
            'Bajra': load_yield('Table_4.7_Yield_Bajra_1.csv'),
        }

        # --- Moong: Use Production and Assumed Area ---
        try:
            moong_df = pd.read_csv('Table_4.7_Production_Moong_3.csv')
            if '2018' in moong_df.columns:
                moong_tonnes = moong_df['2018'].sum() * 1000  # thousand tonnes → kg
            else:
                moong_tonnes = moong_df.iloc[:, -1].sum() * 1000
            moong_area_ha = 10000  # Punjab average
            yields['Moong'] = (moong_tonnes * 1000) / moong_area_ha  # kg/ha
        except:
            yields['Moong'] = 600  # fallback

        # --- Arhar: Use Production and Assumed Area ---
        try:
            arhar_df = pd.read_csv('Table_4.7_Production_Arhar.csv')
            if '2018' in arhar_df.columns:
                arhar_tonnes = arhar_df['2018'].sum() * 1000  # thousand tonnes → kg
            else:
                arhar_tonnes = arhar_df.iloc[:, -1].sum() * 1000
            arhar_area_ha = 5000  # Punjab average
            yields['Arhar'] = (arhar_tonnes * 1000) / arhar_area_ha  # kg/ha
        except:
            yields['Arhar'] = 700  # fallback

        # --- Input Costs (₹/acre) ---
        input_costs = {
            'Wheat': 9200,
            'Rice': 13200,
            'Maize': 9200,
            'Bajra': 5800,
            'Moong': 5400,
            'Arhar': 6500  # PAU estimate
        }

        # --- Calculate Revenue & Profit ---
        results = []
        for crop in yields:
            yield_kg_per_ha = yields[crop]
            msp = msp_2023.get(crop, 0)
            if msp == 0:
                continue

            # Convert to per acre: kg/ha → quintal/acre
            yield_quintal_per_acre = (yield_kg_per_ha * 0.4047) / 100
            revenue = yield_quintal_per_acre * msp
            cost = input_costs[crop]
            profit = revenue - cost

            results.append({
                'Crop': crop,
                'Cost (₹/acre)': int(cost),
                'Revenue (₹/acre)': int(revenue),
                'Profit (₹/acre)': int(profit)
            })

        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, width="stretch")
            results_df.to_csv('Revenue_Analysis_Punjab.csv', index=False)
            
        else:
            st.warning("No data available for profit analysis")

    except Exception as e:
        st.warning(f"⚠️ Profit data not loaded: {e}")

# =================== TAB 8: Weather Forecast ===================
# =================== TAB 8: Weather Forecast ===================
with tab8:
    st.header("📅 7-Day Weather Forecast")

    try:
        import requests
        import pandas as pd

        # 🔑 Your WeatherAPI Key
        API_KEY = "08d2c6272dff488181e160916250609"  # Replace with your key

        # Punjab districts
        districts = [
            'Amritsar', 'Bathinda', 'Ludhiana', 'Jalandhar', 'Patiala',
            'Mansa', 'Fazilka', 'Hoshiarpur', 'Sangrur', 'Moga',
            'Firozpur', 'Gurdaspur', 'Rupnagar', 'Kapurthala', 'Barnala'
        ]

        def get_forecast(city):
            url = "http://api.weatherapi.com/v1/forecast.json"
            params = {
                'key': API_KEY,
                'q': f"{city}, India",
                'days': 7,
                'aqi': 'no'
            }
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()['forecast']['forecastday']
                else:
                    error = response.json().get('error', {}).get('message', 'Unknown error')
                    st.warning(f"❌ {city}: {response.status_code} — {error}")
                    return None
            except Exception as e:
                st.error(f"⚠️ {city}: {e}")
                return None

        selected_district = st.selectbox("Select District", districts, key="forecast_select")

        with st.spinner(f"Fetching forecast for {selected_district}..."):
            forecast = get_forecast(selected_district)

        if forecast:
            data = []
            for day in forecast:
                data.append({
                    'Date': day['date'],
                    'Avg Temp (°C)': day['day']['avgtemp_c'],
                    'Rain (mm)': day['day']['totalprecip_mm'],
                    'Humidity (%)': day['day']['avghumidity'],
                    'Condition': day['day']['condition']['text'],
                    'Wind (kph)': day['day']['maxwind_kph']
                })

            st.dataframe(pd.DataFrame(data), width="stretch")

            # Alert
            total_rain = sum(day['day']['totalprecip_mm'] for day in forecast[:3])
            if total_rain > 20:
                st.warning(f"⚠️ Heavy rain expected (next 3 days: {total_rain:.1f} mm) — delay sowing")
            elif total_rain == 0:
                st.info("✅ No rain expected — irrigate if needed")
            else:
                st.success(f"🌧️ Rain expected: {total_rain:.1f} mm — plan accordingly")
        else:
            st.error("Could not fetch forecast. Check internet or API key.")

    except Exception as e:
        st.error(f"Error: {e}")

# =================== TAB 9: AI Sowing Calendar ===================
with tab9:
    st.header("📅 AI-Powered Sowing Calendar")

    try:
        import requests
        import pandas as pd

        # 🔑 Your WeatherAPI Key
        API_KEY = "08d2c6272dff488181e160916250609 "  # Replace with your key

        # --- Load Soil Data ---
        soil_df = pd.read_csv('Qwen_csv_20250905_bpqico7po.csv')
        soil_df['District'] = soil_df['District'].str.replace('_', ' ')
        districts = soil_df['District'].tolist()

        # --- Crop Sowing Rules (PAU Guidelines) ---
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
                'avoid_rain': False,
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

        # --- District to City Mapping for WeatherAPI ---
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

        # --- Fetch Weather Forecast ---
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
                    st.warning(f"❌ {city}: {response.status_code} — {error}")
                    return None
            except Exception as e:
                st.error(f"⚠️ {city}: {e}")
                return None

        # --- Generate Sowing Advice ---
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

            current_month = pd.Timestamp.now().strftime('%B')
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

                ph_match = rules['soil_ph_low'] <= row['pH'] <= rules['soil_ph_high']
                temp_match = rules['min_temp'] <= avg_temp <= rules['max_temp']
                rain_risk = rules['avoid_rain'] and total_rain > 10
                in_month = rules['ideal_month'] == current_month

                advice = []

                if not in_month:
                    advice.append("📅 Not ideal month")
                if not ph_match:
                    advice.append(f"⚠️ pH {row['pH']:.1f} not ideal")
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

        # --- UI: Select District & Crops ---
        selected_district = st.selectbox("Select District", districts, key="sowing_district")
        selected_crops = st.multiselect(
            "Select Crops to Compare",
            options=['Wheat', 'Rice', 'Bajra', 'Maize', 'Moong', 'Arhar'],
            default=['Wheat', 'Rice', 'Bajra']
        )

        if st.button("Generate Sowing Advice"):
            with st.spinner("Fetching weather and soil analysis..."):
                advice_list = get_sowing_advice(selected_district, selected_crops)

            # Show as DataFrame
            advice_df = pd.DataFrame(advice_list)
            st.dataframe(advice_df, width="stretch")

            # Highlight best option
            best = [a for a in advice_list if "✅" in a['Advice']]
            if best:
                st.success(f"✅ Best option: **{best[0]['Crop']}** — {best[0]['Advice']}")
            else:
                st.warning("⏳ No crop is ideal right now — monitor weather")

    except Exception as e:
        st.error(f"Error: {e}")        