import streamlit as st 
import requests
import pickle
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

# --- Page config ---
st.set_page_config(page_title="Flood sight AI", layout="wide")

# --- Sidebar info ---
with st.sidebar:
    st.title("🧭 Navigation")
    st.markdown("**Flood Sight AI** helps estimate flood risks in real-time.")
    st.markdown("1. 📍 Enter a city name\n2. 🚀 Click to predict\n3. 📊 View flood risk")
    st.markdown("---")
    st.info("🔍 Powered by AI + Live Weather APIs")

# --- Load model ---
with open("flood_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- Load static features ---
@st.cache_data
def load_static_features():
    return pd.read_csv("city_static_features.csv")

city_data = load_static_features()

# --- Get weather data ---
def get_weather_data(city):
    api_key = "1151b7fac28fa16856cd53b49c9e21ed"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    if response.get("cod") != 200:
        return None
    return {
        "temp": response["main"]["temp"],
        "humidity": response["main"]["humidity"],
        "wind": response["wind"]["speed"],
        "condition": response["weather"][0]["description"],
        "lat": response["coord"]["lat"],
        "lon": response["coord"]["lon"]
    }

# --- Rainfall data ---
def get_rainfall(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "hourly" in data and "precipitation" in data["hourly"]:
            return data["hourly"]["precipitation"][0]
    except requests.exceptions.RequestException:
        pass
    return 0.0

# --- Dynamic features ---
def get_dynamic_features(weather, rainfall):
    temp = weather["temp"]
    humidity = weather["humidity"]
    wind = weather["wind"]
    return [
        rainfall,         # MonsoonIntensity
        wind / 10,        # RiverManagement
        humidity / 100,   # DrainageSystems
        temp / 50,        # ClimateChange
        0.5               # InadequatePlanning (placeholder)
    ]

# --- Gradient flood risk bar ---
def gradient_bar(pred):
    html_code = f"""
    <div style="width: 100%; background: linear-gradient(to right, green, yellow, red); height: 30px; border-radius: 8px; position: relative; margin-top: 10px;">
        <div style="position: absolute; left: {pred*100}%; top: 0; transform: translateX(-50%);">
            <span style="background: black; color: white; padding: 2px 6px; border-radius: 4px; font-size: 12px;">{pred:.2%}</span>
        </div>
    </div>
    """
    components.html(html_code, height=50)

# --- App UI ---
st.title("🌊 Flood Prediction App")
city = st.text_input("Enter City Name", "Chennai")

if st.button("🚀 Predict Flood Probability"):
    weather = get_weather_data(city)

    if not weather:
        st.error("❌ City not found in weather API.")
    else:
        rainfall = get_rainfall(weather["lat"], weather["lon"])
        dynamic_features = get_dynamic_features(weather, rainfall)

        city_row = city_data[city_data["City"].str.lower() == city.strip().lower()]
        if city_row.empty:
            st.error("❌ City not found in static features CSV.")
        else:
            static_features = city_row.iloc[0].drop("City").tolist()
            final_features = static_features + dynamic_features
            input_array = np.array([final_features])

            prediction = model.predict(input_array)[0]
            st.metric(label="Flood Probability", value=f"{prediction:.2%}")
            gradient_bar(prediction)

            if prediction < 0.3:
                st.success("🟢 Low Risk")
                explanation = "Low flood risk expected. Conditions appear manageable."
            elif prediction < 0.6:
                st.warning("🟡 Moderate Risk")
                explanation = "Moderate flood risk. Stay alert and follow advisories."
            else:
                st.error("🔴 High Risk")
                explanation = "High flood risk! Prepare for emergency and follow local authority guidance."
            
            st.info(f"📘 Explanation: {explanation}")

# --- Real-time weather ---
if city:
    st.subheader("📡 Real-Time Weather")
    weather = get_weather_data(city)
    if weather:
        rainfall = get_rainfall(weather["lat"], weather["lon"])
        st.write(f"**City:** {city}")
        st.write(f"🌡️ Temperature: {weather['temp']} °C")
        st.write(f"💧 Humidity: {weather['humidity']}%")
        st.write(f"💨 Wind Speed: {weather['wind']} m/s")
        st.write(f"🌧️ Rainfall: {rainfall} mm")
        st.write(f"☁️ Condition: {weather['condition']}")
