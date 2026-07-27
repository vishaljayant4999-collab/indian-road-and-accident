import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Indian Road Accident Severity Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load trained model using joblib
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.exception(e)
    st.stop()

# Title and description
st.title("🚗 Indian Road Accident Severity Prediction App")
st.markdown("""
This app predicts **Accident Severity** (*Minor*, *Major*, or *Fatal*) based on location, 
environmental factors, road infrastructure, and collision parameters.
""")

st.write("---")

# Input layout across three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location & Time Factors")
    
    city_map = {'Bangalore': 0, 'Chandigarh': 1, 'Chennai': 2, 'Delhi': 3, 'Hyderabad': 4, 'Kolkata': 5, 'Mumbai': 6, 'Pune': 7}
    city = st.selectbox("City", list(city_map.keys()))
    
    state_map = {'Chandigarh': 0, 'Delhi': 1, 'Karnataka': 2, 'Maharashtra': 3, 'Tamil Nadu': 4, 'Telangana': 5, 'West Bengal': 6}
    state = st.selectbox("State", list(state_map.keys()))
    
    latitude = st.number_input("Latitude", value=19.0760)
    longitude = st.number_input("Longitude", value=72.8777)
    
    hour = st.slider("Hour of Day (0–23)", 0, 23, 12)
    
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    day_of_week = st.selectbox("Day of Week", list(day_map.keys()))
    
    is_weekend = st.selectbox("Is Weekend?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_peak_hour = st.selectbox("Is Peak Hour?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    st.subheader("🛣️ Infrastructure & Environment")
    
    road_type_map = {'Highway': 0, 'Rural Road': 1, 'Urban Road': 2}
    road_type = st.selectbox("Road Type", list(road_type_map.keys()))
    
    lanes = st.selectbox("Number of Lanes", [1, 2, 3, 4, 5, 6], index=1)
    traffic_signal = st.selectbox("Traffic Signal Present?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    weather_map = {'Clear': 0, 'Fog': 1, 'Rain': 2}
    weather = st.selectbox("Weather Condition", list(weather_map.keys()))
    
    visibility_map = {'Low': 0, 'Medium': 1, 'High': 2}
    visibility = st.selectbox("Visibility Level", list(visibility_map.keys()))
    
    traffic_density_map = {'Low': 0, 'Medium': 1, 'High': 2}
    traffic_density = st.selectbox("Traffic Density", list(traffic_density_map.keys()))
    
    temperature = st.slider("Temperature (°C)", -5, 50, 28)

with col3:
    st.subheader("⚠️ Incident Impact Details")
    
    cause_map = {'Driver Distraction': 0, 'Drunk Driving': 1, 'Overspeeding': 2, 'Poor Road Infrastructure': 3, 'Severe Weather': 4}
    cause = st.selectbox("Primary Cause", list(cause_map.keys()))
    
    festival_map = {'Diwali': 0, 'Eid': 1, 'Holi': 2, 'None': 3, 'New Year': 4}
    festival = st.selectbox("Festival Period", list(festival_map.keys()))
    
    vehicles_involved = st.number_input("Vehicles Involved", min_value=1, max_value=10, value=2)
    casualties = st.number_input("Number of Casualties", min_value=0, max_value=10, value=1)
    risk_score = st.slider("Calculated Risk Score", 0.0, 100.0, 50.0)
    month = st.slider("Month (1-12)", 1, 12, 6)
    year = st.selectbox("Year", [2022, 2023, 2024, 2025])

st.write("---")

# Prediction Execution
if st.button("Predict Accident Severity 🚨", type="primary"):
    
    # 1. Create input data dictionary matching potential dataset features
    raw_input_data = {
        'city': city_map[city],
        'state': state_map[state],
        'latitude': latitude,
        'longitude': longitude,
        'hour': hour,
        'is_weekend': is_weekend,
        'road_type': road_type_map[road_type],
        'lanes': lanes,
        'traffic_signal': traffic_signal,
        'weather': weather_map[weather],
        'visibility': visibility_map[visibility],
        'temperature': temperature,
        'cause': cause_map[cause],
        'vehicles_involved': vehicles_involved,
        'casualties': casualties,
        'traffic_density': traffic_density_map[traffic_density],
        'is_peak_hour': is_peak_hour,
        'risk_score': risk_score,
        'festival': festival_map[festival],
        'day_of_week': day_map[day_of_week],
        'month': month,
        'year': year
    }
    
    # 2. Build DataFrame
    input_df = pd.DataFrame([raw_input_data])
    
    # 3. Align DataFrame directly with trained model features
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
        
        # Add any missing expected columns with 0
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Reorder columns to match the exact order expected by the model
        input_df = input_df[expected_features]
    
    # 4. Predict severity
    prediction = model.predict(input_df)[0]
    
    severity_colors = {0: "🟢 Minor", 1: "🟠 Major", 2: "🔴 Fatal"}
    
    st.subheader("Prediction Result:")
    if prediction == 0:
        st.success(f"Predicted Severity: **{severity_colors[prediction]}**")
    elif prediction == 1:
        st.warning(f"Predicted Severity: **{severity_colors[prediction]}**")
    else:
        st.error(f"Predicted Severity: **{severity_colors[prediction]}**")
