# app.py (Streamlit app for deployment)
import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
from datetime import time

# Load pre-trained model using pickle
with open('doctor_targeting_model_lgbm.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Load dataset (static)
doctor_data = pd.read_excel("dummy_npi_data.xlsx")
doctor_data['Login Hour'] = pd.to_datetime(doctor_data['Login Time']).dt.hour
doctor_data['Day of Week'] = pd.to_datetime(doctor_data['Login Time']).dt.dayofweek

# Streamlit UI setup
st.set_page_config(page_title="Doctor Survey Targeting", layout="centered")
st.title("🎯 Doctor Survey Targeting WebApp")
st.subheader("Select a Contact Time to filter doctors likely to attend the survey.")

# Initialize session state for contact time
if 'contact_time' not in st.session_state:
    st.session_state['contact_time'] = time(8, 0)  # Default 8 AM

# Time input (smaller width)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    input_time = st.time_input("Select Contact Time (HH:MM)", value=st.session_state['contact_time'])
st.session_state['contact_time'] = input_time
input_hour = input_time.hour

# Prepare data for prediction
feature_cols = ['State', 'Region', 'Speciality', 'Count of Survey Attempts', 'Usage Time (mins)', 'Login Hour', 'Day of Week']
predict_data = doctor_data[feature_cols].copy()

# Update Login Hour to user input and Day of Week same as original (assumed for demo)
predict_data['Login Hour'] = input_hour  

# Predict attendance probabilities
attendance_probs = pipeline.predict_proba(predict_data)[:, 1]  # Probability of likely to attend
doctor_data['Attendance_Probability'] = attendance_probs

# Filter based on threshold
threshold = 0.5  # You can expose this as slider if dynamic control needed
targeted_doctors = doctor_data[doctor_data['Attendance_Probability'] >= threshold]

# Show results
st.subheader(f"📋 Doctors likely to attend survey around {input_hour}:00")
if targeted_doctors.empty:
    st.warning("No doctors found for the selected time. Please try a different time or adjust the threshold.")
else:
    st.success(f"✅ Found {len(targeted_doctors)} doctors likely to attend.")
    display_cols = ['NPI', 'State', 'Region', 'Speciality', 'Attendance_Probability']
    st.dataframe(targeted_doctors[display_cols].reset_index(drop=True), height=400)

    # CSV Download button
    csv_data = targeted_doctors[display_cols].to_csv(index=False)
    b = BytesIO()
    b.write(csv_data.encode())
    b.seek(0)
    st.download_button(
        label="📥 Download Targeted Doctors CSV",
        data=b,
        file_name="targeted_doctors.csv",
        mime='text/csv'
    )
