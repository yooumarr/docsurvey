# app.py
import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
from datetime import time

# Load Model using pickle (same as training)
with open('doctor_targeting_model_lgbm.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Load dataset (static data)
doctor_data = pd.read_excel("dummy_npi_data.xlsx")
doctor_data['Login Hour'] = pd.to_datetime(doctor_data['Login Time']).dt.hour  

# Streamlit UI Setup
st.set_page_config(page_title="Doctor Survey Targeting", layout="wide")
st.title("Doctor Survey Targeting WebApp")
st.subheader("Select a Contact Time to filter doctors.")

# Set default contact time
if 'contact_time' not in st.session_state:
    st.session_state['contact_time'] = time(8, 0)  

# Time input from user
input_time = st.time_input("Select Contact Time", value=st.session_state['contact_time'])
st.session_state['contact_time'] = input_time 
input_hour = input_time.hour

# Prepare Data for Prediction
feature_cols = ['State', 'Region', 'Speciality', 'Count of Survey Attempts', 'Usage Time (mins)', 'Login Hour', 'Day of Week']
predict_data = doctor_data[feature_cols].copy()
predict_data['Login Hour'] = input_hour  # Replace login hour with selected time

# Predict Attendance Probability
attendance_probs = pipeline.predict_proba(predict_data)[:, 1]  # Probability of likely to attend

# Add predictions to dataset
doctor_data['Attendance_Probability'] = attendance_probs

# Filter Doctors based on Probability Threshold
threshold = 0.5  # Default threshold
targeted_doctors = doctor_data[doctor_data['Attendance_Probability'] >= threshold]

# Show Results & Download Option
st.subheader(f"📋 Doctors likely to attend survey around {input_hour}:00")
if targeted_doctors.empty:
    st.warning("No doctors found for the selected time. Please try a different time or adjust threshold.")
else:
    st.success(f"Found {len(targeted_doctors)} doctors likely to attend.")
    display_cols = ['NPI', 'State', 'Region', 'Speciality', 'Attendance_Probability']
    st.dataframe(targeted_doctors[display_cols].reset_index(drop=True))

    # CSV Download
    csv_data = targeted_doctors[display_cols].to_csv(index=False)
    b = BytesIO()
    b.write(csv_data.encode())
    b.seek(0)
    st.download_button(
        label="Download CSV",
        data=b,
        file_name="targeted_doctors.csv",
        mime='text/csv'
    )
