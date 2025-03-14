# app.py

import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
from datetime import time

# Load Model using Pickle
with open('doctor_targeting_model_lgbm.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Load dataset
doctor_data = pd.read_excel("dummy_npi_data.xlsx")
doctor_data['Login Hour'] = pd.to_datetime(doctor_data['Login Time']).dt.hour
doctor_data['Day of Week'] = pd.to_datetime(doctor_data['Login Time']).dt.dayofweek  # Ensuring 'Day of Week' present

# Streamlit UI Setup
st.set_page_config(page_title="Doctor Survey", layout="wide")
st.title("Doctor Survey WebPage")
st.subheader("Select a Contact Time and Day to get a list of doctors.")

# Session state to store contact time and day
if 'contact_time' not in st.session_state:
    st.session_state['contact_time'] = time(8, 0)  # Default time

if 'contact_day' not in st.session_state:
    st.session_state['contact_day'] = "Monday"  # Default Monday

# Time input
input_time = st.time_input("Select Contact Time", value=st.session_state['contact_time'])
st.session_state['contact_time'] = input_time
input_hour = input_time.hour
input_minute = input_time.minute

# Day input
input_day = st.selectbox(
    "Select Day of Week",
    options=[
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ],
    index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(st.session_state['contact_day'])
)
st.session_state['contact_day'] = input_day

# ✅ Map day name to corresponding number for model input
day_name_to_number = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}
input_day_value = day_name_to_number[input_day]  # Now numeric value

# Prepare Data for Prediction
feature_cols = ['State', 'Region', 'Speciality', 'Count of Survey Attempts', 'Usage Time (mins)', 'Login Hour', 'Day of Week']
predict_data = doctor_data[feature_cols].copy()
predict_data['Login Hour'] = input_hour  # User-selected hour
predict_data['Day of Week'] = input_day_value  # User-selected day as number

# Predict Attendance Probability
attendance_probs = pipeline.predict_proba(predict_data)[:, 1]  # Probability of attending

# Add predictions to dataset
doctor_data['Attendance_Probability'] = attendance_probs

# Filter Doctors based on Probability Threshold
threshold = 0.5  # You can also allow user to adjust this
targeted_doctors = doctor_data[doctor_data['Attendance_Probability'] >= threshold]

# Show Results & Download Option
st.subheader(f"Doctors likely to attend survey around {input_time.strftime('%H:%M')} on {input_day}")
if targeted_doctors.empty:
    st.warning("No doctors found for the selected time and day. Please try a different combination or adjust threshold.")
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
