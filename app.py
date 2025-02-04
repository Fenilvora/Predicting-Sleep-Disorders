import streamlit as st
import numpy as np
import joblib

# Load the trained model
rf_model = joblib.load('rf_model.joblib')

# Set page configuration
st.set_page_config(page_title="Sleep Disorder Prediction", page_icon="🛌", layout="centered")

# App title and description
st.markdown(
    """
    <h1 style="text-align: center;">🛌 Sleep Disorder Prediction</h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    ### Welcome to the Sleep Disorder Prediction Tool!
    Enter your health and lifestyle details below to find out if you're at risk of sleep disorders such as **Insomnia** or **Sleep Apnea**.  
    All predictions are based on a trained machine learning model.
    """
)

# Input section with columns for better organization
st.subheader("👤 Personal Details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
    age = st.number_input("Age (years)", min_value=1, max_value=120, step=1, help="Enter your age.")

with col2:
    occupation = st.selectbox(
        "Occupation", 
        [
            "Software Engineer", "Doctor", "Sales Representative", "Teacher", 
            "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", 
            "Salesperson", "Manager"
        ],
        help="Choose your occupation."
    )

# Health and lifestyle details
st.subheader("🏋️ Health and Lifestyle Details")

col3, col4 = st.columns(2)
with col3:
    sleep_duration = st.number_input(
        "Sleep Duration (hours)", 
        min_value=0.0, max_value=24.0, step=0.1, 
        help="Enter your average daily sleep duration in hours."
    )
    quality_of_sleep = st.slider(
        "Quality of Sleep (1 = Poor, 10 = Excellent)", 
        min_value=1, max_value=10, step=1, 
        help="Rate your quality of sleep on a scale from 1 to 10."
    )
    physical_activity_level = st.slider(
        "Physical Activity Level (1 = Low, 100 = High)", 
        min_value=1, max_value=100, step=1, 
        help="Rate your daily physical activity level."
    )

with col4:
    stress_level = st.slider(
        "Stress Level (1 = Low, 10 = High)", 
        min_value=1, max_value=10, step=1, 
        help="Rate your stress level on a scale from 1 to 10."
    )
    bmi_category = st.selectbox(
        "BMI Category", 
        ["Underweight", "Normal", "Overweight", "Obese"], 
        help="Select your BMI category."
    )
    heart_rate = st.number_input(
        "Heart Rate (bpm)", 
        min_value=30, max_value=200, step=1, 
        help="Enter your resting heart rate in beats per minute."
    )

# Additional health parameters
st.subheader("🩺 Additional Health Metrics")
daily_steps = st.number_input(
    "Daily Steps", min_value=0, max_value=50000, step=1, help="Enter the average number of steps you walk daily."
)
systolic_bp = st.number_input(
    "Systolic BP (mmHg)", min_value=50, max_value=250, step=1, help="Enter your systolic blood pressure."
)
diastolic_bp = st.number_input(
    "Diastolic BP (mmHg)", min_value=30, max_value=150, step=1, help="Enter your diastolic blood pressure."
)

# Manual encoding of inputs
gender_encoded = 0 if gender == "Male" else 1
bmi_category_encoded = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}[bmi_category]
occupation_mapping = {
    "Software Engineer": 0, "Doctor": 1, "Sales Representative": 2, "Teacher": 3,
    "Nurse": 4, "Engineer": 5, "Accountant": 6, "Scientist": 7, "Lawyer": 8,
    "Salesperson": 9, "Manager": 10
}
occupation_encoded = occupation_mapping[occupation]

# Combine inputs into a feature array
features = np.array([[
    gender_encoded, age, occupation_encoded, sleep_duration, 
    quality_of_sleep, physical_activity_level, stress_level, 
    bmi_category_encoded, heart_rate, daily_steps, systolic_bp, diastolic_bp
]])

# Predict button with some spacing
st.markdown("---")
if st.button("🔍 Predict Sleep Disorder"):
    prediction = rf_model.predict(features)[0]
    prediction_mapping = {0: "Insomnia", 1: "Sleep Apnea", 2: "None"}
    st.success(f"🛌 Prediction: **{prediction_mapping[prediction]}**")
    st.markdown(
        """
        ### What This Means:
        - **Insomnia**: Difficulty falling or staying asleep.
        - **Sleep Apnea**: Breathing interruptions during sleep.
        - **None**: No significant sleep disorder detected.
        """
    )
