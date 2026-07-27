import joblib
import pandas as pd
import streamlit as st

# Set Page Title
st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🧠")

st.title("🧠 Student Mental Health Risk Predictor")
st.write("Adjust student daily habits to predict mental health risk category.")

# Load Serialized Model 
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model.joblib: {e}")
    st.stop()

st.markdown("---")

# Note for user
st.info("💡 Enter student data below to make a prediction:")

# Generate prediction button
if st.button("🔮 Predict Risk Status", use_container_width=True):
    st.write("Prediction endpoint ready!")
