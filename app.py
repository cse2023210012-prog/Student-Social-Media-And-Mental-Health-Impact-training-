import joblib
import pandas as pd
import streamlit as st

# Set Page Title
st.set_page_config(
    page_title="Student Mental Health Predictor", page_icon="🧠"
)

st.title("🧠 Student Mental Health Risk Predictor")
st.write("Adjust student daily habits to predict mental health risk category.")


# Load Serialized Model 
@st.cache_resource
def load_model():
  return joblib.load("model.joblib")


try:
  model = load_model()
  st.success("Model loaded successfully!")
except Exception as e:
  st.error(f"Error loading model.joblib: {e}")
  st.stop()

st.markdown("---")

# User Input Widgets
st.subheader("📋 Enter Student Data")

# Sliders & inputs matching your dataset columns
age = st.slider("Age", min_value=15, max_value=30, value=20)
social_media_hours = st.slider(
    "Daily Social Media Hours",
    min_value=0.0,
    max_value=15.0,
    value=4.0,
    step=0.5,
)
sleep_hours = st.slider(
    "Sleep Hours Per Night", min_value=1.0, max_value=12.0, value=7.0, step=0.5
)

# Predict Button
if st.button("🔮 Predict Risk Status", use_container_width=True):
  # Prepare input features matching your model's expected features
  input_data = pd.DataFrame([{
      "Age": age,
      "Social_Media_Hours": social_media_hours,
      "Sleep_Hours": sleep_hours,
  }])

  try:
    prediction = model.predict(input_data)
    st.subheader("Prediction Result:")
    st.write(f"**Predicted Status:** {prediction[0]}")
  except Exception as e:
    st.error(f"Prediction Error: {e}")
