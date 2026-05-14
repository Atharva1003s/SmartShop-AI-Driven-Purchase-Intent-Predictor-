import streamlit as st
import pickle
import pandas as pd

# Load model (pipeline)
model = pickle.load(open("model/model.pkl", "rb"))

st.set_page_config(page_title="SmartShop Predictor", layout="centered")

st.title("🛒 SmartShop Prediction App")
st.write("Enter customer details")

# ✅ Automatically get feature names from model
try:
    features = model.feature_names_in_
except:
    st.error("Feature names not found. Save them during training.")
    st.stop()

# 🎯 Create inputs dynamically
input_data = {}

for feature in features:
    input_data[feature] = st.text_input(f"Enter {feature}")

# Convert to DataFrame (IMPORTANT)
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: {e}")