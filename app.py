import streamlit as st
import pickle

import numpy as np
import pandas as pd

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

# Optional: load scaler
# scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="SmartShop Predictor", layout="centered")

st.title("🛒 SmartShop Prediction App")
st.sidebar.title("About")
st.sidebar.info("SmartShop ML Project")

st.markdown("---")

st.write("Enter customer details to predict outcome")

# Example inputs (modify based on your dataset)
age = st.number_input("Age", min_value=18, max_value=80)
income = st.number_input("Income", min_value=1000)
spending_score = st.slider("Spending Score", 1, 100)


# Convert input to DataFrame with correct column names
input_data = pd.DataFrame(
    [[age, income, spending_score]],
    columns=["Administrative", "ProductRelated_Duration", "BounceRates"]
)

# If scaler used:
# input_data = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data)

    st.success(f"Prediction Result: {prediction[0]}")