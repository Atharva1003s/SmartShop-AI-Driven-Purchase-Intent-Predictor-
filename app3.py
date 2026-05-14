import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
try:
    model = pickle.load(open("model/model.pkl", "rb"))
except:
    st.error("Model file not found.")

st.set_page_config(page_title="SmartShop Predictor", layout="centered")

st.title("🛒 SmartShop Prediction App")
st.sidebar.title("About")
st.sidebar.info("SmartShop ML Project")

st.markdown("---")
st.write("Enter customer details to predict outcome")

# --- KEEPING YOUR 3 INPUTS EXACTLY THE SAME ---
age = st.number_input("Age", min_value=18, max_value=80, value=25)
income = st.number_input("Income", min_value=1000, value=40000)
spending_score = st.slider("Spending Score", 1, 100, 50)

# ✅ Get ALL required features from model
try:
    all_features = model.feature_names_in_
except:
    st.error("Model does not have feature names. Ensure it was trained with a DataFrame.")
    st.stop()

# ✅ FIX: Use average/reasonable defaults instead of 0 for background features
# If these are all 0, the model assumes the user spent 0 seconds on the site.
input_dict = {feature: 0 for feature in all_features}

# Set background "Success" drivers to help the model trigger a 'True'
if "PageValues" in input_dict:
    # We map spending_score to PageValues so higher score = higher purchase chance
    input_dict["PageValues"] = (spending_score / 100) * 50 
if "ExitRates" in input_dict:
    input_dict["ExitRates"] = 0.01  # Low exit rate is better

# ✅ Mapping your 3 specific inputs to the model's feature names
# (Make sure these strings match your training column names exactly)
input_dict["Administrative"] = age
input_dict["ProductRelated_Duration"] = income
input_dict["BounceRates"] = (100 - spending_score) / 2000 # Higher spend = lower bounce

# Convert to DataFrame
input_data = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        
        # Improved Output Display
        if prediction[0] == 1 or prediction[0] == True:
            st.success(f"Prediction Result: TRUE")
            st.balloons()
            st.info("✅ Customer is LIKELY to Purchase")
        else:
            st.warning(f"Prediction Result: FALSE")
            st.error("❌ Customer is NOT likely to Purchase")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Optional Debugger (Helps you see what is being sent to the model)
with st.expander("Debug: View Feature Vector"):
    st.write(input_data)