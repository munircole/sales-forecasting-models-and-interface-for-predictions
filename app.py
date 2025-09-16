import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved models and encoders
sales_model = joblib.load("best_sales_model.pkl")
demand_model = joblib.load("best_demand_model.pkl")
encoders = joblib.load("label_encoders.pkl")  # saved during training

# Features (same as training)
features = ["Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
            "Competitor Pricing", "Category", "Region", "Weather Condition", "Seasonality", "Holiday/Promotion"]

cat_cols = ["Category", "Region", "Weather Condition", "Seasonality"]
binary_col = "Holiday/Promotion"

st.title("📊 Sales & Demand Forecasting App")

# Sidebar for navigation
option = st.sidebar.radio("Choose Prediction Type:", ("Sales Forecasting", "Demand Forecasting"))

# --- Input Section (shared for both sales & demand) ---
st.subheader("Enter Feature Values")
inputs = {}

for feat in features:
    if feat in cat_cols:
        # Dropdown with encoder classes
        inputs[feat] = st.selectbox(f"{feat}", encoders[feat].classes_)
    elif feat == binary_col:
        inputs[feat] = st.selectbox(f"{feat}", ["Yes", "No"])
    else:
        inputs[feat] = st.number_input(f"{feat}", value=0.0)

# --- Preprocess input ---
df_input = pd.DataFrame([inputs])

# Apply encoding
for col in cat_cols:
    df_input[col] = encoders[col].transform(df_input[col])

# Convert binary Yes/No → 1/0
df_input[binary_col] = df_input[binary_col].map({"Yes": 1, "No": 0})

# --- Predictions ---
if option == "Sales Forecasting":
    st.header("🔮 Predict Sales")
    if st.button("Predict Sales"):
        prediction = sales_model.predict(df_input)[0]
        st.success(f"Predicted Sales: {prediction:.2f}")

if option == "Demand Forecasting":
    st.header("📦 Predict Demand")
    if st.button("Predict Demand"):
        prediction = demand_model.predict(df_input)[0]
        st.success(f"Predicted Demand: {prediction:.2f}")
