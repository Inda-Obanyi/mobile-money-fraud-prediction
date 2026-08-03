# =====================================
# Mobile Money Fraud Detection System
# Streamlit Application
# =====================================

# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# Load Saved Model Components (Cached)
# =====================================
@st.cache_resource
def load_assets():
    model = joblib.load("fraud_detection_model/final_fraud_model.pkl")
    scaler = joblib.load("fraud_detection_model/scaler.pkl")
    features = joblib.load("fraud_detection_model/features.pkl")
    return model, scaler, features

model, scaler, features = load_assets()

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Mobile Money Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =====================================
# Application Title
# =====================================
st.title("💳 Mobile Money Fraud Detection System")
st.write(
    """
    This application uses a Machine Learning model
    to detect potentially fraudulent mobile money transactions.
    """
)

# =====================================
# Sidebar Information
# =====================================
st.sidebar.header("About")
st.sidebar.info(
    """
    Machine Learning Fraud Detection System

    Dataset:
    PaySim Mobile Money Dataset

    Model:
    Optimized Machine Learning Classifier
    """
)

# =====================================
# User Input Section
# =====================================
st.header("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Transaction Step", min_value=1, value=1)
    transaction_type = st.selectbox(
        "Transaction Type",
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )
    amount = st.number_input("Transaction Amount", min_value=0.0, value=10000.0)
    oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=50000.0)

with col2:
    newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=40000.0)
    oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=10000.0)
    isFlaggedFraud = st.selectbox("Is Flagged Fraud?", [0, 1])

# =====================================
# Feature Engineering
# =====================================
sender_balance_change = oldbalanceOrg - newbalanceOrig
receiver_balance_change = newbalanceDest - oldbalanceDest
sender_balance_error = oldbalanceOrg - amount - newbalanceOrig
receiver_balance_error = oldbalanceDest + amount - newbalanceDest
amount_balance_ratio = amount / oldbalanceOrg if oldbalanceOrg > 0 else 0
large_transaction = 1 if amount > 200000 else 0

# Encode transaction type
type_mapping = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}
transaction_type_encoded = type_mapping[transaction_type]

# =====================================
# Prediction Button
# =====================================
if st.button("🔍 Detect Fraud"):

    input_data = pd.DataFrame({
        "step": [step],
        "type": [transaction_type_encoded],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest],
        "sender_balance_change": [sender_balance_change],
        "receiver_balance_change": [receiver_balance_change],
        "sender_balance_error": [sender_balance_error],
        "receiver_balance_error": [receiver_balance_error],
        "amount_balance_ratio": [amount_balance_ratio],
        "large_transaction": [large_transaction], 
        "isFlaggedFraud": [isFlaggedFraud] 
    })

    # Arrange columns correctly
    input_data = input_data[features]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    # =====================================
    # Display Result
    # =====================================
    st.subheader("Prediction Result")

    if int(prediction[0]) == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.metric("Fraud Probability", f"{probability*100:.2f}%")

    if probability < 0.3:
        risk = "Low Risk"
    elif probability < 0.7:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.info(f"Risk Level: {risk}")
