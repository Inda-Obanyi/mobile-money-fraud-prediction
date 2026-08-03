# ======================================================
# Mobile Money Fraud Detection Dashboard
# Streamlit Application Version 2
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# Load Model Components
# ======================================================

model = joblib.load(
    "fraud_detection_model/final_fraud_model.pkl"
)

scaler = joblib.load(
    "fraud_detection_model/scaler.pkl"
)

features = joblib.load(
    "fraud_detection_model/features.pkl"
)

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ======================================================
# Header
# ======================================================

st.title("🛡️ FraudGuard AI")
st.subheader(
    "Mobile Money Fraud Detection & Monitoring Dashboard"
)

st.write(
"""
An Artificial Intelligence system that analyzes mobile money
transactions and identifies potential fraudulent activities.
"""
)

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Single Transaction Prediction",
        "Batch Fraud Detection",
        "Fraud Analytics Dashboard"
    ]
)

# Transaction Type Mapping
type_mapping = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

# ======================================================
# FUNCTION: Prediction
# ======================================================

def predict_transaction(data):
    # CRITICAL FIX: If 'isFlaggedFraud' is expected by the model but missing, add it with a default value of 0
    if "isFlaggedFraud" in features and "isFlaggedFraud" not in data.columns:
        data["isFlaggedFraud"] = 0

    # Align columns to match the exact training features order
    data = data[features]

    scaled_data = scaler.transform(
        data
    )

    prediction = model.predict(
        scaled_data
    )

    probability = model.predict_proba(
        scaled_data
    )[:,1]

    return prediction, probability

# ======================================================
# MODULE 1
# Single Transaction Prediction
# ======================================================

if page == "Single Transaction Prediction":

    st.header(
        "🔍 Single Transaction Fraud Detection"
    )

    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Transaction Step",
            value=1
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            list(type_mapping.keys())
        )

        amount = st.number_input(
            "Transaction Amount",
            value=10000.0
        )

        oldbalanceOrg = st.number_input(
            "Sender Previous Balance",
            value=50000.0
        )

    with col2:
        newbalanceOrig = st.number_input(
            "Sender New Balance",
            value=40000.0
        )

        oldbalanceDest = st.number_input(
            "Receiver Previous Balance",
            value=0.0
        )

        newbalanceDest = st.number_input(
            "Receiver New Balance",
            value=10000.0
        )

    if st.button("🚨 Analyze Transaction"):

        transaction = pd.DataFrame({
            "step": [step],
            "type": [type_mapping[transaction_type]],
            "amount": [amount],
            "oldbalanceOrg": [oldbalanceOrg],
            "newbalanceOrig": [newbalanceOrig],
            "oldbalanceDest": [oldbalanceDest],
            "newbalanceDest": [newbalanceDest],
            "sender_balance_change": [oldbalanceOrg - newbalanceOrig],
            "receiver_balance_change": [newbalanceDest - oldbalanceDest],
            "sender_balance_error": [oldbalanceOrg - amount - newbalanceOrig],
            "receiver_balance_error": [oldbalanceDest + amount - newbalanceDest],
            "amount_balance_ratio": [amount / oldbalanceOrg if oldbalanceOrg > 0 else 0],
            "large_transaction": [1 if amount > 200000 else 0]
        })

        prediction, probability = predict_transaction(
            transaction
        )

        st.divider()

        if prediction[0] == 1:
            st.error(
                "🚨 FRAUD DETECTED"
            )
        else:
            st.success(
                "✅ LEGITIMATE TRANSACTION"
            )

        st.metric(
            "Fraud Probability",
            f"{probability[0]*100:.2f}%"
        )

        if probability[0] < 0.3:
            st.info(
                "Risk Level: LOW"
            )
        elif probability[0] < 0.7:
            st.warning(
                "Risk Level: MEDIUM"
            )
        else:
            st.error(
                "Risk Level: HIGH"
            )

# ======================================================
# MODULE 2
# Batch Prediction
# ======================================================

elif page == "Batch Fraud Detection":

    st.header(
        "📂 Upload Transaction File"
    )

    file = st.file_uploader(
        "Upload CSV File",
        type="csv"
    )

    if file:
        data = pd.read_csv(file)

        st.write(
            "Uploaded Data"
        )
        st.dataframe(
            data.head()
        )

        if st.button(
            "Run Fraud Detection"
        ):
            predictions, probabilities = predict_transaction(
                data
            )

            data["Fraud Prediction"] = predictions
            data["Fraud Probability"] = probabilities

            data["Risk Level"] = pd.cut(
                data["Fraud Probability"],
                bins=[0, .3, .7, 1],
                labels=["Low", "Medium", "High"]
            )

            st.success(
                "Analysis Completed"
            )

            st.dataframe(
                data.head()
            )

            csv = data.to_csv(
                index=False
            )

            st.download_button(
                "⬇ Download Report",
                csv,
                "fraud_report.csv",
                "text/csv"
            )

# ======================================================
# MODULE 3
# Analytics Dashboard
# ======================================================

else:

    st.header(
        "📊 Fraud Analytics Dashboard"
    )

    uploaded = st.file_uploader(
        "Upload Fraud Report",
        type="csv"
    )

    if uploaded:
        df = pd.read_csv(uploaded)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Transactions",
            len(df)
        )

        col2.metric(
            "Fraud Cases",
            int(df["Fraud Prediction"].sum())
        )

        col3.metric(
            "Fraud Rate",
            f"{df['Fraud Prediction'].mean()*100:.2f}%"
        )

        st.subheader(
            "Fraud Distribution"
        )

        fig, ax = plt.subplots()
        sns.countplot(
            x="Fraud Prediction",
            data=df,
            ax=ax
        )
        st.pyplot(fig)

        st.subheader(
            "Risk Level Distribution"
        )

        fig, ax = plt.subplots()
        sns.countplot(
            x="Risk Level",
            data=df,
            ax=ax
        )
        st.pyplot(fig)
