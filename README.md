# Mobile Money Fraud Detection System 💳

An end-to-end Machine Learning solution designed to identify potentially fraudulent transactions in mobile money networks using behavioral and financial patterns. This project features a clean, user-friendly interface powered by Streamlit.

## 📁 Project Structure

```text
Mobile_Money_Fraud_Detection/
│
├── app.py                      # Interactive Streamlit Web Application
├── sample_transactions.csv     # Sample dataset for application testing
├── requirements.txt            # Required Python packages and dependencies
├── README.md                   # Project documentation and setup guide
└── fraud_detection_model/      # Saved machine learning artifacts
    ├── final_fraud_model.pkl   # Trained Optimized Classifier
    ├── scaler.pkl              # Fitted data preprocessing scaler
    └── features.pkl            # Serialized feature list for schema mapping
```

## 🚀 Key Features

* **Real-time Threat Scoring**: Inputs financial transactions and outputs localized fraud evaluations immediately.
* **Risk Categorization**: Automatically groups risks into Low, Medium, and High-threat buckets based on computed mathematical probabilities.
* **Automated Feature Engineering**: Behind the scenes, the dashboard aggregates values like balance errors, cash-out structural drains, and transaction size metrics to feeding the core model.
* **High Efficiency**: Uses specialized resource caching (`@st.cache_resource`) to run complex multi-dimensional predictions without overloading system RAM.

## 🛠️ Setup and Installation

### 1. Prerequisites
Ensure you have Python installed on your system. Navigate to your project folder using your terminal/command prompt:
```bash
cd Mobile_Money_Fraud_Detection
```

### 2. Install Dependencies
Install all the required library dependencies listed in your requirements file:
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
Launch the interactive Streamlit local server:
```bash
streamlit run app.py
```
Your default web browser should open automatically to `http://localhost:8501`.

## 🧪 Evaluation Test Scenarios

### Scenario A: Legitimate Transaction (Passed Case)
* **Transaction Type**: `CASH_IN`
* **Amount**: `10,000`
* **Sender Old Balance**: `50,000` | **Sender New Balance**: `40,000`
* **Receiver Old Balance**: `2,000` | **Receiver New Balance**: `12,000`
* **Is Flagged Fraud?**: `0`
* *Result*: **Legitimate Transaction** (~0.36% Fraud Probability)

### Scenario B: Account Drain Attack (Flagged Case)
* **Transaction Type**: `TRANSFER` / `CASH_OUT`
* **Amount**: `250,000`
* **Sender Old Balance**: `250,000` | **Sender New Balance**: `0`
* **Receiver Old Balance**: `0` | **Receiver New Balance**: `0`
* **Is Flagged Fraud?**: `1`
* *Result*: 🚨 **Fraudulent Transaction Detected** (~90.00% Fraud Probability - High Risk)
