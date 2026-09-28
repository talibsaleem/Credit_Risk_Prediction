import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Credit Risk Assessment System", layout="wide")

st.title("🏦 Real-Time Credit Risk & Default Prediction System")
st.markdown("This AI system evaluates loan applications, assesses credit risk probability, and provides **Model Explainability (XAI)** using SHAP values.")

# Load Model & Features
@st.cache_resource
def load_assets():
    model = joblib.load("models/credit_risk_xgb_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names

model, feature_names = load_assets()

# Sidebar - Customer Input Form
st.sidebar.header("📋 Borrower Information")

duration = st.sidebar.number_input("Loan Duration (Months)", min_value=1, max_value=72, value=24)
credit_amount = st.sidebar.number_input("Credit Amount ($)", min_value=250, max_value=20000, value=3000)
age = st.sidebar.number_input("Borrower Age (Years)", min_value=18, max_value=100, value=35)
installment_rate = st.sidebar.slider("Installment Rate (% of Disposable Income)", 1, 4, 2)
existing_credits = st.sidebar.slider("Number of Existing Credits", 1, 4, 1)

# Categorical Selectbox Inputs
checking_status = st.sidebar.selectbox("Checking Account Status", ['A11 (< 0 DM)', 'A12 (0 to 200 DM)', 'A13 (>= 200 DM)', 'A14 (No Checking)'])
savings_status = st.sidebar.selectbox("Savings Account Status", ['A61 (< 100 DM)', 'A62 (100 to 500 DM)', 'A63 (500 to 1000 DM)', 'A64 (>= 1000 DM)', 'A65 (Unknown/None)'])

# Predict Button
if st.sidebar.button("Evaluate Credit Risk", type="primary"):
    
    # 1. Create Empty Input Row matching feature names
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Fill Numerical Features
    if 'duration_in_month' in input_data.columns:
        input_data['duration_in_month'] = duration
    if 'credit_amount' in input_data.columns:
        input_data['credit_amount'] = credit_amount
    if 'age' in input_data.columns:
        input_data['age'] = age
    if 'installment_rate_in_percentage_of_disposable_income' in input_data.columns:
        input_data['installment_rate_in_percentage_of_disposable_income'] = installment_installment = installment_rate
    if 'number_of_existing_credits' in input_data.columns:
        input_data['number_of_existing_credits'] = existing_credits

    # Map One-Hot Features
    checking_code = checking_status.split()[0]
    savings_code = savings_status.split()[0]
    
    for col in input_data.columns:
        if checking_code in col:
            input_data[col] = 1
        if savings_code in col:
            input_data[col] = 1

    # 2. Model Prediction
    probability = model.predict_proba(input_data)[0][1] * 100
    prediction = 1 if probability >= 50 else 0

    # 3. Output Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Risk Prediction Assessment")
        if prediction == 1:
            st.error(f"⚠️ **HIGH CREDIT RISK DETECTED**\n\nDefault Risk Probability: **{probability:.1f}%**")
            st.write("Recommendation: **Reject Application / Require Collateral**")
        else:
            st.success(f"✅ **LOW CREDIT RISK (APPROVED)**\n\nDefault Risk Probability: **{probability:.1f}%**")
            st.write("Recommendation: **Approve Loan Application**")

        st.metric(label="Calculated Default Probability Score", value=f"{probability:.1f}%")

    with col2:
        st.subheader("🔍 Explainable AI (SHAP Reasons)")
        st.write("Top drivers influencing this risk prediction:")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)

        fig, ax = plt.subplots(figsize=(6, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

st.info("👈 Fill borrower details in the sidebar and click **'Evaluate Credit Risk'**.")