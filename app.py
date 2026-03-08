import os
import streamlit as st
import pandas as pd
import joblib

st.title("🏦 Credit Risk Evaluation")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "credit_risk_pipeline.pkl")

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Model file not found. Run `python src/train.py` first.")
    st.stop()

st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 75, 30)
credit_amount = st.sidebar.number_input("Credit Amount", 500, 20000, 5000)
duration = st.sidebar.slider("Duration (Months)", 6, 72, 24)
job = st.sidebar.selectbox("Job Level", [0,1,2,3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
housing = st.sidebar.selectbox("Housing", ["own", "rent", "free"])
saving = st.sidebar.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "Unknown"])
checking = st.sidebar.selectbox("Checking Account", ["little", "moderate", "rich", "Unknown"])
purpose = st.sidebar.selectbox("Purpose", 
                                ["radio/TV", "education", "car",
                                 "furniture/equipment", "business",
                                 "domestic appliances", "repairs",
                                 "vacation/others"])

if st.sidebar.button("Predict Risk"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "Job": [job],
        "Housing": [housing],
        "Saving accounts": [saving],
        "Checking account": [checking],
        "Credit amount": [credit_amount],
        "Duration": [duration],
        "Purpose": [purpose]
    })

    probability = pipeline.predict_proba(input_data)[0][1]
    prediction = pipeline.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.metric("Default Probability", f"{probability:.2f}")

    if prediction == 1:
        st.error("⚠ High Risk Customer")
    else:
        st.success("✔ Low Risk Customer")