import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.title("credit risk evaluation for a Bank")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_pipeline.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "german_credit_data.csv")

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Model file not found. Run `python src/train.py` first.")
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Saving accounts"] = df["Saving accounts"].fillna("Unknown")
    df["Checking account"] = df["Checking account"].fillna("Unknown")
    df["Risk"] = df["Risk"].str.strip().map({"good": 0, "bad": 1})
    return df

df_ref = load_data()

st.sidebar.header("Customer details")

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
    st.metric("Default Probability", f"{probability:.2%}")

    if prediction == 1:
        st.error("⚠ high risk customer")
    else:
        st.success("✔ low risk customer")

    # --- Risk probability gauge ---
    fig, ax = plt.subplots(figsize=(6, 1.2))
    bar_color = "#e74c3c" if probability >= 0.5 else "#2ecc71"
    ax.barh(["Risk"], [probability], color=bar_color, height=0.5)
    ax.barh(["Risk"], [1 - probability], left=[probability], color="#ecf0f1", height=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Default Probability")
    ax.set_title("Risk Gauge")
    ax.tick_params(left=False)
    st.pyplot(fig)
    plt.close(fig)

    # --- Customer vs dataset distributions ---
    st.subheader("Customer vs Dataset")
    numeric_cols = ["Age", "Credit amount", "Duration"]
    customer_vals = [age, credit_amount, duration]
    dataset_means = [df_ref[c].mean() for c in numeric_cols]

    fig2, axes = plt.subplots(1, 3, figsize=(10, 3))
    for i, (col, cval, dmean) in enumerate(zip(numeric_cols, customer_vals, dataset_means)):
        sns.histplot(df_ref[col], ax=axes[i], color="#3498db", alpha=0.5, bins=20, kde=True)
        axes[i].axvline(cval, color="#e74c3c", linewidth=2, label="Customer")
        axes[i].axvline(dmean, color="#2ecc71", linewidth=2, linestyle="--", label="Mean")
        axes[i].set_title(col)
        axes[i].legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # --- Risk distribution by purpose ---
    st.subheader("Risk Rate by Purpose")
    risk_by_purpose = df_ref.groupby("Purpose")["Risk"].mean().sort_values()
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    bars = sns.barplot(x=risk_by_purpose.index, y=risk_by_purpose.values, ax=ax3,
                       palette="RdYlGn_r")
    ax3.axhline(risk_by_purpose[purpose], color="#e74c3c", linestyle="--",
                linewidth=1.5, label=f"Selected: {purpose}")
    ax3.set_ylabel("Default Rate")
    ax3.set_xlabel("")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax3.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
