# Credit Risk Modeling System

## Project Overview

This project builds a Machine Learning system to predict whether a loan applicant is likely to default.

The objective is to simulate a real-world banking credit risk assessment process using historical customer data.

This is framed as a **Binary Classification Problem**:

- 0 → Good Customer (Low Risk)
- 1 → Bad Customer (High Risk / Default Risk)

---

## Dataset Information

Dataset Used: German Credit Dataset  
Total Records: 1000 customers  
Total Features After Encoding: 20 columns  

The dataset contains:

- Age  
- Job  
- Housing  
- Saving accounts  
- Checking account  
- Credit amount  
- Duration  
- Purpose  
- Risk (Target Variable)

Target Distribution:

- Good Customers (0): 700  
- Bad Customers (1): 300  

The dataset is slightly imbalanced (70% / 30%), which reflects real-world banking data.

---

## Problem Formulation

We aim to learn a function:

Customer Information → Credit Risk

This is a supervised learning problem where:

- X = Customer Features  
- y = Credit Risk Label  

The goal is to train a model that predicts the probability of default.

---

## Project Structure

Credit_Risk_Project/

│
├── data/
│   └── german_credit_data.csv
│
├── notebooks/
│   └── Day1.ipynb
│
├── models/
│
├── app/
│
└── README.md

---

# Work Completed

---

## Day 1 – Data Understanding

1. Loaded dataset using pandas.
2. Inspected dataset structure using:
   - df.shape
   - df.info()
   - df.describe()
3. Verified target distribution.
4. Identified missing values in:
   - Saving accounts
   - Checking account

---

## Day 2 – Data Cleaning & Preprocessing

### Handling Missing Values

Missing values were replaced with "Unknown" to preserve information:

df["Saving accounts"] = df["Saving accounts"].fillna("Unknown")
df["Checking account"] = df["Checking account"].fillna("Unknown")

---

### Target Encoding

Converted:

- good → 0  
- bad → 1  

df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})

This ensures compatibility with machine learning models.

---

### One-Hot Encoding

Categorical features were converted into numeric format:

df_encoded = pd.get_dummies(df, drop_first=True)

This transformed features such as:

- Sex
- Housing
- Purpose
- Saving accounts
- Checking account

into binary indicator columns such as:

- Sex_male
- Housing_own
- Purpose_car
- etc.

Final dataset shape:

(1000, 20)

---

### Feature–Target Separation

Separated dataset into inputs and output:

X = df_encoded.drop("Risk", axis=1)
y = df_encoded["Risk"]

- X → All feature columns
- y → Target column (Risk)

This prepares the dataset for machine learning training.

---

# Current Status

The dataset is now:

- Cleaned
- Fully numeric
- Properly encoded
- Ready for model training

Next Steps:

- Perform train-test split
- Train Logistic Regression model
- Evaluate performance using confusion matrix, precision, recall, ROC-AUC
- Compare with Random Forest
- Add explainability (SHAP)
- Build Streamlit dashboard
- Deploy project

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn (upcoming)
- Matplotlib / Seaborn (upcoming)

---

# Project Goal

To build a complete end-to-end credit risk prediction system that:

- Predicts loan default probability
- Assists in loan approval decision-making
- Demonstrates real-world ML workflow
- Showcases production-level machine learning practices

---

Author: Ashutosh