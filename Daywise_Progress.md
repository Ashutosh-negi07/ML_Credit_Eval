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
# 📅 Day 3 – Model Training & Evaluation

## ✅ What Was Completed on Day 3

- Trained a baseline Logistic Regression model using training data.
- Evaluated model performance on unseen test data.
- Calculated accuracy, confusion matrix, and classification metrics.
- Identified poor recall for risky customers (class 1).
- Observed class imbalance impact (700 good vs 300 bad customers).
- Understood why accuracy alone is misleading for imbalanced datasets.
- Interpreted confusion matrix in business terms.
- Learned importance of precision vs recall in credit risk modeling.
- Detected convergence warning from logistic regression.
- Applied feature scaling using StandardScaler.
- Retrained model on scaled data.
- Confirmed scaling improved numerical stability but not recall.
- Implemented class imbalance handling using `class_weight="balanced"`.
- Retrained balanced logistic regression model.
- Compared baseline vs scaled vs balanced models.
- Improved recall for risky customers from 22% → 47%.
- Understood trade-off between accuracy and recall.
- Interpreted model results from a banking/business perspective.
- Established a solid baseline model for future improvement.

---

## 📊 Key Results Summary

- Baseline Accuracy: 70%
- Baseline Recall (Risky Customers): 22%
- Balanced Model Accuracy: 57.5%
- Balanced Model Recall (Risky Customers): 47%

---

## 🧠 Core Concepts Learned

- Logistic Regression fundamentals
- Probability-based classification
- Confusion matrix interpretation
- Precision, Recall, and F1-score
- Impact of class imbalance
- Importance of scaling for linear models
- Use of class weights in imbalanced datasets
- Business-oriented model evaluation

---

## 🎯 Outcome of Day 3

Day 3 successfully:

- Built the first working classification model
- Identified model weaknesses
- Applied systematic improvements
- Shifted evaluation from accuracy-focused to risk-focused thinking
- Prepared foundation for advanced evaluation techniques (ROC, AUC, threshold tuning)
