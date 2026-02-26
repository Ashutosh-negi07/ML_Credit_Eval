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

# Day 4 – Random Forest Model & Model Comparison

## 🎯 Objective

The goal of Day 4 was to:

- Train a Random Forest classifier
- Compare it against Logistic Regression models
- Evaluate performance using business-focused metrics
- Decide which model is better suited for credit risk prediction

---

## 📊 Models Evaluated

Three models were compared:

1. Logistic Regression (Scaled)
2. Logistic Regression (Class Balanced)
3. Random Forest Classifier

---

## 📈 Model Performance Comparison

### Logistic Regression (Scaled)

- Accuracy: 0.76
- Recall (Defaulters - Class 1): 0.41
- False Negatives: 35

This model performed well overall but missed a significant number of risky customers.

---

### Logistic Regression (Class Weight = "balanced")

- Accuracy: 0.68
- Recall (Defaulters - Class 1): 0.63
- False Negatives: 22

This model improved recall significantly by prioritizing detection of risky customers, at the cost of overall accuracy.

---

### Random Forest

- Accuracy: 0.745
- Recall (Defaulters - Class 1): 0.32
- False Negatives: 40

Random Forest improved overall accuracy slightly but performed poorly in identifying defaulters.

---

## 🧠 Key Learnings

### 1️⃣ Accuracy is Not Enough

Higher accuracy does not mean a better model in finance.

In credit risk modeling:
- Missing a risky customer (False Negative) is very costly.
- Therefore, Recall for defaulters is more important than overall accuracy.

---

### 2️⃣ Trade-Off Between Accuracy and Recall

When using class balancing:

- Accuracy decreased
- Recall for defaulters increased significantly

This demonstrates the fundamental trade-off between:
- Catching more risky customers
- Maintaining overall prediction correctness

---

### 3️⃣ Random Forest Is Not Always Better

Although Random Forest is more powerful and captures nonlinear relationships:

- It did not outperform Logistic Regression for recall.
- This highlights that simpler models can sometimes perform better for specific business objectives.

---

## 💼 Business Conclusion

For credit risk prediction:

The most suitable model so far is:

Logistic Regression with class_weight="balanced"

Reason:

- It captures 63% of risky customers.
- It reduces the number of missed defaulters.
- It aligns better with financial risk management priorities.

---

## 📌 Strategic Insight

In financial systems:

Lower Accuracy + Higher Recall for Defaulters  
is often preferred over  
Higher Accuracy + Lower Recall.

Because financial institutions prioritize minimizing loan defaults over maximizing classification accuracy.

---

## 🚀 Next Direction

The next step is to:

- Analyze probability outputs instead of fixed 0.5 threshold
- Use ROC Curve and AUC
- Tune decision threshold to optimize business objectives
- Further improve recall without excessive loss in precision

Day 4 established model comparison and clarified business-driven evaluation criteria.

# Day 5 – ROC Curve, AUC & Threshold Optimization

## 🎯 Objective

The goal of Day 5 was to:

- Move beyond simple accuracy evaluation
- Analyze model probabilities instead of fixed class predictions
- Understand ROC Curve and AUC
- Manually tune decision threshold
- Select a business-optimized operating point for credit risk prediction

This marks the transition from basic modeling to professional risk-based evaluation.

---

# 🧠 Key Concepts Covered

---

## 1️⃣ Probabilities vs Class Predictions

Until now, predictions were made using:

model.predict(X)

This internally:

1. Computes probability of class 1 (default)
2. Uses threshold = 0.5
3. Converts probabilities into 0 or 1

However, 0.5 is arbitrary.

In finance, decision thresholds are chosen based on:
- Risk appetite
- Regulatory constraints
- Default tolerance
- Business growth strategy

So instead of fixed predictions, we extracted probabilities.

---

## 2️⃣ Extracting Default Probabilities

We used:

y_prob = model_balanced.predict_proba(X_test_scaled)[:, 1]

### Explanation:

- predict_proba() returns probability for both classes
- [:, 1] selects probability of class 1 (Defaulter)
- This gives a continuous risk score between 0 and 1

These probabilities represent model confidence in default risk.

---

## 3️⃣ ROC Curve (Receiver Operating Characteristic)

ROC Curve plots:

- X-axis → False Positive Rate (FPR)
- Y-axis → True Positive Rate (TPR = Recall)

### Definitions:

True Positive Rate (Recall):

TPR = TP / (TP + FN)

False Positive Rate:

FPR = FP / (FP + TN)

The ROC curve shows model performance across all possible thresholds (0 to 1).

A curve closer to the top-left corner indicates better performance.

---

## 4️⃣ AUC – Area Under Curve

We computed:

auc_score = 0.746

### Interpretation:

| AUC | Meaning |
|------|----------|
| 0.5 | Random guessing |
| 0.6–0.7 | Weak |
| 0.7–0.8 | Acceptable |
| 0.8–0.9 | Good |
| 0.9+ | Excellent |

An AUC of 0.746 indicates acceptable discriminative ability.

This means the model can reasonably distinguish between good and bad customers.

---

# 🎯 Threshold Optimization

Instead of using default threshold 0.5, we tested:

- 0.5 (baseline)
- 0.4
- 0.3

---

# 📊 Performance Comparison by Threshold

---

## 🔹 Threshold = 0.5 (Default)

Confusion Matrix:

[[99 42]
 [22 37]]

Recall (Class 1): 0.63  
False Negatives: 22  
Accuracy: 0.68  

### Interpretation:
- 22 risky customers were missed
- Moderate recall
- Accuracy acceptable
- Risk exposure still high

---

## 🔹 Threshold = 0.4

Confusion Matrix:

[[78 63]
 [11 48]]

Recall (Class 1): 0.81  
False Negatives: 11  
Accuracy: 0.63  

### Interpretation:
- Missed defaulters reduced from 22 → 11
- Recall significantly improved (63% → 81%)
- Accuracy decreased moderately
- Better balance between safety and business loss

---

## 🔹 Threshold = 0.3

Confusion Matrix:

[[53 88]
 [ 6 53]]

Recall (Class 1): 0.90  
False Negatives: 6  
Accuracy: 0.53  

### Interpretation:
- Almost all defaulters detected (90%)
- False negatives reduced dramatically
- Large increase in false positives (88)
- Accuracy dropped significantly
- Very aggressive risk policy

---

# 🧠 Core Insight – Trade-Off

As threshold decreases:

✔ Recall increases  
✔ False Negatives decrease  
❌ Accuracy decreases  
❌ False Positives increase  

This demonstrates the Precision–Recall trade-off.

Lower threshold:
- More customers classified as risky
- Fewer defaulters missed
- More safe customers rejected

Higher threshold:
- Fewer customers flagged risky
- More defaulters slip through
- Higher accuracy

---

# 💼 Business Interpretation

In credit risk modeling:

False Negative (FN) = Approving a defaulter → Financial loss  
False Positive (FP) = Rejecting a good customer → Opportunity loss  

Banks typically prioritize:

Reducing False Negatives

because loan defaults are more costly than missed business opportunities.

---

# 🏦 Recommended Operating Threshold

Based on evaluation:

Threshold = 0.4 provides the best balance:

- Recall = 81%
- False Negatives cut in half (22 → 11)
- Accuracy remains acceptable (63%)

This threshold improves safety without extreme rejection of good customers.

---

# 📌 Key Learnings from Day 5

1. Accuracy is not the primary metric in financial ML.
2. Models output probabilities, not decisions.
3. Decision threshold is a business choice.
4. ROC curve evaluates performance across all thresholds.
5. AUC measures overall model discrimination ability.
6. Risk modeling requires balancing safety vs growth.

---

# 🚀 Outcome of Day 5

By the end of Day 5, the project now includes:

- Probability-based risk scoring
- ROC curve visualization
- AUC evaluation
- Manual threshold tuning
- Business-driven decision optimization

The model is now evaluated at an intermediate-to-advanced level, aligned with real-world credit risk practices.

# 📅 Day 6 – Model Interpretation & Feature Analysis

## 🎯 Objective

To understand:

- Which features influence credit risk the most
- In which direction they influence risk
- How different models interpret the same data
- Extract business insights from model outputs

---

# 🌲 Random Forest – Feature Importance

## Concept

Random Forest provides `feature_importances_`.

This tells us:

> How much each feature contributes to reducing impurity in decision trees.

- Values range from 0 to 1
- All importances sum to 1
- Higher value = more influence in model decisions

---

## Top 10 Important Features

| Feature | Importance |
|----------|------------|
| Credit amount | 0.2295 |
| Age | 0.1857 |
| Duration | 0.1538 |
| Checking account_little | 0.0637 |
| Job | 0.0604 |
| Saving accounts_little | 0.0348 |
| Checking account_moderate | 0.0341 |
| Sex_male | 0.0325 |
| Purpose_car | 0.0290 |
| Purpose_radio/TV | 0.0263 |

---

## Interpretation

### 1️⃣ Credit amount – Most Important
Higher loan size strongly impacts default prediction.

### 2️⃣ Age
Risk may vary non-linearly across age groups.

### 3️⃣ Duration
Longer loans increase uncertainty and risk.

### 4️⃣ Checking & Saving Accounts
Liquidity plays a key role in risk modeling.

---

## Key Insight

Random Forest prioritizes:

- Loan structure
- Financial strength
- Customer liquidity

---

# 📈 Logistic Regression – Coefficient Analysis

## Concept

Logistic Regression provides `model.coef_`.

Equation:

    log(p / (1 - p)) = β0 + β1X1 + ...

If coefficient > 0:
- Feature increases probability of default

If coefficient < 0:
- Feature reduces probability of default

---

## Top Positive Risk Drivers

| Feature | Coefficient |
|----------|------------|
| Checking account_little | 0.75 |
| Checking account_moderate | 0.53 |
| Duration | 0.30 |
| Saving accounts_little | 0.23 |
| Credit amount | 0.12 |
| Job | 0.10 |

---

## Interpretation

### 🔴 Checking account_little
Strongest risk indicator.
Low liquidity = high default probability.

### 🔴 Duration
Longer loan = higher default risk.

### 🔴 Credit amount
Larger loan = more repayment burden.

---

# 🔍 Comparing Both Models

| Logistic Regression | Random Forest |
|--------------------|---------------|
| Shows direction of impact | Shows magnitude of importance |
| Linear model | Non-linear model |
| Interpretable | Powerful but less interpretable |

---

# 🏦 Business Conclusion

High-risk customers typically:

- Have low checking balance
- Have low savings
- Take large loans
- Choose long loan durations
- Have weaker job categories

---

# 📌 What We Learned Today

✔ Feature importance analysis  
✔ Coefficient interpretation  
✔ Model comparison  
✔ Financial interpretation  
✔ Business-level insight extraction  

Day 6 transformed our model from:

"Prediction Machine"

into

"Decision Intelligence Tool"

---

# 🚀 Next Step

Hyperparameter tuning to improve performance.