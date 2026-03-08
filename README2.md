# Credit Risk Modeling System — Project Guide

## What This Project Does

A machine learning system that predicts whether a loan applicant is likely to **default**, built as a **Streamlit web app**. A bank employee enters customer details in the sidebar and gets an instant risk prediction with default probability.

- **Input:** Customer age, job, housing, savings, checking account, loan amount, duration, purpose
- **Output:** Default probability + risk label (High Risk / Low Risk)

---

## Project Structure

```
Credit_Risk_Project/
├── app.py                 → Streamlit web app (entry point)
├── requirements.txt       → Pinned Python dependencies
├── .gitignore             → Git ignore rules
├── README.md              → Original day-by-day notes
├── Daywise_Progress.md    → Detailed learning journal
├── data/
│   └── german_credit_data.csv   → Raw dataset (1000 records)
├── models/
│   └── credit_risk_pipeline.pkl → Trained sklearn pipeline
├── notebooks/
│   └── Day1.ipynb               → Exploration & experimentation
└── src/
    └── train.py                 → Reproducible training script
```

---

## How The Workflow Progresses

### Phase 1 — Data Understanding (Day 1)
- Loaded the German Credit Dataset (1000 customers, 9 features + target)
- Inspected shape, types, distributions, missing values
- Identified class imbalance: 700 good vs 300 bad customers
- Visualized target distribution

### Phase 2 — Data Cleaning & Preprocessing (Day 2)
- Filled missing values in `Saving accounts` and `Checking account` with "Unknown"
- Stripped whitespace from target column
- Encoded target: good → 0, bad → 1
- One-hot encoded categorical features for model compatibility
- Separated features (X) from target (y)

### Phase 3 — Model Training & Evaluation (Day 3)
- Split data 80/20 (train/test)
- Trained baseline Logistic Regression → 70% accuracy, 22% recall on defaulters
- Applied StandardScaler → improved stability, recall still low
- Used `class_weight="balanced"` → recall jumped to 47% (accuracy dropped to 57.5%)
- Key lesson: in banking, **recall on defaulters matters more than accuracy**

### Phase 4 — Random Forest & Comparison (Day 4)
- Trained Random Forest (100 trees) → 74.5% accuracy but only 32% recall
- Compared all three models side-by-side
- Conclusion: Logistic Regression with balanced weights is the best fit for this use case
- Simpler model outperformed Random Forest on the metric that matters

### Phase 5 — ROC Curve, AUC & Threshold Tuning (Day 5)
- Extracted prediction probabilities instead of hard 0/1 labels
- Plotted ROC curve, computed AUC = 0.746 (acceptable)
- Tested thresholds: 0.5 → 0.4 → 0.3
- Threshold 0.4 gave best balance: 81% recall, 63% accuracy
- Learned that the decision threshold is a **business choice**, not a technical one

### Phase 6 — Feature Importance & Interpretation (Day 6)
- Random Forest feature importances: Credit amount, Age, Duration are top 3
- Logistic Regression coefficients: `Checking account_little` is strongest risk signal
- High-risk profile: low checking balance + low savings + large loan + long duration
- Model transformed from "prediction machine" into "decision intelligence tool"

### Phase 7 — Hyperparameter Tuning & Pipeline (Day 7)
- GridSearchCV on Random Forest (optimizing recall, 5-fold CV)
- Built a production `sklearn.Pipeline` combining preprocessing + model in one object
- Pipeline handles scaling + encoding + prediction in a single `.predict()` call
- Saved pipeline as `credit_risk_pipeline.pkl`

### Phase 8 — Streamlit App & Deployment
- `app.py` loads the saved pipeline and serves predictions through a web UI
- Sidebar inputs map to the exact features the pipeline expects
- Shows default probability as a metric + color-coded risk label
- Ready for Streamlit Cloud deployment

---

## How To Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (optional — pre-trained .pkl is included)
```bash
python src/train.py
```

### 3. Launch the app
```bash
streamlit run app.py
```

---

## How To Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Deploy

Streamlit Cloud will install from `requirements.txt` automatically.

---

## Key Technical Decisions

| Decision | Why |
|----------|-----|
| `class_weight="balanced"` | Prioritizes catching defaulters over raw accuracy |
| Threshold = 0.4 | Best recall/accuracy trade-off for banking use case |
| sklearn Pipeline | Bundles preprocessing + model so the app doesn't need separate scaler/encoder |
| Pinned requirements | Prevents scikit-learn version mismatch breaking `joblib.load` |

---

## Dataset

**German Credit Dataset** — 1000 customer records with 9 features:

| Feature | Type |
|---------|------|
| Age | Numeric |
| Job | Numeric (0–3) |
| Housing | Categorical (own/rent/free) |
| Saving accounts | Categorical (little/moderate/quite rich/rich/Unknown) |
| Checking account | Categorical (little/moderate/rich/Unknown) |
| Credit amount | Numeric |
| Duration | Numeric (months) |
| Purpose | Categorical (car/radio-TV/education/etc.) |
| Sex | Categorical (male/female) |

**Target:** Risk — good (0) / bad (1) — 70/30 split (imbalanced)
