# Credit Risk Modeling System

An end-to-end machine learning project that predicts whether a loan applicant is likely to **default**. Built with scikit-learn and served as a **Streamlit web app**.

A bank employee enters customer details → the model returns a default probability and a risk label (High Risk / Low Risk).

---

## Project Structure

```
Credit_Risk_Project/
├── app.py                          # Streamlit web app (entry point)
├── requirements.txt                # Pinned Python dependencies
├── .gitignore                      # Git ignore rules
├── data/
│   └── german_credit_data.csv      # Raw dataset (1000 records)
├── models/
│   └── credit_risk_pipeline.pkl    # Trained sklearn pipeline
├── notebooks/
│   └── Day1.ipynb                  # Exploration & experimentation (Day 1–7)
└── src/
    └── train.py                    # Reproducible training script
```

---

## Dataset

**German Credit Dataset** — 1000 customer records with 9 features + 1 target.

| Feature | Type | Values |
|---------|------|--------|
| Age | Numeric | 18–75 |
| Job | Numeric | 0–3 (skill level) |
| Housing | Categorical | own, rent, free |
| Saving accounts | Categorical | little, moderate, quite rich, rich, Unknown |
| Checking account | Categorical | little, moderate, rich, Unknown |
| Credit amount | Numeric | 250–18,424 |
| Duration | Numeric | 4–72 months |
| Purpose | Categorical | car, radio/TV, education, furniture/equipment, business, etc. |
| Sex | Categorical | male, female |

**Target:** Risk — good (0) / bad (1)  
**Class distribution:** 700 good (70%) / 300 bad (30%) — imbalanced

---

## Approach

### Problem
Binary classification — predict probability of loan default from customer features.

### Data Preprocessing
- Missing values in `Saving accounts` (183) and `Checking account` (394) filled with `"Unknown"`
- Target encoded: good → 0, bad → 1
- Categorical features one-hot encoded, numeric features scaled via `StandardScaler`
- All preprocessing bundled into an sklearn `Pipeline` so the app processes raw data directly

### Models Evaluated

| Model | Accuracy | Recall (Defaulters) | False Negatives |
|-------|----------|---------------------|-----------------|
| Logistic Regression (baseline) | 70% | 22% | 46 |
| Logistic Regression (scaled) | 76% | 41% | 35 |
| **Logistic Regression (balanced)** | **68%** | **61%** | **22** |
| Random Forest (100 trees) | 74.5% | 32% | 40 |
| Random Forest (tuned via GridSearchCV) | varies | varies | varies |

### Why Balanced Logistic Regression?

In credit risk, **missing a defaulter (False Negative) is far more costly than rejecting a good customer (False Positive)**. The balanced model catches significantly more defaulters at the cost of some accuracy — the right trade-off for banking.

### Threshold Tuning

Default threshold (0.5) was adjusted after ROC/AUC analysis (AUC = 0.746):

| Threshold | Recall | Accuracy | False Negatives |
|-----------|--------|----------|-----------------|
| 0.5 | 63% | 68% | 22 |
| **0.4** | **81%** | **63%** | **11** |
| 0.3 | 90% | 53% | 6 |

Threshold 0.4 provides the best balance for real-world banking use.

### Key Risk Drivers

| Feature | Impact |
|---------|--------|
| Checking account (little) | Strongest risk signal |
| Duration | Longer loans → higher risk |
| Credit amount | Larger loans → more risk |
| Saving accounts (little) | Low savings → higher risk |

---

## How to Use

### Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the model
python src/train.py

# Launch the app
streamlit run app.py
```

### Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select this repo → set main file to `app.py`
4. Deploy

Streamlit Cloud installs from `requirements.txt` automatically.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Language |
| pandas | Data manipulation |
| scikit-learn 1.8.0 | Pipeline, model training, preprocessing |
| Streamlit | Web app framework |
| joblib | Model serialization |
| matplotlib / seaborn | Visualizations (notebook only) |

---

## How It Works

```
german_credit_data.csv
        │
        ▼
  src/train.py ──── cleans data, builds pipeline, trains, saves .pkl
        │
        ▼
  models/credit_risk_pipeline.pkl ──── single file: preprocessing + model
        │
        ▼
  app.py ──── loads pipeline, renders Streamlit UI, serves predictions
```

The pipeline object handles everything — scaling, encoding, and prediction — in a single `pipeline.predict()` call. The app never touches raw data processing.

---

Author: Ashutosh