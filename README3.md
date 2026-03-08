# Credit Risk Project — File-by-File Breakdown

Every file in this project, what it does, what it produces, and how it connects to the rest.

---

## `data/german_credit_data.csv`

**What it is:** The raw dataset — 1000 rows of real customer records from a German bank.

**Columns:** Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose, Risk

**What's in it:**
- Each row is one loan applicant
- `Risk` column is the target — either `good` or `bad`
- Two columns have missing values: `Saving accounts` (183 missing) and `Checking account` (394 missing) — stored as `NA` in the CSV
- 700 good customers, 300 bad — an imbalanced dataset

**How it's used:** This is the single source of truth. Both the notebook and `src/train.py` read from this file. The Streamlit app does NOT read it — it only uses the trained model.

---

## `notebooks/Day1.ipynb`

**What it is:** The exploration and experimentation notebook. Contains all the day-by-day work in sequential cells. Despite the name "Day1", it covers Day 1 through Day 7.

### What happens inside (cell by cell):

**Day 1 — Data Understanding (cells 1–9)**
- Loads the CSV with pandas
- Runs `df.shape` → (1000, 10), `df.info()`, `df.describe()` to understand the data
- Checks `df["Risk"].value_counts()` → 700 good, 300 bad
- Checks `df.isnull().sum()` → finds missing values in Saving/Checking accounts
- Plots a countplot of the Risk distribution

**Result:** We now know the data shape, types, missing values, and class imbalance. No transformations yet.

---

**Day 2 — Data Cleaning (cells 10–21)**
- Fills missing `Saving accounts` and `Checking account` with `"Unknown"`
- Strips whitespace from the `Risk` column
- Maps `Risk`: `good → 0`, `bad → 1`
- One-hot encodes all categorical columns with `pd.get_dummies(df, drop_first=True)` → expands from 10 columns to 20
- Splits into `X` (19 feature columns) and `y` (Risk target)

**Result:** A clean, fully numeric dataset ready for modeling. Shape: X = (1000, 19), y = (1000,)

---

**Day 3 — Model Training & Evaluation (cells 22–29)**
- Splits data 80/20 → 800 train, 200 test
- Trains **baseline Logistic Regression** → 70% accuracy, but only 22% recall on defaulters (class 1) — the model barely catches bad customers
- Applies `StandardScaler` to numeric features → retrains → stability improves but recall stays low
- Trains with `class_weight="balanced"` → recall jumps to 47%, accuracy drops to 57.5%

**Result:** Balanced Logistic Regression is the best so far. Key insight discovered: accuracy is misleading for imbalanced data — recall for defaulters is what matters in banking.

---

**Day 4 — Random Forest (cells 30–35)**
- Trains `RandomForestClassifier(n_estimators=100)` on unscaled data
- Gets 74.5% accuracy but only 32% recall on defaulters — worse than balanced LR
- Compares all three models side by side

**Result:** Random Forest looks better by accuracy but misses more bad customers. Balanced Logistic Regression confirmed as the better choice for this use case.

---

**Day 5 — ROC, AUC & Threshold Tuning (cells 36–41)**
- Extracts prediction probabilities with `predict_proba()[:, 1]` instead of hard 0/1
- Computes ROC curve and AUC = 0.746 (acceptable discrimination)
- Plots the ROC curve
- Tests threshold = 0.3 → 90% recall but 53% accuracy (too aggressive)
- Tests threshold = 0.4 → 81% recall, 63% accuracy (best trade-off)

**Result:** Threshold 0.4 identified as the optimal operating point. The decision threshold is a business choice — lower = catch more defaulters but reject more good customers.

---

**Day 6 — Feature Importance (cells 42–45)**
- Extracts Random Forest `feature_importances_` → top 3: Credit amount, Age, Duration
- Extracts Logistic Regression `coef_` → strongest risk signal: `Checking account_little` (coefficient 0.75)
- Plots horizontal bar charts for both

**Result:** Business profile of high-risk customers: low checking balance, low savings, large loans, long duration. The model is now interpretable, not just a black box.

---

**Day 7 — Hyperparameter Tuning & Pipeline (cells 46–54)**
- Runs `GridSearchCV` on Random Forest with 108 parameter combinations, optimizing for recall, 5-fold cross-validation
- Evaluates best RF model
- Saves `model_balanced` and `scaler` as separate `.pkl` files (these were the old approach)
- **Builds the production pipeline**: `ColumnTransformer` (scaling + encoding) + `LogisticRegression` in a single `sklearn.Pipeline`
- Trains pipeline on the full cleaned dataframe
- Saves as `credit_risk_pipeline.pkl`

**Result:** A single `.pkl` file that takes raw customer data (with original column names like "Sex", "Housing", etc.) and outputs predictions. No separate scaler or encoder needed. This is what the Streamlit app loads.

---

## `src/train.py`

**What it is:** A standalone, reproducible training script. It does the same thing as the notebook's final pipeline cell, but cleanly — no notebooks needed.

**What it does step by step:**
1. Reads `data/german_credit_data.csv`
2. Cleans: fills missing values with "Unknown", strips whitespace, maps Risk to 0/1
3. Defines numeric features (Age, Credit amount, Duration, Job) and categorical features (Sex, Housing, Saving accounts, Checking account, Purpose)
4. Builds an sklearn `Pipeline` with two stages:
   - `ColumnTransformer`: applies `StandardScaler` to numeric columns, `OneHotEncoder(drop="first")` to categorical columns
   - `LogisticRegression(max_iter=1000, class_weight="balanced")`
5. Does an 80/20 split, trains, prints evaluation metrics (accuracy + classification report) — this is just for verification
6. Retrains on the **full** dataset (all 1000 rows)
7. Saves to `models/credit_risk_pipeline.pkl`

**Why it exists:** So you can regenerate the model without opening a notebook. Run `python src/train.py` and the fresh `.pkl` appears in `models/`.

**What it produces:** `models/credit_risk_pipeline.pkl`

---

## `models/credit_risk_pipeline.pkl`

**What it is:** The trained sklearn Pipeline serialized with `joblib`. ~3 KB file.

**What's inside it:** A single Python object that contains:
- A `ColumnTransformer` with fitted `StandardScaler` (knows the mean/std of Age, Credit amount, Duration, Job) and fitted `OneHotEncoder` (knows all categories for Sex, Housing, etc.)
- A `LogisticRegression` model with trained coefficients and `class_weight="balanced"`

**How it's used:** `app.py` loads this file with `joblib.load()`. Then calls `pipeline.predict(input_data)` and `pipeline.predict_proba(input_data)` on raw customer data — the pipeline handles all preprocessing internally.

**Critical note:** This file must be generated by the same scikit-learn version that the app uses. Version is pinned in `requirements.txt` to `scikit-learn==1.8.0`. If versions mismatch, `joblib.load()` will crash.

---

## `app.py`

**What it is:** The Streamlit web application. This is what gets deployed.

**What it does step by step:**
1. Loads `models/credit_risk_pipeline.pkl` — if the file is missing, shows an error and stops
2. Renders a sidebar with 9 input widgets matching the 9 features the model expects:
   - Sliders for Age (18–75) and Duration (6–72 months)
   - Number input for Credit amount (500–20,000)
   - Dropdowns for Job level, Sex, Housing, Saving accounts, Checking account, Purpose
3. When "Predict Risk" is clicked:
   - Builds a single-row pandas DataFrame from the inputs
   - Calls `pipeline.predict_proba()` → gets default probability (0 to 1)
   - Calls `pipeline.predict()` → gets 0 or 1
   - Shows the probability as a metric
   - Shows a red "High Risk" or green "Low Risk" banner

**What it produces:** A live web UI at `localhost:8501` (or on Streamlit Cloud after deployment).

**Column name mapping:** The DataFrame column names in `app.py` (`"Saving accounts"`, `"Checking account"`, etc.) must exactly match what the pipeline was trained on. They do — both use the raw CSV column names.

---

## `requirements.txt`

**What it is:** Pinned Python dependencies for reproducible installs.

**Contents:**
| Package | Version | Why it's needed |
|---------|---------|-----------------|
| pandas | 2.3.3 | Data loading and DataFrame operations |
| numpy | 2.4.2 | Numerical computations (pandas/sklearn dependency) |
| scikit-learn | 1.8.0 | Model training, pipeline, preprocessing — **must match the version that generated the .pkl** |
| matplotlib | 3.10.8 | Plotting in the notebook (not used by app.py) |
| seaborn | 0.13.2 | Statistical plots in the notebook (not used by app.py) |
| streamlit | 1.54.0 | Web app framework |
| joblib | 1.5.3 | Model serialization/deserialization |

**How it's used:** `pip install -r requirements.txt` installs everything. Streamlit Cloud reads this file automatically during deployment.

---

## `.gitignore`

**What it is:** Tells Git which files to exclude from version control.

**What it ignores:**
- `__pycache__/`, `*.pyc` — Python bytecode
- `.ipynb_checkpoints/` — Jupyter auto-saves
- `.DS_Store` — macOS folder metadata
- `.env`, `.venv/`, `venv/` — virtual environments
- `.vscode/`, `.idea/` — IDE settings

**Why it matters:** Without this, Git would track auto-generated junk files that clutter the repo and cause merge conflicts.

---

## `README.md`

**What it is:** The original project readme written during development. Contains the project overview, dataset description, problem formulation, and an older version of the project structure.

**Note:** This file was written incrementally as the project progressed. Some sections (like "Technologies Used" listing scikit-learn as "upcoming") reflect the state at the time of writing, not the final project.

---

## `Daywise_Progress.md`

**What it is:** A detailed learning journal. 718 lines covering Day 1 through Day 6 with:
- What was done each day
- Key results and numbers
- Core concepts learned
- Business interpretations
- Strategic insights

**How it differs from README.md:** The README is a project overview. Daywise_Progress is a chronological learning diary with much more depth — it explains *why* each decision was made, not just *what* was done.

---

## `README2.md`

**What it is:** A deployment-focused project guide. Covers the workflow in 8 phases, the final project structure, how to run/deploy, and key technical decisions.

---

## How Data Flows Through The Project

```
german_credit_data.csv
        │
        ▼
   Day1.ipynb ──── explore, clean, train, compare, tune (experimentation)
        │
        ▼
   src/train.py ── clean reproduction of the final pipeline (production)
        │
        ▼
   models/credit_risk_pipeline.pkl ── trained model artifact
        │
        ▼
   app.py ──────── loads model, serves predictions via Streamlit
        │
        ▼
   User sees web UI at localhost:8501 or on Streamlit Cloud
```

The notebook is where all the experimentation happened. `src/train.py` distills the final working approach into a clean script. The `.pkl` file is the bridge between training and serving. `app.py` never sees the raw data or the training code — it only needs the saved pipeline.
