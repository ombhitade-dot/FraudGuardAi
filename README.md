# FraudGuardAI — Streamlit Dashboard

A Streamlit dashboard for credit-card fraud analytics and machine-learning model comparison.

## Features
- Upload `creditcard.csv` or place it next to `app.py`
- Dataset overview, fraud-rate KPIs and data-quality checks
- Exploratory data analysis and descriptive statistics
- Confidence interval and Welch two-sample t-test for transaction amount
- Correlation heatmap using Matplotlib
- Train Logistic Regression, Decision Tree and Random Forest
- Evaluate precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix and ROC/precision-recall curves
- Download the trained model as `FraudGuardAI_Model.pkl`
- Upload a trusted existing `.pkl` or `.joblib` model for predictions

## Dataset format
Use the ULB/Kaggle credit-card fraud dataset with a target column named `Class`:
- `0` = legitimate transaction
- `1` = fraudulent transaction

The dashboard expects numeric model features. The standard dataset's `Time`, `Amount`, and `V1`–`V28` columns are supported.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints in the terminal (usually `http://localhost:8501`).

## Run on Streamlit Community Cloud
1. Put `app.py`, `requirements.txt`, and this `README.md` in a GitHub repository.
2. Do not commit private datasets or model files if they contain sensitive data.
3. In Streamlit Community Cloud, create a new app and select the repository, branch, and `app.py`.
4. If the dataset is not in the repository, upload it through the dashboard sidebar.

## Important notes
- Exact duplicate removal is optional in the sidebar.
- Model training uses a stratified 80/20 train/test split.
- Since fraud is rare, focus on recall, precision and PR-AUC, not accuracy alone.
- The best model is selected by PR-AUC among the three evaluated models.
- The prediction upload should contain one row with the same numeric feature columns used for training.
- Only load `.pkl`/`.joblib` files from sources you trust; these formats can execute code during loading.
- This is an educational prototype, not a production fraud prevention system.
