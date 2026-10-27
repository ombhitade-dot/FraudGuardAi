import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay
)

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="FraudGuardAI | Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- STYLE -------------------------
st.markdown("""
<style>
    .stApp { background: #0b1220; color: #e8eef8; }
    [data-testid="stSidebar"] { background: #101a2c; }
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #142c4e, #18233a 60%, #173b42);
        border: 1px solid #29445f; margin-bottom: 1.1rem;
    }
    .hero h1 { margin: 0; color: #f5f9ff; font-size: 2.2rem; }
    .hero p { margin: .5rem 0 0; color: #a9bfd8; }
    .metric-card {
        padding: 1rem; border-radius: 14px; background: #121e31;
        border: 1px solid #263b54;
    }
    div[data-testid="stMetric"] {
        background: #121e31; padding: 14px 16px; border-radius: 12px;
        border: 1px solid #263b54;
    }
    .small-note { color: #a9bfd8; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛡️ FraudGuardAI</h1>
  <p>Credit Card Fraud Detection · Statistical Inference · Model Evaluation</p>
</div>
""", unsafe_allow_html=True)

# ------------------------- HELPERS -------------------------
@st.cache_data(show_spinner=False)
def read_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

def get_target_column(df):
    if "Class" in df.columns:
        return "Class"
    return None

def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores)
    except Exception:
        return np.nan

def train_models(df, target):
    X = df.drop(columns=[target]).copy()
    y = df[target].astype(int).copy()

    # Keep numeric columns only for this demonstration dashboard.
    X = X.select_dtypes(include=[np.number])
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            ))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                class_weight="balanced", min_samples_leaf=5, random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=150, class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ))
        ])
    }

    fitted, records, scores_by_model, preds_by_model = {}, [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = model.predict_proba(X_test)[:, 1]
        fitted[name] = model
        scores_by_model[name] = score
        preds_by_model[name] = pred
        records.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": safe_auc(y_test, score),
            "PR-AUC": average_precision_score(y_test, score)
        })

    metrics = pd.DataFrame(records).sort_values("PR-AUC", ascending=False)
    return X, y, X_train, X_test, y_train, y_test, fitted, metrics, scores_by_model, preds_by_model

# ------------------------- SIDEBAR -------------------------
st.sidebar.header("⚙️ Dashboard Controls")
uploaded = st.sidebar.file_uploader("Upload creditcard.csv", type=["csv"])
model_file = st.sidebar.file_uploader("Optional trained .pkl/.joblib model", type=["pkl", "joblib"])
st.sidebar.caption("Expected target column: Class (0 = legitimate, 1 = fraud).")

# Allow using a local CSV if present
local_path = "creditcard.csv"
df = None
if uploaded is not None:
    try:
        df = read_csv(uploaded.getvalue())
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
elif os.path.exists(local_path):
    try:
        df = pd.read_csv(local_path)
        st.sidebar.success("Loaded local creditcard.csv")
    except Exception as e:
        st.error(f"Could not read local CSV: {e}")

if df is None:
    st.info("Upload your `creditcard.csv` from the sidebar to open the dashboard. The app can also load a file named `creditcard.csv` placed beside app.py.")
    st.markdown("""
    ### Dashboard includes
    - Dataset overview and data-quality checks
    - Class imbalance and transaction amount exploration
    - Correlation heatmap using Matplotlib
    - Statistical summary
    - Train and compare Logistic Regression, Decision Tree and Random Forest
    - Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix and curves
    - Downloadable trained model and test predictions
    """)
    st.stop()

if "Class" not in df.columns:
    st.error("The uploaded CSV must contain a `Class` target column (0 = legitimate, 1 = fraud).")
    st.stop()

# ------------------------- DATA PREP -------------------------
st.sidebar.subheader("Data settings")
remove_duplicates = st.sidebar.checkbox("Remove exact duplicate rows", value=True)
analysis_df = df.drop_duplicates().copy() if remove_duplicates else df.copy()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Explore Data", "Statistical Analysis", "Train Models", "Predict Transaction"]
)

# ------------------------- OVERVIEW -------------------------
if page == "Overview":
    st.subheader("📊 Executive Overview")
    n_rows, n_cols = analysis_df.shape
    fraud_n = int((analysis_df["Class"] == 1).sum())
    legit_n = int((analysis_df["Class"] == 0).sum())
    fraud_rate = (fraud_n / n_rows * 100) if n_rows else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{n_rows:,}")
    c2.metric("Features", f"{n_cols - 1:,}")
    c3.metric("Fraud cases", f"{fraud_n:,}")
    c4.metric("Fraud rate", f"{fraud_rate:.4f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Transaction class distribution")
        counts = analysis_df["Class"].value_counts().reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values)
        ax.set_ylabel("Number of transactions")
        ax.set_title("Class Distribution")
        ax.tick_params(axis="x", rotation=8)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Transaction amount distribution")
        if "Amount" in analysis_df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            amounts = analysis_df["Amount"].dropna()
            if len(amounts):
                ax.hist(amounts.clip(upper=amounts.quantile(.99)), bins=35, edgecolor="white")
            ax.set_title("Amount (clipped at 99th percentile)")
            ax.set_xlabel("Amount")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No Amount column found.")

    st.markdown("#### Data quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Missing values", f"{int(analysis_df.isna().sum().sum()):,}")
    q2.metric("Duplicate rows remaining", f"{int(analysis_df.duplicated().sum()):,}")
    q3.metric("Columns with missing data", f"{int((analysis_df.isna().sum() > 0).sum()):,}")
    st.dataframe(analysis_df.head(10), use_container_width=True)

# ------------------------- EXPLORE DATA -------------------------
elif page == "Explore Data":
    st.subheader("🔎 Explore the Dataset")
    st.write("Preview and descriptive statistics.")
    st.dataframe(analysis_df.head(100), use_container_width=True)
    st.markdown("#### Descriptive statistics")
    st.dataframe(analysis_df.describe().T, use_container_width=True)

    numeric_cols = analysis_df.select_dtypes(include=np.number).columns.tolist()
    selected = st.selectbox("Select numeric feature for distribution", numeric_cols,
                            index=numeric_cols.index("Amount") if "Amount" in numeric_cols else 0)
    vals = analysis_df[selected].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals.clip(upper=vals.quantile(.99)) if len(vals) else vals, bins=40, edgecolor="white")
    ax.set_title(f"Distribution of {selected}")
    ax.set_xlabel(selected)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)

    if "Amount" in analysis_df.columns:
        st.markdown("#### Amount by class")
        fig, ax = plt.subplots(figsize=(7, 4))
        groups = [
            analysis_df.loc[analysis_df["Class"] == 0, "Amount"].dropna(),
            analysis_df.loc[analysis_df["Class"] == 1, "Amount"].dropna()
        ]
        ax.boxplot(groups, labels=["Legitimate", "Fraud"], showfliers=False)
        ax.set_ylabel("Amount")
        ax.set_title("Transaction amount by class (outliers hidden)")
        st.pyplot(fig)
        plt.close(fig)

# ------------------------- STATISTICAL ANALYSIS -------------------------
elif page == "Statistical Analysis":
    st.subheader("📐 Statistical Analysis")
    st.markdown("#### Summary by transaction class")
    cols = [c for c in ["Time", "Amount"] if c in analysis_df.columns]
    if cols:
        st.dataframe(analysis_df.groupby("Class")[cols].agg(["count", "mean", "median", "std"]).T,
                     use_container_width=True)

    if "Amount" in analysis_df.columns:
        amount = analysis_df["Amount"].dropna()
        st.markdown("#### 95% confidence interval for mean Amount")
        if len(amount) > 1:
            mean_amt = amount.mean()
            sem_amt = stats.sem(amount)
            low, high = stats.t.interval(0.95, df=len(amount)-1, loc=mean_amt, scale=sem_amt)
            a, b, c = st.columns(3)
            a.metric("Sample mean", f"{mean_amt:.4f}")
            b.metric("95% CI lower", f"{low:.4f}")
            c.metric("95% CI upper", f"{high:.4f}")

        st.markdown("#### Welch two-sample t-test: Amount by class")
        legit = analysis_df.loc[analysis_df["Class"] == 0, "Amount"].dropna()
        fraud = analysis_df.loc[analysis_df["Class"] == 1, "Amount"].dropna()
        if len(legit) > 1 and len(fraud) > 1:
            t_stat, p_val = stats.ttest_ind(legit, fraud, equal_var=False)
            st.write(f"t-statistic: **{t_stat:.4f}**")
            st.write(f"p-value: **{p_val:.6g}**")
            st.write("Decision at α = 0.05: **" + ("Reject H₀" if p_val < .05 else "Fail to reject H₀") + "**")

    st.markdown("#### Pearson correlation heatmap")
    numeric = analysis_df.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        image = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=7)
        ax.set_title("Feature Correlation Heatmap")
        fig.colorbar(image, ax=ax, shrink=.8)
        st.pyplot(fig)
        plt.close(fig)

# ------------------------- TRAIN MODELS -------------------------
elif page == "Train Models":
    st.subheader("🤖 Train and Compare Models")
    st.warning("Training can take some time on the full dataset. Metrics use a stratified holdout test set. For fraud data, focus on PR-AUC, recall and precision—not accuracy alone.")
    if st.button("Train models", type="primary"):
        with st.spinner("Training Logistic Regression, Decision Tree and Random Forest..."):
            try:
                results = train_models(analysis_df, "Class")
                X, y, X_train, X_test, y_train, y_test, fitted, metrics, scores, preds = results
                st.session_state["trained_results"] = {
                    "X_columns": X.columns.tolist(),
                    "X_test": X_test,
                    "y_test": y_test,
                    "fitted": fitted,
                    "metrics": metrics,
                    "scores": scores,
                    "preds": preds,
                    "best_name": metrics.iloc[0]["Model"]
                }
                st.success("Training complete!")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if "trained_results" in st.session_state:
        res = st.session_state["trained_results"]
        st.markdown("#### Model comparison")
        st.dataframe(res["metrics"].style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1-score": "{:.4f}", "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"
        }), use_container_width=True)
        best_name = res["best_name"]
        st.success(f"Best model by PR-AUC: **{best_name}**")
        model_choice = st.selectbox("Inspect model", list(res["fitted"].keys()),
                                    index=list(res["fitted"].keys()).index(best_name))
        model = res["fitted"][model_choice]
        y_test = res["y_test"]
        y_pred = res["preds"][model_choice]
        y_score = res["scores"][model_choice]

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        c2.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
        c3.metric("F1-score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")

        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            ax.imshow(cm)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred. Legit", "Pred. Fraud"])
            ax.set_yticklabels(["Actual Legit", "Actual Fraud"])
            ax.set_title("Confusion Matrix")
            for (i, j), value in np.ndenumerate(cm):
                ax.text(j, i, str(value), ha="center", va="center")
            st.pyplot(fig); plt.close(fig)
        with right:
            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y_test, y_score, ax=ax)
            ax.set_title("ROC Curve")
            st.pyplot(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=ax)
        ax.set_title("Precision–Recall Curve")
        st.pyplot(fig); plt.close(fig)

        st.markdown("#### Download trained model")
        model_bytes = io.BytesIO()
        joblib.dump(model, model_bytes)
        st.download_button(
            "⬇️ Download selected model (.pkl)",
            data=model_bytes.getvalue(),
            file_name="FraudGuardAI_Model.pkl",
            mime="application/octet-stream"
        )
        st.download_button(
            "⬇️ Download model comparison (.csv)",
            data=res["metrics"].to_csv(index=False).encode("utf-8"),
            file_name="FraudGuardAI_Model_Comparison.csv",
            mime="text/csv"
        )

# ------------------------- PREDICT -------------------------
elif page == "Predict Transaction":
    st.subheader("🔮 Predict a Transaction")
    st.write("Upload a single-row CSV containing the same numeric feature columns used by the training dataset, or train a model first.")
    if "trained_results" not in st.session_state and model_file is None:
        st.info("Go to Train Models, click Train models, and then return here. You can also upload a trusted trained .pkl/.joblib file in the sidebar.")
    loaded_model = None
    if model_file is not None:
        try:
            loaded_model = joblib.load(model_file)
        except Exception as e:
            st.error(f"Could not load model file: {e}")

    if loaded_model is None and "trained_results" in st.session_state:
        tr = st.session_state["trained_results"]
        choice = st.selectbox("Model", list(tr["fitted"].keys()), key="prediction_model")
        loaded_model = tr["fitted"][choice]

    if loaded_model is not None:
        prediction_file = st.file_uploader("Upload one transaction as CSV", type=["csv"], key="single_transaction")
        if prediction_file is not None:
            try:
                row = pd.read_csv(prediction_file)
                if "Class" in row.columns:
                    row = row.drop(columns=["Class"])
                if "trained_results" in st.session_state:
                    expected_cols = st.session_state["trained_results"]["X_columns"]
                    missing = [c for c in expected_cols if c not in row.columns]
                    extra = [c for c in row.columns if c not in expected_cols]
                    if missing:
                        st.error(f"Missing required feature columns: {missing}")
                    elif extra:
                        st.warning(f"Extra columns will be ignored: {extra}")
                    else:
                        row = row[expected_cols]
                        proba = loaded_model.predict_proba(row)[:, 1]
                        pred = (proba >= 0.5).astype(int)
                        st.metric("Prediction", "FRAUD" if pred[0] else "LEGITIMATE")
                        st.metric("Fraud probability", f"{proba[0]:.2%}")
                else:
                    st.warning("For an uploaded model, make sure the input CSV has exactly the feature columns and order expected by that model.")
                    proba = loaded_model.predict_proba(row)[:, 1]
                    pred = loaded_model.predict(row)[0]
                    st.metric("Prediction", "FRAUD" if pred else "LEGITIMATE")
                    st.metric("Fraud probability", f"{proba[0]:.2%}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("FraudGuardAI is an educational prototype. It is not a production banking decision system. Model outputs can contain false positives and false negatives.")
