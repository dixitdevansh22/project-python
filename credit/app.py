import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2e3250;
        text-align: center;
    }
    .fraud-alert {
        background: linear-gradient(135deg, #3d0f0f, #5c1a1a);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .legit-alert {
        background: linear-gradient(135deg, #0f2d1a, #1a4a2a);
        border: 2px solid #00cc66;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    data = pd.read_csv('creditcard.csv')

    X = data.drop(columns='Class')
    y = data['Class']

    # SMOTE for better class balancing (upgrade from simple undersampling)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    trained = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        results[name] = {
            "Accuracy":  round(accuracy_score(y_test, preds) * 100, 2),
            "Precision": round(precision_score(y_test, preds) * 100, 2),
            "Recall":    round(recall_score(y_test, preds) * 100, 2),
            "F1 Score":  round(f1_score(y_test, preds) * 100, 2),
            "cm":        confusion_matrix(y_test, preds),
            "preds":     preds,
            "y_test":    y_test,
        }
        trained[name] = m

    return trained, results, data


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("*Machine Learning powered real-time fraud analysis · Built by Devansh Dixit*")
st.markdown("---")

with st.spinner("Training models with SMOTE balancing… (first load only)"):
    trained_models, results, raw_data = load_and_train()

tab1, tab2, tab3 = st.tabs(["🔍 Detect Fraud", "📊 Model Comparison", "📈 Data Insights"])


# ─── TAB 1 — Predict ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Transaction Fraud Checker")
    st.markdown("Select a model and enter the transaction features to get a prediction.")

    col_model, col_space = st.columns([1, 2])
    with col_model:
        chosen_model = st.selectbox("Choose Model", list(trained_models.keys()))

    st.markdown("##### Enter Transaction Features")
    st.caption("Fill in V1–V28, Time, and Amount (all 30 features)")

    cols = st.columns(5)
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    default_legit = [
        -1.3598, -0.0728, 2.5363, 1.3782, -0.3383,
         0.4624,  0.2396, 0.0987, 0.3638, 0.0908,
        -0.5516, -0.6178, -0.9914, -0.3112, 1.4682,
        -0.4704,  0.2080, 0.0258, 0.4040, 0.2514,
        -0.0183,  0.2778, -0.1105, 0.0669, 0.1285,
        -0.1891,  0.1336, -0.0210, 149.62, 0.00
    ]

    feature_values = []
    for i, (fname, dval) in enumerate(zip(feature_names, default_legit)):
        with cols[i % 5]:
            val = st.number_input(fname, value=float(dval),
                                  format="%.4f", key=f"feat_{i}")
            feature_values.append(val)

    st.markdown("")
    predict_btn = st.button("🔎 Analyse Transaction", use_container_width=True,
                            type="primary")

    if predict_btn:
        model = trained_models[chosen_model]
        features = np.array(feature_values).reshape(1, -1)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 2)

        st.markdown("")
        if prediction == 1:
            st.markdown(f"""
            <div class="fraud-alert">
                <h2>🚨 FRAUDULENT TRANSACTION DETECTED</h2>
                <h3>Confidence: {confidence}%</h3>
                <p>This transaction has been flagged as potentially fraudulent.<br>
                Recommend immediate review and card block.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-alert">
                <h2>✅ LEGITIMATE TRANSACTION</h2>
                <h3>Confidence: {confidence}%</h3>
                <p>This transaction appears to be legitimate.<br>
                No suspicious patterns detected.</p>
            </div>""", unsafe_allow_html=True)

        # Probability bar
        st.markdown("")
        col_l, col_r = st.columns(2)
        col_l.metric("🟢 Legitimate Probability", f"{round(proba[0]*100, 2)}%")
        col_r.metric("🔴 Fraud Probability", f"{round(proba[1]*100, 2)}%")


# ─── TAB 2 — Model Comparison ─────────────────────────────────────────────────
with tab2:
    st.subheader("Model Performance Comparison")
    st.caption("Both models trained on SMOTE-balanced data · 80/20 train-test split")

    # Metrics table
    metrics_df = pd.DataFrame(results).T.drop(columns=["cm", "preds", "y_test"])
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#1a4a2a")
                                  .format("{:.2f}%"), use_container_width=True)

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0f1117")
    ax.set_facecolor("#1e2130")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(metric_cols))
    width = 0.3
    colors = ["#4f8ef7", "#f7a44f"]

    for i, (model_name, color) in enumerate(zip(results.keys(), colors)):
        vals = [results[model_name][m] for m in metric_cols]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom",
                    color="white", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_cols, color="white")
    ax.set_ylim(85, 102)
    ax.set_ylabel("Score (%)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e2130", labelcolor="white")
    ax.spines[:].set_color("#2e3250")
    st.pyplot(fig)
    plt.close()

    # Confusion matrices side by side
    st.markdown("#### Confusion Matrices")
    cm_cols = st.columns(2)
    for i, (model_name, res) in enumerate(results.items()):
        with cm_cols[i]:
            st.markdown(f"**{model_name}**")
            fig2, ax2 = plt.subplots(figsize=(4, 3), facecolor="#0f1117")
            ax2.set_facecolor("#1e2130")
            sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Legit", "Fraud"],
                        yticklabels=["Legit", "Fraud"],
                        ax=ax2, cbar=False,
                        annot_kws={"color": "white", "size": 12})
            ax2.set_xlabel("Predicted", color="white")
            ax2.set_ylabel("Actual", color="white")
            ax2.tick_params(colors="white")
            st.pyplot(fig2)
            plt.close()


# ─── TAB 3 — Data Insights ────────────────────────────────────────────────────
with tab3:
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    total = len(raw_data)
    fraud_count = raw_data['Class'].sum()
    legit_count = total - fraud_count

    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Legitimate", f"{legit_count:,}")
    col3.metric("Fraudulent", f"{fraud_count:,}")
    col4.metric("Fraud Rate", f"{round(fraud_count/total*100, 3)}%")

    # Class distribution pie
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0f1117")

    # Pie chart
    axes[0].set_facecolor("#1e2130")
    wedges, texts, autotexts = axes[0].pie(
        [legit_count, fraud_count],
        labels=["Legitimate", "Fraudulent"],
        autopct="%1.2f%%",
        colors=["#4f8ef7", "#ff4444"],
        startangle=90,
        textprops={"color": "white"}
    )
    axes[0].set_title("Class Distribution (Original)", color="white", pad=15)

    # Transaction amount distribution
    axes[1].set_facecolor("#1e2130")
    axes[1].hist(raw_data[raw_data.Class == 0]["Amount"].clip(upper=500),
                 bins=50, color="#4f8ef7", alpha=0.7, label="Legitimate", density=True)
    axes[1].hist(raw_data[raw_data.Class == 1]["Amount"].clip(upper=500),
                 bins=50, color="#ff4444", alpha=0.7, label="Fraudulent", density=True)
    axes[1].set_xlabel("Transaction Amount (USD)", color="white")
    axes[1].set_ylabel("Density", color="white")
    axes[1].set_title("Amount Distribution by Class", color="white")
    axes[1].tick_params(colors="white")
    axes[1].legend(facecolor="#1e2130", labelcolor="white")
    axes[1].spines[:].set_color("#2e3250")

    fig3.patch.set_facecolor("#0f1117")
    st.pyplot(fig3)
    plt.close()

    st.markdown("#### Sample Data")
    st.dataframe(raw_data.head(10), use_container_width=True)
