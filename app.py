import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_curve, roc_auc_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Prostate Cancer Predictor",
    page_icon="🩺",
    layout="wide"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #f0f4f8; }

    /* Force all text to be dark */
    .stApp, .stApp p, .stApp span, .stApp label,
    .stApp div, .stApp input { color: #1a1a1a !important; }

    /* Title & Headings */
    h1 { color: #1a3c5e !important; font-family: 'Segoe UI', sans-serif; }
    h2, h3 { color: #1a3c5e !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a3c5e;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Predict button */
    div.stButton > button {
        background-color: #1a3c5e;
        color: white !important;
        border-radius: 10px;
        padding: 0.6em 2em;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #2e6da4;
        color: white !important;
    }

    /* Cards */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        color: #1a1a1a !important;
    }

    /* Number inputs */
    [data-testid="stNumberInput"] input {
        background-color: white !important;
        color: #1a1a1a !important;
    }

    /* Result boxes */
    .result-malignant {
        background-color: #fff0f0;
        border-left: 5px solid #e53e3e;
        border-radius: 8px;
        padding: 16px;
        font-size: 1.1rem;
        color: #1a1a1a !important;
    }
    .result-benign {
        background-color: #f0fff4;
        border-left: 5px solid #38a169;
        border-radius: 8px;
        padding: 16px;
        font-size: 1.1rem;
        color: #1a1a1a !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: white !important;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        color: #1a1a1a !important;
    }
    [data-testid="stMetricValue"] {
        color: #1a3c5e !important;
    }
    [data-testid="stMetricLabel"] {
        color: #555 !important;
    }

    /* Divider */
    hr { border-color: #d0dce8; }

    /* Footer */
    .footer {
        text-align: center;
        color: #888 !important;
        font-size: 0.8rem;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load Model & Scaler
# ─────────────────────────────────────────
model  = joblib.load('prostate_model.pkl')
scaler = joblib.load('scaler.pkl')

# ─────────────────────────────────────────
# Load & Prepare Data (for analytics page)
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('Prostate_Cancer.csv')
    df = df.drop(columns=['id', 'area', 'fractal_dimension', 'texture'])
    le = LabelEncoder()
    df['diagnosis_result'] = le.fit_transform(df['diagnosis_result'])
    return df

df = load_data()
FEATURES = ['radius', 'perimeter', 'smoothness', 'compactness', 'symmetry']
X = df[FEATURES]
y = df['diagnosis_result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)
y_pred    = model.predict(X_test_s)
y_proba   = model.predict_proba(X_test_s)[:, 1]

# ─────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Prostate Cancer\nPredictor")
    st.divider()
    page = st.radio("Navigate", ["🔍 Predict", "📊 Model Analytics"])
    st.divider()
    st.markdown("**About**")
    st.markdown("This app uses a Machine Learning model trained on prostate cancer diagnostic data to predict tumor type.")

# ═══════════════════════════════════════════
# PAGE 1 — PREDICT
# ═══════════════════════════════════════════
if page == "🔍 Predict":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("🩺 Prostate Cancer Prediction")
    st.markdown("Enter the patient's diagnostic measurements below to predict whether the tumor is **Malignant** or **Benign**.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Patient Measurements")

    col1, col2 = st.columns(2)
    with col1:
        radius      = st.number_input("Radius",      min_value=0.0, value=18.0,  step=0.1)
        perimeter   = st.number_input("Perimeter",   min_value=0.0, value=110.0, step=0.1)
        smoothness  = st.number_input("Smoothness",  min_value=0.0, value=0.105, step=0.001, format="%.3f")
    with col2:
        compactness = st.number_input("Compactness", min_value=0.0, value=0.15,  step=0.001, format="%.3f")
        symmetry    = st.number_input("Symmetry",    min_value=0.0, value=0.20,  step=0.001, format="%.3f")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Predict", use_container_width=True):
        input_data = pd.DataFrame([{
            'radius': radius, 'perimeter': perimeter,
            'smoothness': smoothness, 'compactness': compactness,
            'symmetry': symmetry
        }])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0][1]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Result")

        if prediction == 1:
            st.markdown(f'<div class="result-malignant">⚠️ <b>Malignant</b> — High risk of prostate cancer detected.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-benign">✅ <b>Benign</b> — No malignancy detected.</div>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.metric("Malignancy Probability", f"{probability:.2%}")
        col_b.metric("Benign Probability",     f"{1 - probability:.2%}")

        st.progress(float(probability))
        st.caption("Bar represents malignancy probability (0% → 100%)")
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 2 — MODEL ANALYTICS
# ═══════════════════════════════════════════
elif page == "📊 Model Analytics":

    st.title("📊 Model Analytics")
    st.markdown("Performance metrics and visualizations of the trained model.")

    # ── Metrics Row ──
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['1']['precision']
    recall    = report['1']['recall']

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc:.2%}")
    c2.metric("ROC-AUC",   f"{auc:.2%}")
    c3.metric("Precision", f"{precision:.2%}")
    c4.metric("Recall",    f"{recall:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Charts Row 1 ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.subheader("🔲 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        fig.tight_layout()
        st.pyplot(fig)

    # ROC Curve
    with col2:
        st.subheader("📉 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(fpr, tpr, color='#2e6da4', lw=2, label=f"AUC = {auc:.2f}")
        ax.plot([0,1],[0,1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Charts Row 2 ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    # Class Distribution
    with col3:
        st.subheader("🧬 Class Distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        counts = df['diagnosis_result'].value_counts()
        ax.bar(['Benign', 'Malignant'], counts.values,
               color=['#38a169', '#e53e3e'])
        ax.set_ylabel('Count')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)

    # Feature Importance
    with col4:
        st.subheader("🏆 Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])
        feat_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(feat_df['Feature'], feat_df['Importance'], color='#2e6da4')
        ax.set_xlabel('Importance Score')
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">⚠️ This tool is for educational purposes only.</div>',
                unsafe_allow_html=True)
