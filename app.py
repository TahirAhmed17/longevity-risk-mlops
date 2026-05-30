"""
🫀 Longevity Risk Analyzer — Streamlit Application
=====================================================
A premium health-tech Streamlit dashboard for cardiac risk prediction.
Uses a trained Random Forest model to estimate the probability of
heart-disease-related longevity risk from 13 clinical and lifestyle
features (UCI Heart Disease dataset encoding).

Author : Longevity Risk MLOps Team
Version: 1.0.0
"""

import json
import os
import streamlit as st
import numpy as np
import joblib

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Longevity Risk Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium Health-Tech Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 40%, #24243e 70%, #302b63 100%);
    color: #e0e0e0;
}

/* ── Sidebar ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13112b 0%, #1d1b3a 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa;
}

/* ── Glassmorphism Cards ────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.10);
    padding: 28px 32px;
    margin-bottom: 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(100, 80, 200, 0.18);
}

/* ── Metric Cards ───────────────────────────────────────── */
.metric-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 22px 24px;
    text-align: center;
}
.metric-card h3 {
    color: #a78bfa;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #9ca3af;
    margin-top: 4px;
}

/* ── Risk Bar ───────────────────────────────────────────── */
.risk-bar-track {
    width: 100%;
    height: 24px;
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    overflow: hidden;
    margin: 12px 0;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 12px;
    transition: width 1.2s cubic-bezier(.4,0,.2,1);
}
.risk-bar-green  { background: linear-gradient(90deg, #22c55e, #4ade80); }
.risk-bar-yellow { background: linear-gradient(90deg, #eab308, #facc15); }
.risk-bar-orange { background: linear-gradient(90deg, #f97316, #fb923c); }
.risk-bar-red    { background: linear-gradient(90deg, #ef4444, #f87171); }

/* ── Verdict Badges ─────────────────────────────────────── */
.verdict-high {
    background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(220,38,38,0.15));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
}
.verdict-low {
    background: linear-gradient(135deg, rgba(34,197,94,0.25), rgba(22,163,74,0.15));
    border: 1px solid rgba(34,197,94,0.35);
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
}
.verdict-text {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
}

/* ── Button Override ────────────────────────────────────── */
div.stButton > button {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #ffffff;
    font-weight: 600;
    font-size: 1rem;
    border: none;
    border-radius: 12px;
    padding: 12px 0;
    transition: all 0.25s ease;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #16a34a, #15803d);
    box-shadow: 0 4px 20px rgba(34,197,94,0.35);
    transform: translateY(-1px);
}

/* ── Section Headers ────────────────────────────────────── */
.section-header {
    color: #a78bfa;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 20px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(167,139,250,0.2);
}

/* ── Disclaimer ─────────────────────────────────────────── */
.disclaimer {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    font-size: 0.8rem;
    color: #9ca3af;
    margin-top: 32px;
}

/* ── Recommendations List ───────────────────────────────── */
.rec-item {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-left: 3px solid #a78bfa;
    font-size: 0.92rem;
    color: #d1d5db;
}

/* hide default streamlit footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS — Feature encoding maps
# ─────────────────────────────────────────────────────────────
SEX_MAP = {"Male": 1, "Female": 0}
CP_MAP = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
FBS_MAP = {"Yes": 1, "No": 0}
RESTECG_MAP = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
EXANG_MAP = {"Yes": 1, "No": 0}
SLOPE_MAP = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
THAL_MAP = {"Normal": 2, "Fixed Defect": 1, "Reversible Defect": 3}

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "longevity_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.json")

# ─────────────────────────────────────────────────────────────
# HELPER — load model artifacts
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained model, scaler, and optional threshold."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    # Threshold is saved as JSON by train.py
    threshold = 0.5
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as fp:
            data = json.load(fp)
        threshold = float(data.get("optimal_threshold", 0.5))
    return model, scaler, threshold


def risk_color_class(prob: float) -> str:
    """Return CSS class name based on risk probability."""
    if prob < 0.30:
        return "risk-bar-green"
    elif prob < 0.50:
        return "risk-bar-yellow"
    elif prob < 0.70:
        return "risk-bar-orange"
    return "risk-bar-red"


def get_recommendations(prob: float, inputs: dict) -> list[str]:
    """Generate personalized health recommendations based on inputs."""
    recs: list[str] = []
    if prob >= 0.50:
        recs.append("🏥 Schedule a comprehensive cardiac evaluation with a cardiologist.")
    if inputs["chol"] > 240:
        recs.append("🥗 Cholesterol is elevated — consider dietary changes and lipid panel monitoring.")
    if inputs["trestbps"] > 140:
        recs.append("💊 Resting blood pressure is high — discuss antihypertensive options with your doctor.")
    if inputs["fbs"] == 1:
        recs.append("🩸 Fasting blood sugar is elevated — screen for diabetes and manage glucose levels.")
    if inputs["thalach"] < 100:
        recs.append("🏃 Maximum heart rate is low — a supervised exercise stress test is recommended.")
    if inputs["exang"] == 1:
        recs.append("⚠️ Exercise-induced angina detected — avoid strenuous activity until medically cleared.")
    if inputs["oldpeak"] > 2.0:
        recs.append("📉 Significant ST depression — further ischemic evaluation is warranted.")
    if inputs["ca"] >= 2:
        recs.append("🫀 Multiple major vessels affected — advanced imaging (CT angiography) may be needed.")
    if inputs["age"] >= 60:
        recs.append("📅 Age is a contributing factor — maintain regular annual cardiac check-ups.")
    if prob < 0.30:
        recs.append("✅ Continue your current healthy lifestyle — regular exercise and balanced nutrition.")
        recs.append("💧 Stay hydrated and manage stress through mindfulness or meditation.")
    if not recs:
        recs.append("📋 Maintain regular health check-ups and a balanced lifestyle.")
    return recs


def get_risk_factors(inputs: dict) -> list[str]:
    """Identify contributing risk factors from the patient inputs."""
    factors: list[str] = []
    if inputs["age"] >= 55:
        factors.append(f"🔴 Age ({inputs['age']}) — elevated risk bracket")
    if inputs["sex"] == 1:
        factors.append("🔵 Male sex — statistically higher baseline risk")
    if inputs["cp"] == 3:
        factors.append("🟠 Asymptomatic chest pain type — often masks underlying disease")
    if inputs["trestbps"] > 140:
        factors.append(f"🔴 Resting BP ({inputs['trestbps']} mmHg) — hypertensive range")
    if inputs["chol"] > 240:
        factors.append(f"🟠 Cholesterol ({inputs['chol']} mg/dL) — above desirable level")
    if inputs["fbs"] == 1:
        factors.append("🟠 Fasting blood sugar > 120 mg/dL — diabetic indicator")
    if inputs["restecg"] != 0:
        factors.append("🟡 Abnormal resting ECG findings")
    if inputs["thalach"] < 120:
        factors.append(f"🟠 Low max heart rate ({inputs['thalach']} bpm)")
    if inputs["exang"] == 1:
        factors.append("🔴 Exercise-induced angina present")
    if inputs["oldpeak"] > 1.5:
        factors.append(f"🟠 ST depression ({inputs['oldpeak']}) — ischemic indicator")
    if inputs["slope"] == 2:
        factors.append("🟡 Downsloping ST segment — associated with higher risk")
    if inputs["ca"] >= 1:
        factors.append(f"🔴 {inputs['ca']} major vessel(s) colored by fluoroscopy")
    if inputs["thal"] == 3:
        factors.append("🔴 Reversible thalassemia defect — perfusion abnormality")
    elif inputs["thal"] == 1:
        factors.append("🟠 Fixed thalassemia defect — prior infarction indicator")
    return factors


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Patient Inputs
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🫀 Longevity Risk Analyzer")
    st.markdown(
        "<p style='color:#9ca3af; font-size:0.88rem;'>"
        "Enter patient clinical data to receive an AI-powered cardiac risk assessment."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Patient Biometrics ──────────────────────────────────
    st.markdown('<div class="section-header">📋 Patient Biometrics</div>', unsafe_allow_html=True)

    age = st.slider("Age", min_value=20, max_value=90, value=45, help="Patient age in years")
    sex = st.selectbox("Sex", options=list(SEX_MAP.keys()))
    cp = st.selectbox(
        "Chest Pain Type",
        options=list(CP_MAP.keys()),
        help="Type of chest pain experienced",
    )
    trestbps = st.slider(
        "Resting Blood Pressure (mmHg)",
        min_value=80, max_value=200, value=120,
        help="Resting blood pressure on admission",
    )
    chol = st.slider(
        "Cholesterol (mg/dL)",
        min_value=100, max_value=600, value=200,
        help="Serum cholesterol level",
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        options=list(FBS_MAP.keys()),
    )
    restecg = st.selectbox(
        "Resting ECG",
        options=list(RESTECG_MAP.keys()),
        help="Resting electrocardiographic results",
    )

    st.markdown("---")

    # ── Lifestyle Factors ───────────────────────────────────
    st.markdown('<div class="section-header">🏃 Lifestyle Factors</div>', unsafe_allow_html=True)

    thalach = st.slider(
        "Max Heart Rate Achieved",
        min_value=60, max_value=220, value=150,
        help="Maximum heart rate during exercise",
    )
    exang = st.selectbox(
        "Exercise Induced Angina",
        options=list(EXANG_MAP.keys()),
    )
    oldpeak = st.slider(
        "ST Depression (Oldpeak)",
        min_value=0.0, max_value=6.0, value=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest",
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST",
        options=list(SLOPE_MAP.keys()),
    )
    ca = st.slider(
        "Number of Major Vessels (0-4)",
        min_value=0, max_value=4, value=0,
        help="Number of major vessels colored by fluoroscopy",
    )
    thal = st.selectbox(
        "Thalassemia",
        options=list(THAL_MAP.keys()),
        help="Thalassemia type",
    )

    st.markdown("---")
    analyze_btn = st.button("🔬 Analyze Risk", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# MAIN AREA — Header
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <h1 style="font-size:2.6rem; font-weight:700; margin-bottom:0;
                    background: linear-gradient(90deg, #a78bfa, #6366f1, #818cf8);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🫀 Longevity Risk Analyzer
        </h1>
        <p style="color:#9ca3af; font-size:1.1rem; margin-top:4px;">
            Preventive Health Early Warning System
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Pre-analysis overview cards ─────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <h3>AI-Powered</h3>
            <div class="value">🧠</div>
            <div class="sub">Random Forest Classifier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <h3>Clinical Features</h3>
            <div class="value">13</div>
            <div class="sub">UCI Heart Disease Dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <h3>MLOps Pipeline</h3>
            <div class="value">⚙️</div>
            <div class="sub">MLflow · Docker · Kubernetes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────

if analyze_btn:
    # Check model availability
    if not os.path.exists(MODEL_PATH):
        st.warning(
            "⚠️ **Model not found.** Please train the model first by running "
            "`python src/train.py`. Expected path: `models/longevity_model.pkl`"
        )
        st.stop()

    # Map inputs to numeric values
    inputs = {
        "age": age,
        "sex": SEX_MAP[sex],
        "cp": CP_MAP[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": FBS_MAP[fbs],
        "restecg": RESTECG_MAP[restecg],
        "thalach": thalach,
        "exang": EXANG_MAP[exang],
        "oldpeak": oldpeak,
        "slope": SLOPE_MAP[slope],
        "ca": ca,
        "thal": THAL_MAP[thal],
    }

    feature_array = np.array([[inputs[f] for f in FEATURE_ORDER]])

    # Load artifacts
    with st.spinner("Loading model and running inference…"):
        try:
            model, scaler, threshold = load_artifacts()
        except Exception as exc:
            st.error(f"❌ Failed to load model artifacts: {exc}")
            st.stop()

        # Scale features if scaler is available
        if scaler is not None:
            feature_array = scaler.transform(feature_array)

        # Predict
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(feature_array)[0][1])
        else:
            prob = float(model.predict(feature_array)[0])

    is_high_risk = prob >= threshold

    # ── Results Section ─────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:16px;">
            <h2 style="color:#e0e0e0; font-weight:600;">📊 Risk Assessment Results</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Probability + Risk Bar ──────────────────────────────
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        prob_pct = prob * 100
        bar_class = risk_color_class(prob)
        st.markdown(
            f"""
            <div class="glass-card" style="text-align:center;">
                <h3 style="color:#a78bfa; font-size:0.9rem; text-transform:uppercase;
                           letter-spacing:1px; margin-bottom:8px;">
                    Risk Probability
                </h3>
                <div style="font-size:3.2rem; font-weight:700; color:#ffffff;">
                    {prob_pct:.1f}%
                </div>
                <div class="risk-bar-track">
                    <div class="risk-bar-fill {bar_class}"
                         style="width:{prob_pct:.1f}%;"></div>
                </div>
                <p style="color:#9ca3af; font-size:0.82rem; margin-top:8px;">
                    Decision threshold: <strong>{threshold:.2f}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        if is_high_risk:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="verdict-high">
                        <div class="verdict-text">⚠️ HIGH RISK</div>
                        <p style="color:#fca5a5; margin-top:8px; font-size:0.95rem;">
                            Consult a physician immediately.<br>
                            Multiple cardiac risk indicators are elevated.
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="verdict-low">
                        <div class="verdict-text">✅ LOW RISK</div>
                        <p style="color:#86efac; margin-top:8px; font-size:0.95rem;">
                            Keep up the healthy habits!<br>
                            Continue regular check-ups and a balanced lifestyle.
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Risk Factors Breakdown ──────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    factors_col, recs_col = st.columns(2)

    with factors_col:
        risk_factors = get_risk_factors(inputs)
        st.markdown(
            '<div class="glass-card">'
            '<h3 style="color:#a78bfa; font-size:0.9rem; text-transform:uppercase; '
            'letter-spacing:1px; margin-bottom:14px;">🔍 Contributing Risk Factors</h3>',
            unsafe_allow_html=True,
        )
        if risk_factors:
            for factor in risk_factors:
                st.markdown(
                    f'<div class="rec-item">{factor}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="rec-item">✅ No significant risk factors identified.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with recs_col:
        recommendations = get_recommendations(prob, inputs)
        st.markdown(
            '<div class="glass-card">'
            '<h3 style="color:#a78bfa; font-size:0.9rem; text-transform:uppercase; '
            'letter-spacing:1px; margin-bottom:14px;">💡 Recommendations</h3>',
            unsafe_allow_html=True,
        )
        for rec in recommendations:
            st.markdown(
                f'<div class="rec-item">{rec}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Feature Summary Table ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 View Full Feature Input Summary", expanded=False):
        summary_data = {
            "Feature": [
                "Age", "Sex", "Chest Pain Type", "Resting BP",
                "Cholesterol", "Fasting Blood Sugar > 120", "Resting ECG",
                "Max Heart Rate", "Exercise Angina", "ST Depression",
                "ST Slope", "Major Vessels", "Thalassemia",
            ],
            "Value": [
                age, sex, cp, f"{trestbps} mmHg",
                f"{chol} mg/dL", fbs, restecg,
                f"{thalach} bpm", exang, oldpeak,
                slope, ca, thal,
            ],
            "Encoded": [inputs[f] for f in FEATURE_ORDER],
        }
        st.table(summary_data)

# ─────────────────────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="disclaimer">
        <strong>⚕️ Medical Disclaimer:</strong> This tool is for <em>educational and
        research purposes only</em>. It is <strong>not</strong> a substitute for
        professional medical advice, diagnosis, or treatment. The predictions are
        generated by a machine-learning model trained on the UCI Heart Disease dataset
        and may not generalize to all populations. Always seek the advice of a
        qualified healthcare provider with any questions regarding a medical condition.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align:center; padding:24px 0 8px 0; color:#4b5563; font-size:0.75rem;">
        Built with ❤️ using Streamlit · MLflow · Scikit-learn · XGBoost<br>
        Longevity Risk MLOps Project © 2024
    </div>
    """,
    unsafe_allow_html=True,
)
