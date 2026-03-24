# predictor.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ModuleNotFoundError:
    import pip
    pip.main(['install','joblib'])
    import joblib

# ==============================
# 1️⃣ Load model and scaler
# ==============================
lr_model = joblib.load("lr_simplified_binary_model.pkl")
scaler = joblib.load("lr_simplified_scaler.pkl")

# Simplified features with labels and units
features_info = {
    'Age (years)': 'age',
    'Glucose (mmol/L)': 'glu',
    'White Blood Cells (10^9/L)': 'wbc',
    'Hemoglobin (g/L)': 'hb',
    'Platelets (10^9/L)': 'plt',
    'ALT (U/L)': 'alt'
}

features_ordered = list(features_info.values())

st.set_page_config(page_title="Simplified Cognitive Aging Prediction", layout="centered")
st.title("🔹 Simplified Cognitive Aging Acceleration Risk Prediction")

# ==============================
# 2️⃣ User input interface
# ==============================
st.sidebar.header(
    "Please fill in the information below and click Predict to get the Cognitive Aging Acceleration Risk Assessment"
)
user_input = {}
for label, col in features_info.items():
    if col == 'age':
        user_input[col] = st.sidebar.number_input(label, min_value=20, max_value=120, value=50)
    elif col in ['wbc', 'hb', 'plt']:
        user_input[col] = st.sidebar.number_input(label, min_value=0.0, value=1.0, format="%.1f")
    else:
        user_input[col] = st.sidebar.number_input(label, min_value=0.0, value=5.0, format="%.2f")

input_df = pd.DataFrame([user_input])

# ==============================
# 3️⃣ Standardize input
# ==============================
input_scaled = scaler.transform(input_df[features_ordered])

# ==============================
# 4️⃣ Logistic Regression prediction
# ==============================
prob_accel = lr_model.predict_proba(input_scaled)[0, 1]
st.subheader("Prediction Result")
st.write(f"📈 Probability of Cognitive Aging Acceleration: **{prob_accel * 100:.1f}%**")

# ==============================
# 5️⃣ SHAP feature explanation
# ==============================
with st.expander("🔍 SHAP Feature Contribution Explanation"):
    explainer = shap.LinearExplainer(lr_model, input_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(input_scaled)

    st.write(
        "Each feature's contribution to acceleration probability (positive values increase risk, negative values decrease risk)")
    shap_df = pd.DataFrame({
        'Feature': list(features_info.keys()),
        'SHAP value': shap_values[0]
    }).sort_values(by='SHAP value', key=abs, ascending=False)
    st.dataframe(shap_df)

    # Bar plot
    plt.figure(figsize=(6, 4))
    shap.bar_plot(shap_values[0], feature_names=list(features_info.keys()))
    st.pyplot(plt)
