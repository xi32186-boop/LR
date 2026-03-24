import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# ==============================
# Page settings
# ==============================
st.set_page_config(page_title="Cognitive Aging Risk Prediction", layout="centered")

# ==============================
# Title
# ==============================
st.title("Cognitive Aging Acceleration Risk Prediction")
st.write("Please enter the following information, then click 'Predict' to get the risk probability and SHAP interpretation.")

# ==============================
# Load model
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load("lr_simplified_binary_model.pkl")
    scaler = joblib.load("lr_simplified_scaler.pkl")
    return model, scaler

lr_model, scaler = load_model()

# ==============================
# Feature info
# ==============================
feature_names = ["age", "glu", "wbc", "hb", "plt", "alt"]

feature_labels = {
    "age": "Age (years)",
    "glu": "Glucose (mmol/L)",
    "wbc": "White Blood Cells (10^9/L)",
    "hb": "Hemoglobin (g/L)",
    "plt": "Platelets (10^9/L)",
    "alt": "ALT (U/L)"
}

# ==============================
# Input section（参考你给的格式）
# ==============================
st.subheader("Enter Clinical Information:")

# session_state 保持输入
if "user_input" not in st.session_state:
    st.session_state.user_input = {f: 0.0 for f in feature_names}

for f in feature_names:
    st.session_state.user_input[f] = st.number_input(
        feature_labels[f],
        value=st.session_state.user_input.get(f, 0.0),
        key=f
    )

# DataFrame
input_df = pd.DataFrame([st.session_state.user_input])

# ==============================
# Prediction
# ==============================
if st.button("Predict"):

    # 保留1位小数（展示更专业）
    input_df = input_df.round(1)

    # 标准化
    input_scaled = scaler.transform(input_df)

    # 预测概率
    prob = lr_model.predict_proba(input_scaled)[0, 1]

    st.subheader("Prediction Results:")

    # 概率
    st.markdown(
        f'<h4>Predicted Risk of Cognitive Aging Acceleration: {prob:.2%}</h4>',
        unsafe_allow_html=True
    )

    # ==============================
    # Risk level
    # ==============================
    if prob < 0.3:
        st.markdown('<h4 style="color:green;">Risk Level: Low Risk</h4>', unsafe_allow_html=True)
    elif 0.3 <= prob <= 0.6:
        st.markdown('<h4 style="color:orange;">Risk Level: Moderate Risk</h4>', unsafe_allow_html=True)
    else:
        st.markdown('<h4 style="color:red;">Risk Level: High Risk</h4>', unsafe_allow_html=True)

    # ==============================
    # SHAP
    # ==============================
    st.subheader("SHAP Interpretation:")
    st.write("The figure below shows how each feature contributes to the prediction:")

    explainer = shap.LinearExplainer(lr_model, input_scaled)
    shap_values = explainer.shap_values(input_scaled)

    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_df.iloc[0],   # ⭐ 用原始值展示
        feature_names=[feature_labels[f] for f in feature_names]
    )

    st.components.v1.html(
        f"""
        <head>{shap.getjs()}</head>
        <body>{force_plot.html()}</body>
        """,
        height=350
    )
