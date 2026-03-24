import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==============================
# Page settings
# ==============================
st.set_page_config(page_title="Cognitive Aging Risk Prediction", layout="centered")

# ==============================
# Title
# ==============================
st.title("Cognitive Aging Acceleration Risk Prediction")
st.write("Please enter the patient's clinical information below, then click 'Predict' to see the risk probability and SHAP interpretation.")

# ==============================
# Load model
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load("lr_simplified_binary_model.pkl")
    scaler = joblib.load("lr_simplified_scaler.pkl")
    return model, scaler

model, scaler = load_model()

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
# Input
# ==============================
st.subheader("Enter Clinical Information:")

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

    # ⭐ 保留1位小数（论文风格）
    input_df = input_df.round(1)

    # 标准化
    input_scaled = scaler.transform(input_df)

    # 预测概率
    prob = model.predict_proba(input_scaled)[0, 1]

    st.subheader("Prediction Results:")

    # 概率
    st.markdown(
        f'<h4 style="color:black;">Predicted Risk of Cognitive Aging Acceleration: {prob:.2%}</h4>',
        unsafe_allow_html=True
    )

    # ==============================
    # Risk Level
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
    st.write("The figure below shows how each feature pushes the prediction:")

    # ⭐ 推荐方式（更标准）
    explainer = shap.Explainer(model, input_scaled)
    shap_values = explainer(input_scaled)

    # ==============================
    # ✅ 方式1：matplotlib（推荐发表）
    # ==============================
    fig = shap.plots.force(
        shap_values[0],
        matplotlib=True,
        show=False
    )

    st.pyplot(plt.gcf())

    # ==============================
    # ✅ 方式2：交互版（可选）
    # ==============================
    with st.expander("Show Interactive SHAP Plot"):
        force_plot = shap.plots.force(shap_values[0])

        st.components.v1.html(
            f"""
            <head>{shap.getjs()}</head>
            <body>{force_plot.html()}</body>
            """,
            height=300
        )
