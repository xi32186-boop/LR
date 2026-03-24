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

feature_units = {
    "age": "(years)",
    "glu": "(mmol/L)",
    "wbc": "(10^9/L)",
    "hb": "(g/L)",
    "plt": "(10^9/L)",
    "alt": "(U/L)"
}

# ==============================
# Input fields
# ==============================
st.subheader("Enter Laboratory Test Results:")

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = {feature: 0.0 for feature in feature_names}

for feature in feature_names:
    label = f"{feature} {feature_units.get(feature, '')}"
    st.session_state.user_input[feature] = st.number_input(
        label,
        value=st.session_state.user_input.get(feature, 0.0),
        key=feature
    )

# Convert input to DataFrame
input_df = pd.DataFrame([st.session_state.user_input])

# ==============================
# Prediction
# ==============================
if st.button("Predict"):

    # 标准化输入
    input_scaled = scaler.transform(input_df)

    # 预测概率
    pred_prob = model.predict_proba(input_scaled)[0, 1]

    st.subheader("Prediction Results:")

    # Display probability
    st.markdown(
        f'<h4 style="color:black;">Predicted Risk of Cognitive Aging Acceleration: {pred_prob:.2%}</h2>',
        unsafe_allow_html=True
    )

    # Display risk level
    if pred_prob < 0.3:
        st.markdown('<h4 style="color:green;">Risk Level: Low Risk</h4>', unsafe_allow_html=True)
    elif 0.3 <= pred_prob <= 0.6:
        st.markdown('<h4 style="color:orange;">Risk Level: Moderate Risk</h4>', unsafe_allow_html=True)
    else:
        st.markdown('<h4 style="color:red;">Risk Level: High Risk</h4>', unsafe_allow_html=True)

    # SHAP interpretation
    st.subheader("SHAP Interpretation:")
    st.write("The figure below shows how each feature pushes the model output:")

    explainer = shap.LinearExplainer(model, input_scaled)
    shap_values = explainer.shap_values(input_scaled)

    # Force plot
    shap.initjs()
    st_shap = shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=True)

    fig = plt.gcf()
    st.pyplot(fig)
