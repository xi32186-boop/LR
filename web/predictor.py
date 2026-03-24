import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ==============================
# 1️⃣ Load model and scaler
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "lr_simplified_binary_model.pkl")
scaler_path = os.path.join(BASE_DIR, "lr_simplified_scaler.pkl")

lr_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ==============================
# 2️⃣ Simplified features
# ==============================
features = {
    "age": "years",
    "glu": "mmol/L",
    "wbc": "10^9/L",
    "hb": "g/L",
    "plt": "10^9/L",
    "alt": "U/L"
}

# ==============================
# 3️⃣ Streamlit interface
# ==============================
st.title("Cognitive Aging Acceleration Risk Prediction")

st.write(
    "Please enter the following information, then click Predict to get your cognitive aging acceleration risk."
)

user_input = {}
for feat, unit in features.items():
    user_input[feat] = st.number_input(f"{feat} ({unit})", value=0.0)

input_df = pd.DataFrame([user_input])

# Scale input
input_scaled = scaler.transform(input_df)

# ==============================
# 4️⃣ Prediction
# ==============================
if st.button("Predict"):
    pred_prob = lr_model.predict_proba(input_scaled)[:, 1][0]
    pred_class = lr_model.predict(input_scaled)[0]

    st.subheader("Prediction Result")
    st.write(f"Risk Probability: {pred_prob:.3f}")
    st.write(f"Risk Class: {'High' if pred_class == 1 else 'Low'}")

    # ==============================
    # 5️⃣ SHAP Explanation
    # ==============================
    explainer = shap.LinearExplainer(lr_model, input_scaled)
    shap_values = explainer(input_scaled)

    st.subheader("Feature Contribution (SHAP values)")
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(values=shap_values.values[0],
                                         base_values=explainer.expected_value,
                                         data=input_scaled[0],
                                         feature_names=list(features.keys())))
    st.pyplot(fig)
