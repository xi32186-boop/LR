# predictor_xgb_raw.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# 1️⃣ Load model (no scaler needed)
# ==============================
xgb_model = joblib.load("xgb_simplified_binary_model.pkl")

# ==============================
# 2️⃣ Feature definition
# ==============================
feature_names = ['age', 'glu', 'wbc', 'hb', 'plt', 'alt']
feature_labels = {
    'age': 'Age (years)',
    'glu': 'Glucose (mmol/L)',
    'wbc': 'White Blood Cells (10^9/L)',
    'hb': 'Hemoglobin (g/L)',
    'plt': 'Platelets (10^9/L)',
    'alt': 'ALT (U/L)'
}

# Check feature number
if len(feature_names) != 6:
    st.error("Error: Feature number is not 6. Please check your features!")
    st.stop()

# ==============================
# 3️⃣ Streamlit Input
# ==============================
st.subheader("Enter Laboratory Test Results:")

# Initialize session state to keep inputs persistent
if 'user_input' not in st.session_state:
    st.session_state.user_input = {feature: 0.0 for feature in feature_names}

for feature in feature_names:
    st.session_state.user_input[feature] = st.number_input(
        feature_labels[feature],
        value=st.session_state.user_input.get(feature, 0.0),
        key=feature
    )

# Convert input to DataFrame
input_df = pd.DataFrame([st.session_state.user_input])

# ==============================
# 4️⃣ Prediction
# ==============================
if st.button("Predict"):
    pred_prob = xgb_model.predict_proba(input_df)[0, 1]

    st.subheader("Prediction Result:")
    st.markdown(
        f'<h4 style="color:black;">Probability of Cognitive Aging Acceleration: {pred_prob*100:.1f}%</h4>',
        unsafe_allow_html=True
    )

    # Risk Level
    if pred_prob < 0.3:
        st.markdown('<h4 style="color:green;">Risk Level: Low Risk</h4>', unsafe_allow_html=True)
    elif pred_prob <= 0.6:
        st.markdown('<h4 style="color:orange;">Risk Level: Moderate Risk</h4>', unsafe_allow_html=True)
    else:
        st.markdown('<h4 style="color:red;">Risk Level: High Risk</h4>', unsafe_allow_html=True)

    # ==============================
    # 5️⃣ SHAP Interpretation
    # ==============================
    st.subheader("SHAP Feature Contribution")
    st.write("Red = increases risk, Blue = decreases risk")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(input_df)

    # Handle expected_value as scalar or array
    ev = explainer.expected_value
    if isinstance(ev, np.ndarray) and len(ev) > 1:
        ev = ev[1]  # positive class

    # Single-sample force plot
    force_plot = shap.force_plot(
        ev,
        shap_values.values[0],
        input_df,
        feature_names=list(feature_labels.values())
    )

    st.components.v1.html(
        f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
        height=400
    )
