# predictor_kernel.py
import streamlit as st
import pandas as pd
import joblib
import shap

# 1️⃣ Load model & scaler
lr_model = joblib.load("lr_simplified_binary_model.pkl")
scaler = joblib.load("lr_simplified_scaler.pkl")

# 2️⃣ Features
features_info = {
    'Age (years)': 'age',
    'Glucose (mmol/L)': 'glu',
    'White Blood Cells (10^9/L)': 'wbc',
    'Hemoglobin (g/L)': 'hb',
    'Platelets (10^9/L)': 'plt',
    'ALT (U/L)': 'alt'
}
features_ordered = list(features_info.values())

# 3️⃣ Page
st.set_page_config(page_title="Cognitive Aging Prediction", layout="centered")
st.title("Cognitive Aging Acceleration Risk Prediction")

# 4️⃣ Input (empty by default)
if "user_input" not in st.session_state:
    st.session_state.user_input = {col: "" for col in features_ordered}

for label, col in features_info.items():
    st.session_state.user_input[col] = st.text_input(label, value="")

# 5️⃣ Predict button
if st.button("Predict"):
    # Convert to float
    input_values = {}
    for col in features_ordered:
        try:
            input_values[col] = float(st.session_state.user_input[col])
        except:
            st.error(f"Please enter a valid number for {col}")
            st.stop()
    input_df = pd.DataFrame([input_values])

    # Standardize
    input_scaled = scaler.transform(input_df[features_ordered])

    # Prediction
    prob = lr_model.predict_proba(input_scaled)[0,1]
    st.subheader("Prediction Result")
    st.write(f"Probability of Cognitive Aging Acceleration: **{prob*100:.1f}%**")
    if prob<0.3: st.markdown('<h4 style="color:green;">Low Risk</h4>', unsafe_allow_html=True)
    elif prob<=0.6: st.markdown('<h4 style="color:orange;">Moderate Risk</h4>', unsafe_allow_html=True)
    else: st.markdown('<h4 style="color:red;">High Risk</h4>', unsafe_allow_html=True)

    # SHAP explanation using KernelExplainer
    st.subheader("SHAP Feature Contribution")
    st.write("Red = increase risk, Blue = decrease risk")

    explainer = shap.KernelExplainer(lr_model.predict_proba, input_df.values)
    shap_values = explainer.shap_values(input_df.values)

    force_plot = shap.force_plot(
        explainer.expected_value[1],
        shap_values[1],
        input_df,
        feature_names=list(features_info.keys())
    )
    st.components.v1.html(
        f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
        height=400
    )
