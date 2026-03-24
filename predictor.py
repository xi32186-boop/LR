# predictor_forceplot_only.py
import streamlit as st
import pandas as pd
import joblib
import shap

# ==============================
# 1️⃣ Load model and scaler
# ==============================
lr_model = joblib.load("lr_simplified_binary_model.pkl")
scaler = joblib.load("lr_simplified_scaler.pkl")

# Features info
features_info = {
    'Age (years)': 'age',
    'Glucose (mmol/L)': 'glu',
    'White Blood Cells (10^9/L)': 'wbc',
    'Hemoglobin (g/L)': 'hb',
    'Platelets (10^9/L)': 'plt',
    'ALT (U/L)': 'alt'
}
features_ordered = list(features_info.values())

# ==============================
# 2️⃣ Page title
# ==============================
st.set_page_config(page_title="Cognitive Aging Prediction", layout="centered")
st.title("🔹 Cognitive Aging Acceleration Risk Prediction")
st.write("Enter the patient's clinical information below, then click Predict to see the risk probability and SHAP explanation.")

# ==============================
# 3️⃣ User input interface (page inputs, fully blank)
# ==============================
st.header("Patient Information Input")

if "user_input" not in st.session_state:
    st.session_state.user_input = {col: "" for col in features_ordered}

for label, col in features_info.items():
    st.session_state.user_input[col] = st.text_input(
        label,
        value=st.session_state.user_input.get(col, "")
    )

# ==============================
# 4️⃣ Prediction button
# ==============================
if st.button("Predict"):

    # Convert input to float
    input_values = {}
    for col in features_ordered:
        try:
            input_values[col] = float(st.session_state.user_input[col])
        except:
            st.error(f"⚠️ Please enter a valid number for {col}")
            st.stop()

    input_df = pd.DataFrame([input_values])

    # Standardize
    input_scaled = scaler.transform(input_df[features_ordered])

    # ==============================
    # 5️⃣ Logistic Regression prediction
    # ==============================
    prob_accel = lr_model.predict_proba(input_scaled)[0, 1]

    st.subheader("Prediction Result")
    st.write(f"📈 Probability of Cognitive Aging Acceleration: **{prob_accel * 100:.1f}%**")

    # Risk level
    if prob_accel < 0.3:
        st.markdown('<h4 style="color:green;">Risk Level: Low Risk</h4>', unsafe_allow_html=True)
    elif 0.3 <= prob_accel <= 0.6:
        st.markdown('<h4 style="color:orange;">Risk Level: Moderate Risk</h4>', unsafe_allow_html=True)
    else:
        st.markdown('<h4 style="color:red;">Risk Level: High Risk</h4>', unsafe_allow_html=True)

    # ==============================
    # 6️⃣ SHAP HTML force plot (红蓝条)
    # ==============================
    st.subheader("SHAP Feature Contribution")
    st.write("Red = increases risk, Blue = decreases risk")

    # LinearExplainer
    explainer = shap.LinearExplainer(lr_model, input_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(input_scaled)

    # Important: pass 2D arrays for single sample to show red/blue bars
    force_plot = shap.force_plot(
        explainer.expected_value,   # base value
        shap_values,                # 2D array
        input_scaled,               # 2D array of input
        feature_names=list(features_info.keys())
    )

    st.components.v1.html(
        f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
        height=400
    )
