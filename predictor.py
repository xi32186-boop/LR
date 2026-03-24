# predictor_fully_blank.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

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
# 3️⃣ User input interface (sidebar)
# ==============================
st.sidebar.header("Patient Information Input (leave blank if unknown)")

# Initialize session state
if "user_input" not in st.session_state:
    st.session_state.user_input = {col: "" for col in features_ordered}

# Create text_inputs (fully blank initial)
for label, col in features_info.items():
    st.session_state.user_input[col] = st.sidebar.text_input(
        label,
        value=st.session_state.user_input.get(col, "")
    )

# ==============================
# 4️⃣ Convert input to DataFrame + validation
# ==============================
if st.button("Predict"):

    # Try converting all inputs to float
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
    # 6️⃣ SHAP explanation
    # ==============================
    with st.expander("🔍 SHAP Feature Contribution Explanation (Red=Increase Risk, Blue=Decrease Risk)"):
        explainer = shap.LinearExplainer(lr_model, input_scaled, feature_perturbation="interventional")
        shap_values = explainer.shap_values(input_scaled)

        # SHAP dataframe
        shap_df = pd.DataFrame({
            'Feature': list(features_info.keys()),
            'SHAP value': shap_values[0]
        }).sort_values(by='SHAP value', key=abs, ascending=False)
        st.dataframe(shap_df)

        # -------------------------
        # Waterfall plot (publication style)
        # -------------------------
        st.write("Waterfall Plot (Recommended for Publication)")
        shap.plots.waterfall(shap_values[0])
        st.pyplot(plt.gcf())

        # -------------------------
        # HTML interactive force plot (red/blue bars)
        # -------------------------
        st.write("Interactive SHAP Force Plot")
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df.iloc[0],
            feature_names=list(features_info.keys())
        )
        st.components.v1.html(
            f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
            height=350
        )
