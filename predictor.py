# predictor_fused.py
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
st.sidebar.header("Patient Information Input")

user_input = {}
for label, col in features_info.items():
    if col == 'age':
        # integer, empty initial display
        user_input[col] = st.sidebar.number_input(label, min_value=20, max_value=120, value=0, step=1, format="%d")
    elif col in ['wbc', 'hb', 'plt']:
        # float with 1 decimal
        user_input[col] = st.sidebar.number_input(label, min_value=0.0, value=0.0, format="%.1f")
    else:
        # float with 2 decimals
        user_input[col] = st.sidebar.number_input(label, min_value=0.0, value=0.0, format="%.2f")

input_df = pd.DataFrame([user_input])

# ==============================
# 4️⃣ Standardize input
# ==============================
input_scaled = scaler.transform(input_df[features_ordered])

# ==============================
# 5️⃣ Prediction
# ==============================
if st.button("Predict"):

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
    with st.expander("🔍 SHAP Feature Contribution Explanation"):
        explainer = shap.LinearExplainer(lr_model, input_scaled, feature_perturbation="interventional")
        shap_values = explainer.shap_values(input_scaled)

        # SHAP dataframe
        shap_df = pd.DataFrame({
            'Feature': list(features_info.keys()),
            'SHAP value': shap_values[0]
        }).sort_values(by='SHAP value', key=abs, ascending=False)
        st.dataframe(shap_df)

        # -------------------------
        # Bar plot (matplotlib)
        # -------------------------
        plt.figure(figsize=(6, 4))
        shap.bar_plot(shap_values[0], feature_names=list(features_info.keys()))
        st.pyplot(plt)

        # -------------------------
        # Waterfall plot (publication)
        # -------------------------
        st.write("Waterfall Plot (Recommended for Publication)")
        shap.plots.waterfall(shap_values[0])
        st.pyplot(plt.gcf())

        # -------------------------
        # HTML interactive force plot
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
