# predictor.py
import streamlit as st
import pandas as pd
import joblib
import shap

# ==============================
# 1️⃣ Load model
# ==============================
lr_model = joblib.load("lr_simplified_binary_model.pkl")
scaler = joblib.load("lr_simplified_scaler.pkl")

# ==============================
# 2️⃣ Feature info
# ==============================
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
# 3️⃣ Page UI
# ==============================
st.set_page_config(page_title="Cognitive Aging Risk", layout="centered")

st.title("Cognitive Aging Acceleration Risk Prediction")

st.write("Please enter the following information:")

# ==============================
# 4️⃣ Input (无 +/- 按钮)
# ==============================
user_input = {}

cols = st.columns(2)

for i, (label, col) in enumerate(features_info.items()):
    with cols[i % 2]:
        user_input[col] = st.text_input(label, value="")

# ==============================
# 5️⃣ Predict button
# ==============================
if st.button("Predict"):

    try:
        # 转 float
        input_values = {k: float(v) for k, v in user_input.items()}
        input_df = pd.DataFrame([input_values])

        # 保留1位小数
        input_df = input_df.round(1)

        # 标准化
        input_scaled = scaler.transform(input_df[features_ordered])

        # ==============================
        # 6️⃣ Prediction
        # ==============================
        prob = lr_model.predict_proba(input_scaled)[0, 1]

        st.subheader("Result")
        st.write(f"Probability of Cognitive Aging Acceleration: **{prob * 100:.1f}%**")

        # ==============================
        # 7️⃣ SHAP Force Plot
        # ==============================
        st.subheader("SHAP Explanation")

        explainer = shap.LinearExplainer(lr_model, input_scaled)
        shap_values = explainer.shap_values(input_scaled)

        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df.iloc[0],
            feature_names=list(features_info.keys())
        )

        st.components.v1.html(
            f"""
            <head>{shap.getjs()}</head>
            <body>{force_plot.html()}</body>
            """,
            height=350
        )

    except:
        st.error("Please enter valid numeric values.")
