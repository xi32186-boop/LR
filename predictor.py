# predictor_xgb_8features_centered_sym.py
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# ==============================
# 1️⃣ Load 8-feature XGBoost model
# ==============================
xgb_model = joblib.load("xgb_8features_model.pkl")

# ==============================
# 2️⃣ Feature definition (8个特征)
# ==============================
feature_names = ['age', 'sbp', 'dbp', 'alt', 'wbc', 'rbc', 'hb', 'plt']
feature_labels = {
    'age': 'Age (years)',
    'sbp': 'Systolic BP (mmHg)',
    'dbp': 'Diastolic BP (mmHg)',
    'alt': 'ALT (U/L)',
    'wbc': 'White Blood Cells (10^9/L)',
    'rbc': 'Red Blood Cells (10^12/L)',
    'hb': 'Hemoglobin (g/L)',
    'plt': 'Platelets (10^9/L)'
}

# ==============================
# 3️⃣ Page layout (居中1/3)
# ==============================
st.set_page_config(page_title="Cognitive Aging Prediction", layout="wide")
empty_left, col_center, empty_right = st.columns([1, 1, 1])

with col_center:
    st.title("Cognitive Aging Acceleration Risk Prediction")
    st.markdown("Enter your laboratory and physiological measurements below to predict the risk of accelerated cognitive aging.")

    # 初始化 session state
    if 'user_input' not in st.session_state:
        st.session_state.user_input = {f: 0.0 for f in feature_names}

    # ==============================
    # 4️⃣ 输入框左右对称 4 个
    # ==============================
    input_left, input_right = st.columns(2)
    with input_left:
        for feature in feature_names[:4]:  # 左列4个
            st.session_state.user_input[feature] = st.number_input(
                feature_labels[feature],
                value=st.session_state.user_input.get(feature, 0.0),
                key=feature
            )
    with input_right:
        for feature in feature_names[4:]:  # 右列4个
            st.session_state.user_input[feature] = st.number_input(
                feature_labels[feature],
                value=st.session_state.user_input.get(feature, 0.0),
                key=feature
            )

    input_df = pd.DataFrame([st.session_state.user_input])

    # ==============================
    # 5️⃣ Prediction
    # ==============================
    prob = xgb_model.predict_proba(input_df)[0, 1]

    st.subheader("Prediction Result")
    st.markdown(
        f'<h4 style="color:black;">Probability of Cognitive Aging Acceleration: {prob*100:.1f}%</h4>',
        unsafe_allow_html=True
    )

    if prob < 0.3:
        st.markdown('<h4 style="color:green;">Risk Level: Low Risk</h4>', unsafe_allow_html=True)
    elif prob <= 0.6:
        st.markdown('<h4 style="color:orange;">Risk Level: Moderate Risk</h4>', unsafe_allow_html=True)
    else:
        st.markdown('<h4 style="color:red;">Risk Level: High Risk</h4>', unsafe_allow_html=True)

    # ==============================
    # 6️⃣ SHAP解释力图
    # ==============================
    st.subheader("SHAP Feature Contribution")
    st.write("Red = increases risk, Blue = decreases risk")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(input_df)

    # 处理 expected_value
    ev = explainer.expected_value
    if isinstance(ev, np.ndarray) and len(ev) > 1:
        ev = ev[1]  # 正类

    # 单样本 force plot
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
