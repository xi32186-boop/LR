# predictor_xgb_10features_button.py
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# ==============================
# 1️⃣ Load XGBoost model
# ==============================
xgb_model = joblib.load("xgb_directional_10features_model.pkl")  # 你训练好的模型

# ==============================
# 2️⃣ Define features and labels
# ==============================
risk_features = ['age', 'sbp', 'dbp', 'hr', 'alt']
protective_features = ['wbc', 'rbc', 'hb', 'plt', 'zhixing']
features = risk_features + protective_features

feature_labels = {
    'age': 'Age (years)',
    'sbp': 'Systolic BP (mmHg)',
    'dbp': 'Diastolic BP (mmHg)',
    'hr': 'Heart Rate (bpm)',
    'alt': 'ALT (U/L)',
    'wbc': 'White Blood Cells (10^9/L)',
    'rbc': 'Red Blood Cells (10^12/L)',
    'hb': 'Hemoglobin (g/L)',
    'plt': 'Platelets (10^9/L)',
    'zhixing': 'Executive Function Score'
}

# ==============================
# 3️⃣ Streamlit page settings
# ==============================
st.set_page_config(page_title="Cognitive Aging Prediction", layout="wide")
st.markdown("<h1 style='text-align: center;'>Cognitive Aging Acceleration Risk Prediction</h1>", unsafe_allow_html=True)

# ==============================
# 4️⃣ Initialize session state
# ==============================
if 'user_input' not in st.session_state:
    st.session_state.user_input = {f: None for f in features}

# ==============================
# 5️⃣ Layout: 左右对称输入
# ==============================
empty1, col_left, col_right, empty2 = st.columns([1, 2, 2, 1])

# 左列：risk_features
with col_left:
    for feature in risk_features:
        st.session_state.user_input[feature] = st.number_input(
            feature_labels[feature],
            value=st.session_state.user_input.get(feature, None),
            key=feature,
            format="%.1f",
            step=None  # 去掉正负箭头
        )

# 右列：protective_features
with col_right:
    for feature in protective_features:
        st.session_state.user_input[feature] = st.number_input(
            feature_labels[feature],
            value=st.session_state.user_input.get(feature, None),
            key=feature,
            format="%.1f",
            step=None  # 去掉正负箭头
        )

# ==============================
# 6️⃣ Predict button
# ==============================
if st.button("Predict"):

    # 填充空值为0或正常值
    input_values = {f: (st.session_state.user_input[f] if st.session_state.user_input[f] is not None else 0.0)
                    for f in features}
    input_df = pd.DataFrame([input_values], columns=features)
    input_df = input_df.astype(np.float64)

    # 预测
    prob = xgb_model.predict_proba(input_df)[0, 1]

    # 显示结果
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
    # SHAP解释
    # ==============================
    st.subheader("SHAP Feature Contribution")
    st.write("Red = increases risk, Blue = decreases risk")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(input_df)

    ev = explainer.expected_value
    if isinstance(ev, np.ndarray) and len(ev) > 1:
        ev = ev[1]  # 正类

    force_plot = shap.force_plot(
        ev,
        shap_values.values[0],
        input_df,
        feature_names=[feature_labels[f] for f in features]
    )

    st.components.v1.html(
        f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
        height=400
    )
