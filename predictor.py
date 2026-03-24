# predictor_xgb_10features.py
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
# 2️⃣ Feature definition
# ==============================
# 升高认知衰老风险 (高 → 升高风险)
risk_features = ['sbp', 'dbp', 'hr', 'alt', 'wbc', 'rbc', 'hb']
# 减缓认知衰老风险 (高 → 降低风险)
protective_features = ['plt', 'zhixing', 'sleeptime']

# 合并为总特征
feature_names = risk_features + protective_features

# 对应网页显示标签
feature_labels = {
    'sbp': 'Systolic BP (mmHg)',
    'dbp': 'Diastolic BP (mmHg)',
    'hr': 'Heart Rate (bpm)',
    'alt': 'ALT (U/L)',
    'wbc': 'White Blood Cells (10^9/L)',
    'rbc': 'Red Blood Cells (10^12/L)',
    'hb': 'Hemoglobin (g/L)',
    'plt': 'Platelets (10^9/L)',
    'zhixing': 'Executive Function Score',
    'sleeptime': 'Sleep Time (hours)'
}

# ==============================
# 3️⃣ Streamlit page setup
# ==============================
st.set_page_config(page_title="Cognitive Aging Prediction", layout="wide")
st.title("Cognitive Aging Acceleration Risk Prediction")

# ==============================
# 4️⃣ Initialize session state (输入保持)
# ==============================
if 'user_input' not in st.session_state:
    st.session_state.user_input = {f: "" for f in feature_names}  # 初始空白

# ==============================
# 5️⃣ Layout: 左右列 + 留白
# 页面分 4 列：左空白 1/6, 左列 1/3, 右列 1/3, 右空白 1/6
# ==============================
empty1, col_left, col_right, empty2 = st.columns([1, 2, 2, 1])

# 左列：前 5 个特征
with col_left:
    for feature in feature_names[:5]:
        st.session_state.user_input[feature] = st.text_input(
            feature_labels[feature],
            value=st.session_state.user_input.get(feature, ""),
            key=feature
        )

# 右列：后 5 个特征
with col_right:
    for feature in feature_names[5:]:
        st.session_state.user_input[feature] = st.text_input(
            feature_labels[feature],
            value=st.session_state.user_input.get(feature, ""),
            key=feature
        )

# ==============================
# 6️⃣ Prediction & SHAP
# ==============================
# 转为 DataFrame
input_dict = {}
for f in feature_names:
    try:
        input_dict[f] = float(st.session_state.user_input[f])
    except:
        input_dict[f] = np.nan  # 空白或非法值

input_df = pd.DataFrame([input_dict])

# 按钮预测
if st.button("Predict"):
    if input_df.isnull().any().any():
        st.error("Please fill in all feature values with valid numbers.")
        st.stop()

    # 预测概率
    prob = xgb_model.predict_proba(input_df)[0, 1]

    # 显示预测结果
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

    # 处理 expected_value
    ev = explainer.expected_value
    if isinstance(ev, np.ndarray) and len(ev) > 1:
        ev = ev[1]  # 正类

    # 单样本 force plot
    force_plot = shap.force_plot(
        ev,
        shap_values.values[0],
        input_df,
        feature_names=[feature_labels[f] for f in feature_names]
    )

    st.components.v1.html(
        f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
        height=400
    )
