# predictor_xgb_layout_centered.py
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# ==============================
# 1️⃣ Load model
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

st.set_page_config(page_title="Cognitive Aging Prediction", layout="wide")
st.title("Cognitive Aging Acceleration Risk Prediction")

# ==============================
# 3️⃣ Initialize session state
# ==============================
if 'user_input' not in st.session_state:
    st.session_state.user_input = {f: 0.0 for f in feature_names}

# ==============================
# 4️⃣ Layout: 左右列 + 留白
# ==============================
# 页面分5列：左空白 1/6，左列 1/3，右列 1/3，右空白 1/6
empty1, col_left, col_right, empty2 = st.columns([1, 2, 2, 1])

# 左列：前三个特征
with col_left:
    for feature in feature_names[:3]:
        st.session_state.user_input[feature] = st.number_input(
            feature_labels[feature],
            value=st.session_state.user_input.get(feature, 0.0),
            key=feature
        )

# 右列：后三个特征
with col_right:
    for feature in feature_names[3:]:
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
