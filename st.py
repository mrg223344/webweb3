import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# === 1. 加载模型 ===
model = joblib.load("logit_model.pkl")

# 模型特征名和临床展示名的映射
feature_map = {
    "admission_age": "Age (years)",
    "wbc_idx1": "WBC (10⁹/L)",
    "rdw_idx1": "RDW (%)",
    "bun_idx1": "BUN (mg/dL)",
    "anion_gap_idx1": "AG (mEq/L)",
    "fbg_idx1": "FBG (mg/dL)",
    "charlson_comorbidity_index": "CCI (score)",
    "sirs_max": "SIRS Score",
    "oasis_avg": "OASIS Score",
    "ventilator_flag": "Mechanical Ventilation",
    "vasoactive": "Vasoactive Drugs",
    "anticoagulants_icu_used": "Anticoagulants",
    "statin_icu_used": "Statins"
}

# === 2. 页面标题 ===
st.set_page_config(page_title="28-day Mortality Risk Predictor", layout="centered")
st.title("🧪 28-day Mortality Risk Predictor")
st.markdown("This tool predicts the **28-day mortality risk** in critically ill IS patients with SIRS.")

# === 3. 输入表单 ===
with st.sidebar:
    st.header("Patient Parameters")
    input_data = {}
    input_data["admission_age"] = st.number_input(feature_map["admission_age"], 18, 100, 65)
    input_data["wbc_idx1"] = st.number_input(feature_map["wbc_idx1"], 0.1, 50.0, 10.0)
    input_data["rdw_idx1"] = st.number_input(feature_map["rdw_idx1"], 10.0, 30.0, 14.0)
    input_data["bun_idx1"] = st.number_input(feature_map["bun_idx1"], 1.0, 50.0, 18.0)
    input_data["anion_gap_idx1"] = st.number_input(feature_map["anion_gap_idx1"], 0.0, 40.0, 12.0)
    input_data["fbg_idx1"] = st.number_input(feature_map["fbg_idx1"], 40.0, 500.0, 100.0)  # mg/dL 范围
    input_data["charlson_comorbidity_index"] = st.number_input(feature_map["charlson_comorbidity_index"], 0, 20, 2)
    input_data["sirs_max"] = st.number_input(feature_map["sirs_max"], 0, 10, 2)
    input_data["oasis_avg"] = st.number_input(feature_map["oasis_avg"], 0, 80, 25)
    input_data["ventilator_flag"] = st.selectbox(feature_map["ventilator_flag"], [0, 1])
    input_data["vasoactive"] = st.selectbox(feature_map["vasoactive"], [0, 1])
    input_data["anticoagulants_icu_used"] = st.selectbox(feature_map["anticoagulants_icu_used"], [0, 1])
    input_data["statin_icu_used"] = st.selectbox(feature_map["statin_icu_used"], [0, 1])

# 转换为 DataFrame（内部用特征名，显示时替换为临床名）
df_input = pd.DataFrame([input_data])
df_display = df_input.rename(columns=feature_map)

# === 4. 显示输入表格 ===
st.subheader("📋 Input Summary")
st.dataframe(df_display, use_container_width=True)

# === 5. 模型预测 ===
if st.button("🔍 Predict"):
    prob = model.predict_proba(df_input)[0][1]

    st.subheader("📊 Prediction Result")
    st.metric("Predicted 28-day Mortality Risk", f"{prob:.2%}")

    # 可视化风险
    st.subheader("🩺 Risk Visualization")
    fig, ax = plt.subplots(figsize=(5, 1.2))
    ax.barh(["Risk"], [prob], color="red" if prob > 0.4 else "green")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    st.pyplot(fig)

    # 进度条可视化
    st.progress(prob)
