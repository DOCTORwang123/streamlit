import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（适用于 Windows）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 设置页面标题
st.title('🦵 膝骨关节炎（KOA）发病风险预测计算器')

# 加载模型
model = joblib.load("GradientBoosting_model.pkl")

# 创建输入表单
st.sidebar.header('📋 患者信息录入')

# 输入字段
合并症数量 = st.sidebar.selectbox('合并症数量', [0, 1, 2])
性别 = st.sidebar.selectbox('性别', [1, 2])
户口 = st.sidebar.selectbox('户口', [1, 2])
教育程度 = st.sidebar.selectbox('教育程度', [0, 1])
抑郁情况 = st.sidebar.selectbox('抑郁情况', [0, 1, 2])
睡眠时长 = st.sidebar.slider('睡眠时长', 0, 15)
童年时身体状况 = st.sidebar.selectbox('童年时身体状况', [1, 2, 3, 4, 5])
握力是否下降 = st.sidebar.selectbox('握力是否下降', [1, 0])


# 将输入数据转换为 DataFrame
input_data = pd.DataFrame({
    '合并症数量': [合并症数量],
    '性别': [性别],
    '户口': [户口],
    '教育程度': [教育程度],
    '抑郁情况': [抑郁情况],
    '睡眠时长': [睡眠时长],
    '童年时身体状况': [童年时身体状况],
    '握力是否下降': [握力是否下降]
})

# 显示用户输入的数据
st.write('### 患者输入信息')
st.write(input_data)

# 预测按钮
if st.button('预测'):
    # 使用模型进行预测
    prediction = model.predict_proba(input_data)[:, 1]  # 假设模型输出概率
    st.write(f'### 预测结果')
    st.write(f'患者 KOA 概率: {prediction[0]:.2f}')

    # SHAP 解释
    st.write('### SHAP 解释')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # 绘制 SHAP 图
    shap.summary_plot(shap_values, input_data, plot_type="bar")
    st.pyplot(plt.gcf())
