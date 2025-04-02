import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_excel(r'D:\OneDrive\无速度无高度\无速度无高度\航空发动机高度0马赫数0仿真3.xlsx')

data=data.drop(columns=["高度", "马赫数","加力燃烧室燃油流量/g/s"])
data=data.drop(index=0).reset_index(drop=True)
print(data)
print(data.head())
print(data.describe())
print("数据类型:\n", data.dtypes)
print("\n缺失值统计:\n", data.isnull().sum())
# 计算相关系数矩阵
correlation_matrix = data["环境总压"].corr(data["推力/kN"])
print(correlation_matrix)
# # 提取推力与其他变量的相关系数
# thrust_corr = correlation_matrix["推力/kN"].sort_values(ascending=False)

# print("变量与推力的相关系数:\n", thrust_corr)