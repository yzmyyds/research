#For the data set,
#there are 27 parameters that nay influence the thrust.
#############################################################################################
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False
#读csv文件
data2 = pd.read_csv(r'D:\file\Research\research\Senior_Thesis\DataSets\full_data.csv')
data1 = pd.read_csv(r'D:\file\Research\research\Senior_Thesis\DataSets\data_without_trb.csv')
#############################################################################################
#Data clean
# A[原始数据] --> B[物理约束过滤]
# B --> C[Z-score/IQR检测]
# C --> D[箱线图验证]
# D --> E{是否仍有异常?}
# E -->|是| F[领域知识复核]
# E -->|否| G[相关性分析]
# G --> H[数据清理结束]
#############################################################################################
#1.Handle missing values
#Handle data without 推力燃料比
print(data1.head())
print(data1.describe()) #data info
print("数据类型:\n", data1.dtypes)
print("\n缺失值统计:\n", data1.isnull().sum())
missing_row = data1[data1["高压压缩机5th-6th stage压比"].isnull()]
print("缺失值所在行:\n", missing_row)
missing_time = missing_row['时间'].values[0]
print(f"缺失时间点: {missing_time} s")
window = data1[
    (data1['时间'] >= missing_time - 0.1) & 
    (data1['时间'] <= missing_time + 0.1)
]
print("缺失点附近数据:\n", window[['时间', '高压压缩机5th-6th stage压比']])
data1 = data1.interpolate(method='linear')
filled_value = data1.loc[data1['时间'] == 0.10, '高压压缩机5th-6th stage压比'].values[0]
print(f"填充值 (线性插值): {filled_value:.6f}")
print("全局缺失值统计:\n", data1.isnull().sum())

# 异常值检测与处理

def detect_outliers_zscore(df, col, threshold=3):
    # 若数据方差为0，zscore会产生nan，需特殊处理
    if df[col].std() == 0 or df[col].dropna().std() == 0:
        return pd.DataFrame()  # 无异常值
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    outliers = df.loc[df[col].dropna().index[z_scores > threshold]]
    return outliers

# 对data2进行Z-score异常值检测和处理
exclude_cols = ['时间', '推力']
param_cols = [col for col in data2.columns if col not in exclude_cols]

# 检测异常值
abnormal_dict = {}
for col in param_cols:
    outliers = detect_outliers_zscore(data2, col)
    abnormal_dict[col] = outliers
    print(f"{col} 的Z-score异常值数: {len(outliers)}")

# 为每个参数画单独的箱线图
os.makedirs("Boxplots", exist_ok=True)

def sanitize_filename(name):
    # Replace invalid filename characters with '_'
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

# 将所有参数的箱线图绘制在一个subplot网格中并保存
n_cols = 4
n_rows = int(np.ceil(len(param_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
axes = axes.flatten()

for idx, col in enumerate(param_cols):
    sns.boxplot(y=data2[col], ax=axes[idx])
    axes[idx].set_title(f'{col} 箱线图')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('')

# 隐藏多余的子图
for i in range(len(param_cols), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig("Figures/fulldata_boxplot_all.png")
plt.close()

# 用中位数替换异常值
for col in param_cols:
    q1 = data2[col].quantile(0.25)
    q3 = data2[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    data2.loc[(data2[col] < lower) | (data2[col] > upper), col] = data2[col].median()

print("data2异常值处理后描述统计:\n", data2[param_cols].describe())

# 基于物理约束的异常值检测
constraints = {
    "高压压缩机1st-4th stage压比": (1.0, 5.0),
    "高压压缩机5th-6th stage压比": (1.0, 5.0),
    "燃烧室燃油流量/g/s": (0, 10000),
    "低压涡轮转速": (-30000, 20000),
    "喷嘴喉道静温": (300, 2000)
}
for col, (min_val, max_val) in constraints.items():
    abn = data2[(data2[col] < min_val) | (data2[col] > max_val)]
    print(f"{col} 物理异常数: {len(abn)}")
    if not abn.empty:
        print(abn[[col]].head())
anomalies = data2[
    np.logical_or.reduce([
        (data2[col] < min_val) | (data2[col] > max_val)
        for col, (min_val, max_val) in constraints.items()
    ])
]
print(f"物理异常记录总数: {len(anomalies)}")

data2.to_csv('DataSets/cleaned_raw.csv', index=False)
scaler = StandardScaler()
data2[param_cols] = scaler.fit_transform(data2[param_cols])
print("标准化后描述统计:\n", data2[param_cols].describe())

data2.to_csv('DataSets/cleaned_std.csv',index=False)