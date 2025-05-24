import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 3. Simple relation analysis
# Set Chinese font for plotting to avoid missing glyph warnings
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

data_raw = pd.read_csv(r'D:\file\Research\research\Senior_Thesis\DataSets\cleaned_raw.csv')
data_std = pd.read_csv(r'D:\file\Research\research\Senior_Thesis\DataSets\cleaned_std.csv')

# Use standardized data for plotting
factor_names = data_std.columns[1:-1]  # Exclude first and last columns if needed

# Plot each factor vs thrust in a separate subplot (2x3 grid as example)
num_factors = len(factor_names)
cols = 3
rows = (num_factors + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
axes = axes.flatten()

for i, name in enumerate(factor_names):
    axes[i].plot(data_std[name], data_std['推力/kN'], 'o', alpha=0.6)
    axes[i].set_xlabel(name)
    axes[i].set_ylabel('推力/kN')
    axes[i].set_title(f"{name} vs 推力/kN")

# Hide unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("Figures/Factors_vs_Thrust_Subplots.png")
plt.close()
# Plot each factor and thrust vs index in a separate subplot (2x3 grid as example)

fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
axes = axes.flatten()

for i, name in enumerate(factor_names):
    axes[i].plot(data_std.index, data_std[name], label=name)
    axes[i].plot(data_std.index, data_std['推力/kN'], label='推力/kN')
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel('Value')
    axes[i].set_title(f"{name} & 推力/kN vs Index")
    axes[i].legend()

# Hide unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("Figures/Factors_and_Thrust_vs_Index_Subplots.png")
plt.close()


# # Set Chinese character for plotting
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
# plt.rcParams['axes.unicode_minus'] = False 

# 1. Linear analysis (Pearson)
plt.figure(figsize=(20, 12))
corr_matrix = data_raw.corr()
thrust_corr = corr_matrix[['推力/kN']].sort_values('推力/kN', ascending=False)
print("\nPearson Correlation Coefficients：\n",thrust_corr)
sns.heatmap(thrust_corr, annot=True, cmap='coolwarm')
plt.title("Pearson factors of thrust with variables")
plt.savefig("Figures/Pearson_Factors.png")
plt.close()

# 2. Unlinear analysis
# Spearman
plt.figure(figsize=(20, 12))
spearman_corr = data_raw.corr(method='spearman')[['推力/kN']].sort_values('推力/kN', ascending=False)
print("\nSpearman Correlation Coefficients：\n", spearman_corr)
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm')
plt.title("Spearman factors of thrust with variables")
plt.savefig("Figures/Spearman_Factors.png")
plt.close()

# Kendall
kendall_corr = data_raw.corr(method='kendall')[['推力/kN']].sort_values('推力/kN', ascending=False)
print("\nKendall Correlation Coefficients：\n", kendall_corr)
sns.heatmap(kendall_corr, annot=True, cmap='coolwarm')
plt.title("Kendall factors of thrust with variables")
plt.savefig("Figures/Kendall_Factors.png")
plt.close()

# Mutual Information
from sklearn.feature_selection import mutual_info_regression
import numpy as np

features = data_raw.drop(columns=['推力/kN', '时间'])
target = data_raw['推力/kN']
mi = mutual_info_regression(features, target, random_state=0)
mi_series = pd.Series(mi, index=features.columns).sort_values(ascending=False)
print("\nMutual Information：\n", mi_series)
plt.figure(figsize=(12, 8))
sns.barplot(x=mi_series.values, y=mi_series.index)
plt.title("Mutual Information with Thrust")
plt.xlabel("Mutual Information")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("Figures/Mutual_Information_Factors.png")
plt.close()

# 3. Feature selection
uncorrelation = ["环境总压", "环境总温/K"]
threshold_mi = 3.5
threshold_corr = 0.6
selected_features = mi_series[(mi_series > threshold_mi)].index.tolist()

# 用数据中找出这些变量的相关系数
for col in selected_features:
    if abs(data_raw[col].corr(data_raw['推力/kN'])) < threshold_corr or \
       abs(data_raw[col].corr(data_raw['推力/kN'], method='spearman')) < threshold_corr:
        uncorrelation.append(col)

print("无关变量：", uncorrelation)

# 保留相关变量
related_features = [col for col in data_raw.columns if col not in uncorrelation]

# 保存raw和std两个文件
data_raw[related_features].to_csv("DataSets/related_data_raw.csv", index=False)
data_std[related_features].to_csv("DataSets/related_data_std.csv", index=False)
