import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 3. Simple relation analysis
data = pd.read_csv(r'D:\file\Research\research\Senior_Thesis\cleaned_full.csv')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(data.drop(columns=['时间']))
fig=plt.figure(figsize=(16,12))
plt.xlabel("Time")
plt.ylabel("Engine Factor")
plt.title("Factors vs Time")
ax1=fig.add_subplot(211)
ax1.plot(data["时间"],X[:,-4],label=data.columns[-4])
ax1.plot(data["时间"],X[:,-5],label=data.columns[-5])
ax1.legend()
ax2=fig.add_subplot(212)
for i in range(22) :
    ax2.plot(data["时间"],X[:,i],label=data.columns[i+1])
ax2.plot(data["时间"],X[:,-1],label=data.columns[-1])
ax2.plot(data["时间"],X[:,-2],label=data.columns[-2])
ax2.plot(data["时间"],X[:,-3],label=data.columns[-3])
ax2.legend()
plt.savefig("Factors_vs_Time")
plt.close()
plt.show()

# Set Chinese character for plotting
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False 

# 1. Linear analysis (Pearson)
plt.figure(figsize=(16, 12))
corr_matrix = data.corr()
thrust_corr = corr_matrix[['推力/kN']].sort_values('推力/kN', ascending=False)
print("\n相关系数：\n",thrust_corr)
sns.heatmap(thrust_corr, annot=True, cmap='coolwarm')
plt.title("Pearson factors of thrust with variables")
plt.savefig("Pearson_Factors.png")
plt.close()

# 2. 非线性相关性分析（如Spearman、Kendall、互信息）
# Spearman相关性
spearman_corr = data.corr(method='spearman')[['推力/kN']].sort_values('推力/kN', ascending=False)
print("\nSpearman相关系数：\n", spearman_corr)
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm')
plt.title("Spearman factors of thrust with variables")
plt.savefig("Spearman_Factors.png")
plt.close()

# Kendall相关性
kendall_corr = data.corr(method='kendall')[['推力/kN']].sort_values('推力/kN', ascending=False)
print("\nKendall相关系数：\n", kendall_corr)
sns.heatmap(kendall_corr, annot=True, cmap='coolwarm')
plt.title("Kendall factors of thrust with variables")
plt.savefig("Kendall_Factors.png")
plt.close()

# 互信息（非线性相关性）
from sklearn.feature_selection import mutual_info_regression
import numpy as np

features = data.drop(columns=['推力/kN', '时间'])
target = data['推力/kN']
mi = mutual_info_regression(features, target, random_state=0)
mi_series = pd.Series(mi, index=features.columns).sort_values(ascending=False)
print("\n互信息（Mutual Information）：\n", mi_series)
plt.figure(figsize=(12, 8))
sns.barplot(x=mi_series.values, y=mi_series.index)
plt.title("Mutual Information with Thrust")
plt.xlabel("Mutual Information")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("Mutual_Information_Factors.png")
plt.close()

# Get the factors that have high correlation with thrust (Pearson)
strong_feats = thrust_corr[thrust_corr.abs() > 0.8] \
                    .drop(labels=['推力/kN']) \
                    .dropna() \
                    .index.tolist()
print("\n强相关变量（Pearson > 0.8）：\n", strong_feats)
# drop the factors not in strong_feats
data = data[strong_feats+['推力/kN']+['时间']]
print(data.head())
data.to_csv('linear_strong_factors.csv',index=False)
