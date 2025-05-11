import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#3.Simple relation analysis
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
#Set Chinese character for plotting
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False 
#1.Linear analysis (Pearson)
plt.figure(figsize=(16, 12))
corr_matrix = data.corr()
thrust_corr = corr_matrix[['推力/kN']].sort_values('推力/kN', ascending=False)
print("\n相关系数：\n",thrust_corr)
sns.heatmap(thrust_corr, annot=True, cmap='coolwarm')
plt.title("Pearson factors of thrust with virables")
plt.savefig("Pearson_Factors.png")
plt.close()
#Get the factors that have high correlation with thrust
strong_feats = thrust_corr[thrust_corr.abs() > 0.8] \
                    .drop(labels=['推力/kN']) \
                    .dropna() \
                    .index.tolist()
print("\n强相关变量：\n", strong_feats)
#drop the factors not in strong_feats
data = data[strong_feats+['推力/kN']+['时间']]
print(data.head())
data.to_csv('linear_strong_factors.csv',index=False)
