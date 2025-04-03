import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('cleaned.csv',header=0)
#############################################################################################
#Correlation analysis
#############################################################################################
#Set Chinese character for plotting
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False 
#1.Linear analysis (Pearson)
plt.figure(figsize=(16, 12))
corr_matrix = data.corr()
sns.heatmap(corr_matrix[['推力/kN']].sort_values('推力/kN', ascending=False), 
            annot=True, cmap='coolwarm')
plt.title("Pearson factors of thrust with virables")
plt.show()

#############################################################################################
#Model construction
#############################################################################################
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#1.Standarlize (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop(columns=['时间', '推力/kN']))
X = data.drop(columns=['时间', '推力/kN'])
y = data['推力/kN']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差 (MSE): {mse}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("实际值与预测值对比")
plt.xlabel("实际推力/kN")
plt.ylabel("预测推力/kN")
plt.show()