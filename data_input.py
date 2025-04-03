#For the data set,
#there are 27 parameters that nay influence the thrust.
#############################################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_excel(r'D:\OneDrive\无速度无高度\无速度无高度\航空发动机高度0马赫数0仿真3.xlsx')
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
data=data.drop(columns=["高度", "马赫数","加力燃烧室燃油流量/g/s"]) #null parameters
data=data.drop(index=0).reset_index(drop=True) #text row
data = data.apply(pd.to_numeric, errors='coerce')
print(data)
print(data.head())
print(data.describe()) #data info
data = data.apply(pd.to_numeric, errors='coerce')
print("数据类型:\n", data.dtypes)
print("\n缺失值统计:\n", data.isnull().sum())
missing_row = data[data["高压压缩机5th-6th stage压比"].isnull()]
print("缺失值所在行:\n", missing_row)
missing_time = missing_row['时间'].values[0]
print(f"缺失时间点: {missing_time} s")
window = data[
    (data['时间'] >= missing_time - 0.1) & 
    (data['时间'] <= missing_time + 0.1)
]
print("缺失点附近数据:\n", window[['时间', '高压压缩机5th-6th stage压比']])
data_linear = data.interpolate(method='linear')
filled_value = data_linear.loc[data_linear['时间'] == 0.10, '高压压缩机5th-6th stage压比'].values[0]
print(f"填充值 (线性插值): {filled_value:.6f}")

#construct DatetimeIndex:
# # 假设时间单位为秒，以 1970-01-01 为起点构建时间戳
# base_time = pd.to_datetime('1970-01-01')
# data_datetime = data.copy()
# data_datetime['时间'] = base_time + pd.to_timedelta(data_datetime['时间'], unit='s')

# # 设置时间戳索引并插值
# data_timeindex = (
#     data_datetime.set_index('时间')
#     .interpolate(method='time')
#     .reset_index()
# )

# # 转换回原始时间格式
# data_timeindex['时间'] = (data_timeindex['时间'] - base_time).dt.total_seconds()

# # 提取填充值
# filled_value = data_timeindex.loc[abs(data_timeindex['时间'] - 0.10) < 1e-6, '高压压缩机5th-6th stage压比'].values[0]
# print(f"填充值 (时间插值): {filled_value:.6f}")  # 输出：1.453581

data=data_linear
print("全局缺失值统计:\n", data.isnull().sum())



#2.Handle unormal values 
# constraints = {
#     "高压压缩机1st-4th stage压比": (1.0, 5.0),          # >1
#     "高压压缩机5th-6th stage压比": (1.0, 5.0),
#     "燃烧室燃油流量/g/s": (0, 10000),       # >0
#     "低压涡轮转速": (-20000, 20000),       # allow direction change
#     "喷嘴喉道静温": (300, 2000)           # temperature >= normal
# }
# # 检测并标记异常值
# for col, (min_val, max_val) in constraints.items():
#     data[f'{col}_异常'] = (data[col] < min_val) | (data[col] > max_val)
# # 输出异常记录
# anomalies = data[data.filter(like='_异常').any(axis=1)]
# print(f"物理异常记录数: {len(anomalies)}")
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=data[['高压压缩机1st-4th stage压比']])
# plt.xticks(rotation=45)
# plt.title('关键参数箱线图（异常值检测）')
# plt.show()

data_cleaned=data

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
X_scaled = scaler.fit_transform(data_cleaned.drop(columns=['时间', '推力/kN']))
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