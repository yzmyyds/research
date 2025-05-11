#For the data set,
#there are 27 parameters that nay influence the thrust.
#############################################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#读csv文件
data2 = pd.read_csv(r'D:\file\Research\research\Senior_Thesis\full_data.csv')
data1 = pd.read_csv(r'D:\file\Research\research\Senior_Thesis\data_without_trb.csv')
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
# data1 = data1.apply(pd.to_numeric, errors='coerce')
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

#Handle data with full data
print(data2.head())
print(data2.describe()) #data info
# data2 = data2.apply(pd.to_numeric, errors='coerce')
print("数据类型:\n", data2.dtypes)

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
data2.to_csv('cleaned_full.csv',index=False)