#For the data set,
#there are 27 parameters that nay influence the thrust.
#############################################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_excel(r'D:\OneDrive\无速度无高度\无速度无高度\航空发动机高度0马赫数0仿真3.xlsx')
#############################################################################################
#Data preprocessing
#############################################################################################
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
print(f"\n缺失时间点: {missing_time} s")
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
