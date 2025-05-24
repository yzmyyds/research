import pandas as pd
import os
from glob import glob

# folder_path
folder_path = r"D:\file\Research\无速度无高度\全"

# use glob to get all Excel files in the folder
file_list = glob(os.path.join(folder_path, "*.xlsx"))

# read and process each file
all_data = []
for file in file_list:
    df = pd.read_excel(file)
    df = df.drop(columns=["高度", "马赫数", "加力燃烧室燃油流量/g/s"], errors='ignore')
    df = df.drop(index=0).reset_index(drop=True)  # 去掉第一行（假设是表头说明）
    df = df.apply(pd.to_numeric, errors='coerce')
    all_data.append(df)

# fuse all data into a single DataFrame
merged_data = pd.concat(all_data, ignore_index=True)

# outcome
print(merged_data.shape)
print(merged_data.head())

# save the merged data to a CSV file
merged_data.to_csv("DataSets/full_data.csv", index=False)

data_without_trb = pd.read_excel(r"D:\file\Research\无速度无高度\航空发动机高度0马赫数0仿真3.xlsx")
data_without_trb = data_without_trb.drop(columns=["高度", "马赫数", "加力燃烧室燃油流量/g/s"], errors='ignore')
data_without_trb = data_without_trb.drop(index=0).reset_index(drop=True)
data_without_trb = data_without_trb.apply(pd.to_numeric, errors='coerce')

print(data_without_trb.shape)
print(data_without_trb.head())

data_without_trb.to_csv("DataSets/data_without_trb.csv", index=False)