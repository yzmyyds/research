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
plt.savefig("Pearson_Factors.png")
plt.close()

#############################################################################################
#Model construction
#############################################################################################
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#1.Standarlize (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop(columns=['时间', '推力/kN']))
X = data.drop(columns=['时间', '推力/kN'])
y = data['推力/kN']
#2.Split the train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#3.Model training

#3_1.Random forest
model_RF = RandomForestRegressor(n_estimators=100, random_state=42)
model_RF.fit(X_train, y_train)
#3_1_1.Prediction
y_pred = model_RF.predict(X_test)
#3_1_2.Visualize and asess performance
mse = mean_squared_error(y_test, y_pred)
score_test=model_RF.score(X_test,y_test)
score_train=model_RF.score(X_train,y_train)
r2=r2_score(y_test,y_pred)
print("Random Forest:")
print(f"MSE: {mse}")
print(f"Accuracy of test dataset: {score_test}")
print(f"Accuracy of train dataset: {score_train}")
print(f"R2 score: {r2}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predict")
plt.xlabel("Actual Thrust/kN")
plt.ylabel("Predict Thrust/kN")
plt.savefig("Performance_asess_of_RF.png")
plt.close()

#3_2.Decision Tree
from sklearn.tree import DecisionTreeRegressor, plot_tree
model_DT=DecisionTreeRegressor(random_state=42)
model_DT.fit(X_train,y_train)
y_pred=model_DT.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
score_test=model_DT.score(X_test,y_test)
score_train=model_DT.score(X_train,y_train)
r2=r2_score(y_test,y_pred)
print("Decision Tree:")
print(f"MSE: {mse}")
print(f"Accuracy of test dataset: {score_test}")
print(f"Accuracy of train dataset: {score_train}")
print(f"R2 score: {r2}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predict")
plt.xlabel("Actual Thrust/kN")
plt.ylabel("Predict Thrust/kN")
plt.savefig("Performance_asess_of_DT.png")
plt.close()

#3_3. SVM
from sklearn.svm import SVR
model_SVR=SVR(kernel='rbf', C=1.0, epsilon=0.1)
model_SVR.fit(X_train,y_train)
y_pred=model_SVR.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
score_test=model_SVR.score(X_test,y_test)
score_train=model_SVR.score(X_train,y_train)
r2=r2_score(y_test,y_pred)
print("SVM:")
print(f"MSE: {mse}")
print(f"Accuracy of test dataset: {score_test}")
print(f"Accuracy of train dataset: {score_train}")
print(f"R2 score: {r2}")
plt.show()