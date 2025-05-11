import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='Microsoft YaHei')   # 让 Seaborn 也使用

data=pd.read_csv('linear_strong_factors.csv',header=0)

#############################################################################################
#Model construction
#############################################################################################
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

data = data.sort_index()  # 保持原始顺序
data['RunID'] = (data['时间'].diff() < 0).cumsum()
print(data[['时间','RunID']].head(15))

#1.Standarlize (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop(columns=['时间','推力/kN']))
X = data.drop(columns=['时间','推力/kN'])
y = data['推力/kN']
#2.Split the train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
#3.Model training

#3_1.Random forest
from sklearn.ensemble import RandomForestRegressor
model_RF = RandomForestRegressor(n_estimators=100, random_state=42)
model_RF.fit(X_train, y_train)
#3_1_1.Prediction
y_pred = model_RF.predict(X_test)
#3_1_2.Visualize and assess performance
mse = mean_squared_error(y_test, y_pred)
score_test = model_RF.score(X_test, y_test)
score_train = model_RF.score(X_train, y_train)
r2 = r2_score(y_test, y_pred)
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
plt.savefig("Performance_assess_of_RF.png")
plt.close()

# #3_1_3.Cross-validation
sub_feats = ['喷嘴喉道静压', '低压涡轮出口压力', '喷嘴喉道速度', '喷嘴喉道静温', '推力燃料比']
sub_corr = data[sub_feats].corr().abs()
print(sub_corr)
sns.clustermap(sub_corr, annot=True, cmap='coolwarm')
plt.savefig("sub_corr.png")
plt.close()

chosen = ['喷嘴喉道静压', '推力燃料比']
X_sub = data[chosen].values
y = data['推力/kN'].values
groups = data['RunID'].values

# Standardize
scaler = StandardScaler()
X_sub_scaled = scaler.fit_transform(X_sub)

# Use GroupKFold to evaluate, preventing data leakage
gkf = GroupKFold(n_splits=5)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X_sub_scaled, y, groups=groups, cv=gkf, scoring='r2')

print("Using only 喷嘴喉道静压 + 推力燃料比:")
print("GroupKFold R2 scores:", scores)
print("Mean R2:", scores.mean(), "Std:", scores.std())

# Reduce multicollinearity and dimensionality using PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
X_pca = pca.fit_transform(X_sub_scaled)

# Evaluate the model using the reduced features
scores_pca = cross_val_score(rf, X_pca, y, groups=groups, cv=gkf, scoring='r2')

print("Using PCA-reduced features:")
print("GroupKFold R2 scores:", scores_pca)
print("Mean R2:", scores_pca.mean(), "Std:", scores_pca.std())
# Use Recursive Feature Elimination (RFE) to select important features

# Initialize the model for RFE
rf_rfe = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf_rfe, n_features_to_select=2)  # Select top 2 features

# Fit RFE on the scaled data
rfe.fit(X_sub_scaled, y)

# Get the selected features
selected_features = [chosen[i] for i in range(len(chosen)) if rfe.support_[i]]
print("Selected features by RFE:", selected_features)

# Evaluate the model using only the selected features
X_rfe = X_sub_scaled[:, rfe.support_]
scores_rfe = cross_val_score(rf, X_rfe, y, groups=groups, cv=gkf, scoring='r2')

print("Using RFE-selected features:")
print("GroupKFold R2 scores:", scores_rfe)
print("Mean R2:", scores_rfe.mean(), "Std:", scores_rfe.std())
# #3_2.Decision Tree
# from sklearn.tree import DecisionTreeRegressor, plot_tree
# model_DT=DecisionTreeRegressor(random_state=42)
# model_DT.fit(X_train,y_train)
# y_pred=model_DT.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# score_test=model_DT.score(X_test,y_test)
# score_train=model_DT.score(X_train,y_train)
# r2=r2_score(y_test,y_pred)
# print("Decision Tree:")
# print(f"MSE: {mse}")
# print(f"Accuracy of test dataset: {score_test}")
# print(f"Accuracy of train dataset: {score_train}")
# print(f"R2 score: {r2}")
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
# plt.title("Actual vs Predict")
# plt.xlabel("Actual Thrust/kN")
# plt.ylabel("Predict Thrust/kN")
# plt.savefig("Performance_asess_of_DT.png")
# plt.close()

# #3_3. SVM
# from sklearn.svm import SVR
# model_SVR=SVR(kernel='rbf', C=1.0, epsilon=0.1)
# model_SVR.fit(X_train,y_train)
# y_pred=model_SVR.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# score_test=model_SVR.score(X_test,y_test)
# score_train=model_SVR.score(X_train,y_train)
# r2=r2_score(y_test,y_pred)
# print("SVM:")
# print(f"MSE: {mse}")
# print(f"Accuracy of test dataset: {score_test}")
# print(f"Accuracy of train dataset: {score_train}")
# print(f"R2 score: {r2}")
# plt.show()

#3_4. Linear Regression 
from sklearn.linear_model import LinearRegression
model_LR=LinearRegression()
model_LR.fit(X_train,y_train)
y_pred=model_LR.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
score_test=model_LR.score(X_test,y_test)
score_train=model_LR.score(X_train,y_train)
print("\nLinear Regression:")
print(f"MSE: {mse}")
print(f"Accuracy of test dataset: {score_test}")
print(f"Accuracy of train dataset: {score_train}")