import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import GroupKFold

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='Microsoft YaHei')   # 让 Seaborn 也使用

data_raw=pd.read_csv(r'D:\file\Research\research\Senior_Thesis\DataSets\related_data_raw.csv')
data_std=pd.read_csv(r'D:\file\Research\research\Senior_Thesis\DataSets\related_data_std.csv')

data_std = data_std.sort_index()  # 保持原始顺序
data_std['RunID'] = (data_std['时间'].diff() < 0).cumsum()
print(data_std[['时间','RunID']].iloc[1390:1410])

# # 3_1_3.Cross-validation
sub_corr = data_std.drop(columns=["时间", "推力/kN", "RunID"]).corr().abs()
print(sub_corr)

high_corr = []
cols = sub_corr.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if sub_corr.iloc[i, j] > 0.9:
            high_corr.append((cols[i], cols[j], sub_corr.iloc[i, j]))
if high_corr:
    print("相关性大于0.9的变量对：")
    for var1, var2, corr_val in high_corr:
        print(f"{var1} <-> {var2}: {corr_val:.3f}")
else:
    print("没有相关性大于0.9的变量对。")

plt.figure(figsize=(40, 24))
sns.clustermap(sub_corr, annot=True, cmap='coolwarm')
plt.savefig("Figures/sub_corr.png")
plt.close()

chosen =  ['喷嘴喉道静压', '低压涡轮出口压力','推力燃料比']
# chosen = ['喷嘴喉道静压', '低压涡轮出口压力','推力燃料比',"高压涡轮机械功率","高压涡轮转速","高压涡轮出口温度","低压涡轮出口温度","低压涡轮转速"]
# 只保留 chosen 以外的特征
X_sub = data_std.drop(columns=chosen + ["时间", "RunID", "推力/kN"]).values
y = data_std['推力/kN'].values
print(X_sub.shape)

# PCA降维（如降到2维，可根据需要调整n_components）
pca = PCA(n_components=2)
X_sub_pca = pca.fit_transform(X_sub)

X_train, X_test, y_train, y_test = train_test_split(X_sub_pca, y, test_size=0.2, random_state=42)

# 定义所有模型
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "SVR": SVR(),
    "Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0),
    "MLPRegressor": MLPRegressor(random_state=42, max_iter=1000)
}

results = {}
plt.figure(figsize=(10, 8))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']  # Add more if needed

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    score_test = model.score(X_test, y_test)
    score_train = model.score(X_train, y_train)
    r2 = r2_score(y_test, y_pred)
    results[name] = {
        "mse": mse,
        "score_test": score_test,
        "score_train": score_train,
        "r2": r2
    }
    print(f"\n{name} with PCA:")
    print(f"MSE: {mse}")
    print(f"Accuracy of test dataset: {score_test}")
    print(f"Accuracy of train dataset: {score_train}")
    print(f"R2 score: {r2}")

    ax = axes[idx]
    ax.scatter(y_test, y_pred, color=colors[idx % len(colors)], alpha=0.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
    ax.set_title(f"{name}\nR²={score_test:.5f}")
    ax.set_xlabel("Actual Thrust/kN")
    ax.set_ylabel("Predicted Thrust/kN")

# Hide any unused subplots
for j in range(len(models), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("Figures/All_Models_Performance_with_PCA.png")
plt.close()

# Cross-validation and overfitting analysis for each model
cv_results = {}
fig_cv, axes_cv = plt.subplots(2, 3, figsize=(18, 10))
axes_cv = axes_cv.flatten()

groups = data_std['RunID'].values
gkf = GroupKFold(n_splits=5)

for idx, (name, model) in enumerate(models.items()):
    # 5-fold GroupKFold cross-validation (R² score)
    cv_scores = cross_val_score(model, X_sub_pca, y, cv=gkf, groups=groups, scoring='r2')
    cv_results[name] = cv_scores
    print(f"\n{name} 5-fold GroupKFold cross-validation R² scores: {cv_scores}")
    print(f"{name} mean R² score: {cv_scores.mean():.5f}")

    # Overfitting analysis
    model.fit(X_train, y_train)
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    print(f"{name} overfitting analysis:")
    print(f"Train score: {score_train:.5f}")
    print(f"Test score: {score_test:.5f}")
    if score_train - score_test > 0.1:
        print("Potential overfitting detected.")
    else:
        print("No obvious overfitting.")

    # Plot cross-validation results
    ax_cv = axes_cv[idx]
    ax_cv.plot(range(1, 6), cv_scores, marker='o', linestyle='-')
    ax_cv.set_title(f"{name}\nMean R²={cv_scores.mean():.3f}")
    ax_cv.set_xlabel('Fold')
    ax_cv.set_ylabel('R² Score')
    ax_cv.set_ylim(0, 1.05)
    ax_cv.grid(True)

# Hide any unused subplots
for j in range(len(models), len(axes_cv)):
    fig_cv.delaxes(axes_cv[j])

plt.tight_layout()
plt.savefig("Figures/All_Models_CV_R2.png")
plt.close()

# 定义基学习器和元学习器
# 定义基学习器和元学习器
base_learners = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeRegressor(random_state=42)),
    ('svr', SVR()),
    ('lr', LinearRegression()),
    ('xgb', XGBRegressor(random_state=42, verbosity=0)),
    ('mlp', MLPRegressor(random_state=42, max_iter=1000))
]

# 构建Stacking回归器
stacking_reg = StackingRegressor(
    estimators=base_learners,
    passthrough=True,
    cv=5
)

# 训练Stacking回归器
stacking_reg.fit(X_train, y_train)
y_pred_stack = stacking_reg.predict(X_test)

# 评估Stacking模型
mse_stack = mean_squared_error(y_test, y_pred_stack)
r2_stack = r2_score(y_test, y_pred_stack)
score_test_stack = stacking_reg.score(X_test, y_test)
score_train_stack = stacking_reg.score(X_train, y_train)

print("\nStacking Regressor with PCA:")
print(f"MSE: {mse_stack}")
print(f"Test set R2: {score_test_stack}")
print(f"Train set R2: {score_train_stack}")
print(f"R2 score: {r2_stack}")

# 可视化Stacking模型预测效果
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_stack, color='navy', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.title(f"Stacking Regressor\nR²={score_test_stack:.5f}")
plt.xlabel("Actual Thrust/kN")
plt.ylabel("Predicted Thrust/kN")
plt.tight_layout()
plt.savefig("Figures/Stacking_Model_Performance_with_PCA.png")
plt.close()

# 交叉验证Stacking模型
cv_scores_stack = cross_val_score(
    stacking_reg, X_sub_pca, y, cv=gkf, groups=groups, scoring='r2'
)
print(f"\nStacking Regressor 5-fold CV R² scores: {cv_scores_stack}")
print(f"Stacking Regressor mean R²: {cv_scores_stack.mean():.5f}")
