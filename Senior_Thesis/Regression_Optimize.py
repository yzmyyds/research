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
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone

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

# 计算VIF（方差膨胀因子）
X_vif = data_std.drop(columns=["时间", "推力/kN", "RunID"])
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\n各自变量的VIF：")
print(vif_data)
# 处理VIF过高的变量（如VIF>10或inf），建议剔除或进一步分析
high_vif_features = vif_data[vif_data["VIF"] > 10]["feature"].tolist()
inf_vif_features = vif_data[vif_data["VIF"] == np.inf]["feature"].tolist()
# print("\nVIF大于10的变量：", high_vif_features)
# print("VIF为inf的变量：", inf_vif_features)

chosen =  ['喷嘴喉道静压', '低压涡轮出口压力','推力燃料比']
# chosen = ['喷嘴喉道静压', '低压涡轮出口压力','推力燃料比',"高压涡轮机械功率","高压涡轮转速","高压涡轮出口温度","低压涡轮出口温度","低压涡轮转速"]
# 只保留 chosen 以外的特征
X=data_std.drop(columns=["时间", "RunID", "推力/kN"]).values
X_sub = data_std.drop(columns=high_vif_features+inf_vif_features+ ["时间", "RunID", "推力/kN"]).values
y = data_std['推力/kN'].values
# print(X_sub.shape)

# PCA降维（如降到2维，可根据需要调整n_components）
pca = PCA(n_components=2)
X_sub_pca = pca.fit_transform(X)

# 对PCA降维后的2个主成分进行VIF检测
vif_pca = pd.DataFrame()
vif_pca["feature"] = [f"PC{i+1}" for i in range(X_sub_pca.shape[1])]
vif_pca["VIF"] = [variance_inflation_factor(X_sub_pca, i) for i in range(X_sub_pca.shape[1])]
print("\nPCA主成分的VIF：")
print(vif_pca)
# X_sub_pca=X_sub

# LSTM+time sliding

def create_sequences(X, y, run_ids, seq_len=10):
    Xs, ys = [], []
    for run_id in np.unique(run_ids):
        idx = np.where(run_ids == run_id)[0]
        X_run, y_run = X[idx], y[idx]
        for i in range(len(X_run) - seq_len):
            Xs.append(X_run[i:i+seq_len])
            ys.append(y_run[i+seq_len])
    return np.array(Xs), np.array(ys)

seq_len = 10
run_ids = data_std['RunID'].values
X_lstm, y_lstm = create_sequences(X_sub_pca, y, run_ids, seq_len=seq_len)

# 为每个序列分配对应的RunID（即每个序列的最后一个点的RunID）
seq_run_ids = []
for run_id in np.unique(run_ids):
    idx = np.where(run_ids == run_id)[0]
    for i in range(len(idx) - seq_len):
        seq_run_ids.append(run_id)
seq_run_ids = np.array(seq_run_ids)

# 划分训练集和测试集（保持RunID不混合）
unique_run_ids = np.unique(run_ids)
train_ids, test_ids = train_test_split(unique_run_ids, test_size=0.2, random_state=42)
train_mask = np.isin(seq_run_ids, train_ids)
test_mask = np.isin(seq_run_ids, test_ids)
X_train_lstm, y_train_lstm = X_lstm[train_mask], y_lstm[train_mask]
X_test_lstm, y_test_lstm = X_lstm[test_mask], y_lstm[test_mask]

# 构建LSTM模型
model = Sequential([
    LSTM(32, input_shape=(seq_len, X_sub_pca.shape[1])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_test_lstm, y_test_lstm),
    epochs=50, batch_size=64, callbacks=[early_stop], verbose=1
)

# 预测与评估
y_pred_lstm = model.predict(X_test_lstm).flatten()
mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
r2_lstm = r2_score(y_test_lstm, y_pred_lstm)
print("\nLSTM with time sliding window:")
print(f"MSE: {mse_lstm}")
print(f"R2 score: {r2_lstm}")

# 可视化
plt.figure(figsize=(6, 6))
plt.scatter(y_test_lstm, y_pred_lstm, color='teal', alpha=0.5)
plt.plot([min(y_test_lstm), max(y_test_lstm)], [min(y_test_lstm), max(y_test_lstm)], color='black', linestyle='--')
plt.title(f"LSTM\nR²={r2_lstm:.5f}")
plt.xlabel("Actual Thrust/kN")
plt.ylabel("Predicted Thrust/kN")
plt.tight_layout()
plt.savefig("Figures/LSTM_Performance_with_PCA.png")
plt.close()

# 可视化LSTM时间序列预测表现（实际值与预测值随时间变化）
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test_lstm)), y_test_lstm, label='Actual', color='teal')
plt.plot(range(len(y_pred_lstm)), y_pred_lstm, label='Predicted', color='orange', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Thrust/kN')
plt.title('LSTM Time Series Prediction\nActual vs Predicted Thrust/kN')
plt.legend()
plt.tight_layout()
plt.savefig("Figures/LSTM_TimeSeries_Prediction.png")
plt.close()

# 增加双向LSTM

model_bi = Sequential([
    Bidirectional(LSTM(32), input_shape=(seq_len, X_sub_pca.shape[1])),
    Dense(1)
])
model_bi.compile(optimizer='adam', loss='mse')

early_stop_bi = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history_bi = model_bi.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_test_lstm, y_test_lstm),
    epochs=50, batch_size=64, callbacks=[early_stop_bi], verbose=1
)

# 预测与评估（双向LSTM）
y_pred_bilstm = model_bi.predict(X_test_lstm).flatten()
mse_bilstm = mean_squared_error(y_test_lstm, y_pred_bilstm)
r2_bilstm = r2_score(y_test_lstm, y_pred_bilstm)
print("\nBidirectional LSTM with time sliding window:")
print(f"MSE: {mse_bilstm}")
print(f"R2 score: {r2_bilstm}")

# 可视化（双向LSTM）
plt.figure(figsize=(6, 6))
plt.scatter(y_test_lstm, y_pred_bilstm, color='purple', alpha=0.5)
plt.plot([min(y_test_lstm), max(y_test_lstm)], [min(y_test_lstm), max(y_test_lstm)], color='black', linestyle='--')
plt.title(f"Bidirectional LSTM\nR²={r2_bilstm:.5f}")
plt.xlabel("Actual Thrust/kN")
plt.ylabel("Predicted Thrust/kN")
plt.tight_layout()
plt.savefig("Figures/BiLSTM_Performance_with_PCA.png")
plt.close()

# 可视化双向LSTM时间序列预测表现
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test_lstm)), y_test_lstm, label='Actual', color='teal')
plt.plot(range(len(y_pred_bilstm)), y_pred_bilstm, label='BiLSTM Predicted', color='purple', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Thrust/kN')
plt.title('Bidirectional LSTM Time Series Prediction\nActual vs Predicted Thrust/kN')
plt.legend()
plt.tight_layout()
plt.savefig("Figures/BiLSTM_TimeSeries_Prediction.png")
plt.close()


# 继续原有PCA分割
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

# 分别展示几种模型x轴为时间或序号（与LSTM划分保持一致），y轴为推力，每幅图上有相应的预测和实际，用subplot画出

# 选取部分模型进行对比（如Random Forest, SVR, Linear Regression, XGBoost, LSTM）
compare_models = ["Random Forest", "Decision Tree", "SVR", "Linear Regression", "XGBoost", "MLPRegressor"]
n_models = len(compare_models) + 1  # +1 for LSTM

plt.figure(figsize=(18, 3 * n_models))
for idx, name in enumerate(compare_models):
    model = models[name]
    # 用与LSTM相同的测试集（即X_test_lstm的最后一个时刻的特征）进行预测
    # 由于LSTM输入是序列，取每个序列的最后一个特征作为普通模型的输入
    X_test_last = X_test_lstm[:, -1, :]  # shape: (n_samples, n_features)
    y_pred = model.predict(X_test_last)
    ax = plt.subplot(n_models, 1, idx + 1)
    ax.plot(range(len(y_test_lstm)), y_test_lstm, label='Actual', color='teal')
    ax.plot(range(len(y_pred)), y_pred, label='Predicted', color='orange', alpha=0.7)
    ax.set_title(f"{name} (Test Set)\nR²={r2_score(y_test_lstm, y_pred):.4f}")
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Thrust/kN')
    ax.legend()

# LSTM
ax = plt.subplot(n_models, 1, n_models)
ax.plot(range(len(y_test_lstm)), y_test_lstm, label='Actual', color='teal')
ax.plot(range(len(y_pred_lstm)), y_pred_lstm, label='Predicted', color='orange', alpha=0.7)
ax.set_title(f"LSTM (Test Set)\nR²={r2_lstm:.4f}")
ax.set_xlabel('Sample Index')
ax.set_ylabel('Thrust/kN')
ax.legend()

plt.tight_layout()
plt.savefig("Figures/Model_TimeSeries_Comparison.png")
plt.close()
# Cross-validation and overfitting analysis for each model
cv_results = {}
fig_cv, axes_cv = plt.subplots(2, 3, figsize=(18, 10))
axes_cv = axes_cv.flatten()

groups = data_std['RunID'].values
gkf = GroupKFold(n_splits=5)

# 5-fold GroupKFold cross-validation (R² score) for all models, plot on one figure
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']
plt.figure(figsize=(10, 6))

for idx, (name, model) in enumerate(models.items()):
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

    # Plot all models' CV R² on the same figure
    plt.plot(range(1, 6), cv_scores, marker='o', linestyle='-', color=colors[idx % len(colors)], label=f"{name} (mean={cv_scores.mean():.3f})")

plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.title('5-fold GroupKFold Cross-Validation R² Scores for All Models')
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Figures/All_Models_CV_R2.png")
plt.close()

# # 定义基学习器和元学习器
# # 定义基学习器和元学习器
# base_learners = [
#     ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('dt', DecisionTreeRegressor(random_state=42)),
#     ('svr', SVR()),
#     ('lr', LinearRegression()),
#     ('xgb', XGBRegressor(random_state=42, verbosity=0)),
#     ('mlp', MLPRegressor(random_state=42, max_iter=1000))
# ]

# # 构建Stacking回归器
# stacking_reg = StackingRegressor(
#     estimators=base_learners,
#     passthrough=True,
#     cv=5
# )

# # 训练Stacking回归器
# stacking_reg.fit(X_train, y_train)
# y_pred_stack = stacking_reg.predict(X_test)

# # 评估Stacking模型
# mse_stack = mean_squared_error(y_test, y_pred_stack)
# r2_stack = r2_score(y_test, y_pred_stack)
# score_test_stack = stacking_reg.score(X_test, y_test)
# score_train_stack = stacking_reg.score(X_train, y_train)

# print("\nStacking Regressor with PCA:")
# print(f"MSE: {mse_stack}")
# print(f"Test set R2: {score_test_stack}")
# print(f"Train set R2: {score_train_stack}")
# print(f"R2 score: {r2_stack}")

# # 可视化Stacking模型预测效果
# plt.figure(figsize=(6, 6))
# plt.scatter(y_test, y_pred_stack, color='navy', alpha=0.5)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
# plt.title(f"Stacking Regressor\nR²={score_test_stack:.5f}")
# plt.xlabel("Actual Thrust/kN")
# plt.ylabel("Predicted Thrust/kN")
# plt.tight_layout()
# plt.savefig("Figures/Stacking_Model_Performance_with_PCA.png")
# plt.close()

# # 交叉验证Stacking模型
# cv_scores_stack = cross_val_score(
#     stacking_reg, X_sub_pca, y, cv=gkf, groups=groups, scoring='r2'
# )
# print(f"\nStacking Regressor 5-fold CV R² scores: {cv_scores_stack}")
# print(f"Stacking Regressor mean R²: {cv_scores_stack.mean():.5f}")
# 定义SVR参数网格，支持多种kernel
# 降低计算量：缩小参数网格、减少交叉验证折数
# 更小的参数空间，减少计算量
svr_param_dist = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10, 50],
    'gamma': ['scale', 0.01, 0.1, 1]
}

svr = SVR()
gkf = GroupKFold(n_splits=4)  # 4折交叉验证

# 使用tqdm显示RandomizedSearchCV进度

class TqdmRandomizedSearchCV(RandomizedSearchCV):
    def fit(self, X, y=None, **fit_params):
        n_iter = self.n_iter
        with tqdm(total=n_iter, desc="RandomizedSearchCV Progress") as pbar:
            self._pbar = pbar
            return super().fit(X, y, **fit_params)
    def _run_search(self, evaluate_candidates):
        def wrapper(candidate_params):
            self._pbar.update(len(candidate_params))
            return evaluate_candidates(candidate_params)
        super()._run_search(wrapper)

random_search = TqdmRandomizedSearchCV(
    svr, svr_param_dist, n_iter=12, cv=gkf, scoring='r2', n_jobs=1, verbose=2, random_state=42
)
random_search.fit(X_sub_pca, y, groups=groups)

print("\nSVR最佳参数：", random_search.best_params_)
print("SVR最佳交叉验证R²得分：", random_search.best_score_)

# 用最佳参数在训练/测试集上评估
best_svr = random_search.best_estimator_
best_svr.fit(X_train, y_train)
y_pred_svr = best_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print(f"SVR优化后测试集MSE: {mse_svr}")
print(f"SVR优化后测试集R2: {r2_svr}")

# 可视化   
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_svr, color='dodgerblue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.title(f"SVR (Tuned, RandomizedSearchCV)\nR²={r2_svr:.5f}")
plt.xlabel("Actual Thrust/kN")
plt.ylabel("Predicted Thrust/kN")
plt.tight_layout()
plt.savefig("Figures/SVR_Tuned_Randomized_Performance_with_PCA.png")
plt.close()


# # XGBoost参数调优（使用RandomizedSearchCV降低计算量，带进度条）


# class TqdmRandomizedSearchCV(RandomizedSearchCV):
#     def fit(self, X, y=None, **fit_params):
#         n_iter = self.n_iter
#         with tqdm(total=n_iter, desc="RandomizedSearchCV Progress (XGBoost)") as pbar:
#             self._pbar = pbar
#             return super().fit(X, y, **fit_params)
#     def _run_search(self, evaluate_candidates):
#         def wrapper(candidate_params):
#             self._pbar.update(len(candidate_params))
#             return evaluate_candidates(candidate_params)
#         super()._run_search(wrapper)

# xgb_param_dist = {
#     'n_estimators': [50, 100, 150, 200],
#     'max_depth': [3, 4, 5, 6, 7],
#     'learning_rate': [0.01, 0.03, 0.05, 0.1],
#     'subsample': [0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
#     'gamma': [0, 0.05, 0.1, 0.2],
#     'reg_alpha': [0, 0.001, 0.01, 0.1],
#     'reg_lambda': [1, 1.2, 1.5, 2]
# }

# xgb = XGBRegressor(random_state=42, verbosity=0)
# random_search_xgb = TqdmRandomizedSearchCV(
#     xgb, xgb_param_dist, n_iter=80, cv=4, scoring='r2', n_jobs=-1, verbose=1, random_state=42
# )
# random_search_xgb.fit(X_sub_pca, y, groups=groups)

# print("\nXGBoost最佳参数：", random_search_xgb.best_params_)
# print("XGBoost最佳交叉验证R²得分：", random_search_xgb.best_score_)

# # 用最佳参数在训练/测试集上评估
# best_xgb = random_search_xgb.best_estimator_
# best_xgb.fit(X_train, y_train)
# y_pred_xgb = best_xgb.predict(X_test)
# mse_xgb = mean_squared_error(y_test, y_pred_xgb)
# r2_xgb = r2_score(y_test, y_pred_xgb)
# print(f"XGBoost优化后测试集MSE: {mse_xgb}")
# print(f"XGBoost优化后测试集R2: {r2_xgb}")

# # 可视化
# plt.figure(figsize=(6, 6))
# plt.scatter(y_test, y_pred_xgb, color='forestgreen', alpha=0.5)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
# plt.title(f"XGBoost (Tuned, RandomizedSearchCV)\nR²={r2_xgb:.5f}")
# plt.xlabel("Actual Thrust/kN")
# plt.ylabel("Predicted Thrust/kN")
# plt.tight_layout()
# plt.savefig("Figures/XGBoost_Tuned_Randomized_Performance_with_PCA.png")
# plt.close()

# # MLPRegressor参数调优
# mlp_param_grid = {
#     'hidden_layer_sizes': [(50,), (100,), (50, 50)],
#     'activation': ['relu', 'tanh'],
#     'solver': ['adam'],
#     'alpha': [0.0001, 0.001],
#     'learning_rate': ['constant', 'adaptive'],
#     'max_iter': [500]
# }

# mlp = MLPRegressor(random_state=42)
# grid_search_mlp = GridSearchCV(
#     mlp, mlp_param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2
# )
# grid_search_mlp.fit(X_sub_pca, y, groups=groups)

# print("\nMLPRegressor最佳参数：", grid_search_mlp.best_params_)
# print("MLPRegressor最佳交叉验证R²得分：", grid_search_mlp.best_score_)

# # 用最佳参数在训练/测试集上评估
# best_mlp = grid_search_mlp.best_estimator_
# best_mlp.fit(X_train, y_train)
# y_pred_mlp = best_mlp.predict(X_test)
# mse_mlp = mean_squared_error(y_test, y_pred_mlp)
# r2_mlp = r2_score(y_test, y_pred_mlp)
# print(f"MLPRegressor优化后测试集MSE: {mse_mlp}")
# print(f"MLPRegressor优化后测试集R2: {r2_mlp}")

# # 可视化
# plt.figure(figsize=(6, 6))
# plt.scatter(y_test, y_pred_mlp, color='crimson', alpha=0.5)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
# plt.title(f"MLPRegressor (Tuned)\nR²={r2_mlp:.5f}")
# plt.xlabel("Actual Thrust/kN")
# plt.ylabel("Predicted Thrust/kN")
# plt.tight_layout()
# plt.savefig("Figures/MLPRegressor_Tuned_Performance_with_PCA.png")
# plt.close()