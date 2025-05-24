import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='Microsoft YaHei')   # 让 Seaborn 也使用

data_raw=pd.read_csv(r'D:\file\Research\research\Senior_Thesis\DataSets\related_data_raw.csv')
data_std=pd.read_csv(r'D:\file\Research\research\Senior_Thesis\DataSets\related_data_std.csv')

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score



X = data_std.drop(columns=["时间","推力/kN"])
y = data_std["推力/kN"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    score_test = model.score(X_test, y_test)
    score_train = model.score(X_train, y_train)
    r2 = r2_score(y_test, y_pred)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"MSE: {mse}")
    print(f"Accuracy of test dataset: {score_test}")
    print(f"Accuracy of train dataset: {score_train}")
    print(f"R2 score: {r2}")
    return y_pred, mse, score_test, score_train, r2

# 统一可视化所有模型的 Actual vs Predict
def plot_all_models_performance(y_test, results):
    n = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(5 * ncols, 5 * nrows))
    for idx, (label, res) in enumerate(results.items()):
        plt.subplot(nrows, ncols, idx + 1)
        plt.scatter(y_test, res['y_pred'], color=res['color'], alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
        plt.title(f"{label}\nTest Score={res['score_test']:.5f}\nMSE={res['mse']:.4f}")
        plt.xlabel("Actual Thrust/kN")
        plt.ylabel("Predict Thrust/kN")
    plt.tight_layout()
    plt.savefig("Figures/All_Models_Performance.png")
    plt.close()

# 使用示例
models = [
    (RandomForestRegressor(n_estimators=100, random_state=42), 'blue', 'Random Forest'),
    (DecisionTreeRegressor(random_state=42), 'green', 'Decision Tree'),
    (SVR(kernel='rbf', C=1.0, epsilon=0.1), 'purple', 'SVM'),
    (LinearRegression(), 'orange', 'Linear Regression'),
    (XGBRegressor(n_estimators=100, random_state=42), 'red', 'XGBoost'),
    (MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42), 'cyan', 'Neural Network')
]

results = {}
for model, color, label in models:
    y_pred, mse, score_test, score_train, r2 = train_and_evaluate(
        model, X_train, X_test, y_train, y_test
    )
    results[label] = {
        'y_pred': y_pred, 'mse': mse, 'score_test': score_test,
        'score_train': score_train, 'r2': r2, 'color': color
    }
plot_all_models_performance(y_test, results)
# 综合对比各模型的R2和MSE
plt.figure(figsize=(10, 6))
labels = list(results.keys())
r2_scores = [results[l]['r2'] for l in labels]
mses = [results[l]['mse'] for l in labels]
plt.bar(labels, r2_scores, color=[results[l]['color'] for l in labels])
plt.ylabel("R2 Score")
plt.title("Model R2 Comparison")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("Figures/Model_R2_Comparison.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(labels, mses, color=[results[l]['color'] for l in labels])
plt.ylabel("MSE")
plt.title("Model MSE Comparison")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("Figures/Model_MSE_Comparison.png")
plt.close()


