import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载鸢尾花数据集
iris = load_iris()
print("数据集特征名称:", iris.feature_names)
print("目标类别名称:", iris.target_names)
print("数据形状:", iris.data.shape)
print("目标形状:", iris.target.shape)

# 创建DataFrame便于分析
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

# 数据基本信息
print("数据头部信息:")
print(df.head())
print("\n数据基本信息:")
print(df.info())
print("\n描述性统计:")
print(df.describe())

# 检查缺失值
print("\n缺失值统计:")
print(df.isnull().sum())

# 特征分布可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = iris.feature_names
for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    for species in iris.target_names:
        species_data = df[df['species'] == species][feature]
        axes[row, col].hist(species_data, alpha=0.7, label=species)
    axes[row, col].set_title(f'{feature} 分布')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('频数')
    axes[row, col].legend()

plt.tight_layout()
plt.show()

# 特征关系散点图矩阵
sns.pairplot(df, hue='species', diag_kind='hist', palette='viridis')
plt.suptitle('特征关系散点图矩阵', y=1.02)
plt.show()

# 相关性热力图
plt.figure(figsize=(10, 8))
correlation_matrix = df[iris.feature_names].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.show()

# 准备特征和标签
X = df[iris.feature_names]
y = df['target']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n标准化后数据统计:")
print("均值:", X_scaled.mean(axis=0))
print("标准差:", X_scaled.std(axis=0))

# 划分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集形状: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"测试集形状: X_test {X_test.shape}, y_test {y_test.shape}")

# 检查各类别在训练测试集中的分布
print("\n训练集中各类别样本数:")
print(pd.Series(y_train).value_counts().sort_index())
print("测试集中各类别样本数:")
print(pd.Series(y_test).value_counts().sort_index())

# 创建并训练逻辑回归模型
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 在训练集和测试集上进行预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算准确率
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")

# 详细分类报告
print("\n测试集分类报告:")
print(classification_report(y_test, y_test_pred, target_names=iris.target_names))

# 混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()

# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'coefficient': model.coef_[0]  # 以第一类为基准
}).sort_values('coefficient', key=abs, ascending=False)

print("\n特征重要性排序:")
print(feature_importance)

# 使用前两个特征进行二维可视化（便于展示决策边界）
X_2d = X_scaled[:, :2]  # 只使用萼片长度和萼片宽度
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.3, random_state=42, stratify=y
)

# 训练二维模型
model_2d = LogisticRegression(max_iter=1000, random_state=42)
model_2d.fit(X_train_2d, y_train_2d)

# 创建网格点进行决策边界绘制
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.Paired)
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y,
                     edgecolors='k', cmap=plt.cm.Paired)
plt.colorbar(scatter)
plt.xlabel('萼片长度 (标准化后)')
plt.ylabel('萼片宽度 (标准化后)')
plt.title('逻辑回归决策边界 (使用前两个特征)')
plt.show()

# 三维特征空间可视化（使用前三个特征）
from sklearn.decomposition import PCA

# 使用PCA进行降维可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                     cmap='viridis', edgecolors='k')
plt.colorbar(scatter)
plt.xlabel('主成分 1 (解释方差: {:.2f}%)'.format(pca.explained_variance_ratio_[0]*100))
plt.ylabel('主成分 2 (解释方差: {:.2f}%)'.format(pca.explained_variance_ratio_[1]*100))
plt.title('PCA降维可视化')
plt.show()

from sklearn.model_selection import cross_val_score, GridSearchCV

# 交叉验证评估模型稳定性
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"交叉验证准确率: {cv_scores}")
print(f"平均交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 超参数调优（修复版本）
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # 扩展C值范围
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [5000, 10000]  # 显著增加迭代次数
}

# 确保数据已经标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 对整个数据集进行标准化

grid_search = GridSearchCV(LogisticRegression(random_state=42),
                         param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)  # 使用标准化后的数据

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数重新训练模型
best_model = grid_search.best_estimator_
best_test_accuracy = best_model.score(X_test, y_test)
print(f"优化后测试集准确率: {best_test_accuracy:.4f}")

# 学习曲线分析
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 使用更稳健的参数设置
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_scaled, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42,  # 添加随机种子确保可重复性
    shuffle=True,     # 打乱数据顺序
    n_jobs=-1         # 使用所有CPU核心
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练得分")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="交叉验证得分")
plt.xlabel("训练样本数")
plt.ylabel("准确率")
plt.title("学习曲线")
plt.legend()
plt.grid(True)
plt.show()

import joblib

# 保存模型和标准化器
model_dict = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist()
}

joblib.dump(model_dict, 'iris_classifier.pkl')
print("模型已保存为 'iris_classifier.pkl'")


# 模拟新数据预测
def predict_new_flower(sepal_length, sepal_width, petal_length, petal_width):
    """预测新鸢尾花样本的类别"""
    # 加载模型
    loaded_model_dict = joblib.load('iris_classifier.pkl')
    model_loaded = loaded_model_dict['model']
    scaler_loaded = loaded_model_dict['scaler']

    # 准备新数据
    new_sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    new_sample_scaled = scaler_loaded.transform(new_sample)

    # 预测
    prediction = model_loaded.predict(new_sample_scaled)[0]
    probability = model_loaded.predict_proba(new_sample_scaled)[0]

    target_name = loaded_model_dict['target_names'][prediction]

    print(f"\n预测结果:")
    print(f"类别: {target_name} (编号: {prediction})")
    print("类别概率分布:")
    for i, prob in enumerate(probability):
        print(f"  {loaded_model_dict['target_names'][i]}: {prob:.4f}")

    return prediction, probability


# 测试新数据预测
print("新样本预测示例:")
predict_new_flower(5.1, 3.5, 1.4, 0.2)  # setosa
predict_new_flower(6.0, 2.7, 5.1, 1.6)  # virginica