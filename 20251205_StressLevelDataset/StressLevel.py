import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

warnings.filterwarnings('ignore')
import os

# 1. 设置中文字体和输出目录
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
output_dir = r"D:\Typora\Programs_python\20251205_StressLevelDataset"
os.makedirs(output_dir, exist_ok=True)

# 2. 加载数据
df = pd.read_csv("StressLevelDataset.csv")
original_df = df.copy()  # 保留原始数据副本
print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"前5行数据:\n{df.head()}")

# 3. 数据清洗与预处理
print("\n=== 数据清洗与预处理 ===")
print(f"缺失值检查:\n{df.isnull().sum()}")
print(f"重复值检查: {df.duplicated().sum()}个重复行")
df = df.drop_duplicates()
print(f"删除重复值后形状: {df.shape}")


# 异常值处理 - 使用IQR方法
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


for col in df.columns[:-1]:  # 除了目标变量外的所有列
    df = remove_outliers(df, col)
print(f"去除异常值后形状: {df.shape}")

# 数据标准化
scaler = StandardScaler()
X = df.drop('stress_level', axis=1)
y = df['stress_level']
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 4. 探索性数据分析与特征工程
print("\n=== 探索性数据分析 ===")

# 压力水平分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['stress_level'].value_counts().sort_index().plot(kind='bar', ax=axes[0])
axes[0].set_title('压力水平分布')
axes[0].set_xlabel('压力水平')
axes[0].set_ylabel('频数')

df['stress_level'].value_counts().sort_index().plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
axes[1].set_title('压力水平占比')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_压力水平分布.png'), dpi=300, bbox_inches='tight')
plt.show()

# 特征相关性分析
correlation_matrix = X_scaled.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_特征相关性热力图.png'), dpi=300, bbox_inches='tight')
plt.show()

# 与目标变量的相关性
target_corr = pd.DataFrame({'特征': X.columns, '与压力的相关性': X_scaled.corrwith(y)})
target_corr = target_corr.sort_values('与压力的相关性', key=abs, ascending=False)
print("\n特征与压力水平的相关性排序:")
print(target_corr.head(10))

plt.figure(figsize=(10, 6))
bars = plt.barh(target_corr['特征'][:15], target_corr['与压力的相关性'][:15])
plt.xlabel('相关性系数')
plt.title('前15个与压力水平最相关的特征')
for bar, corr in zip(bars, target_corr['与压力的相关性'][:15]):
    plt.text(corr + (0.01 if corr >= 0 else -0.05), bar.get_y() + bar.get_height() / 2,
             f'{corr:.3f}', ha='left' if corr >= 0 else 'right', va='center')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_特征与压力相关性.png'), dpi=300, bbox_inches='tight')
plt.show()

# 按五大因素分组分析
factors = {
    '心理因素': ['anxiety_level', 'self_esteem', 'mental_health_history', 'depression'],
    '生理因素': ['headache', 'blood_pressure', 'sleep_quality', 'breathing_problem'],
    '环境因素': ['noise_level', 'living_conditions', 'safety', 'basic_needs'],
    '学术因素': ['academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns'],
    '社会因素': ['social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']
}

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

for i, (factor_name, factor_features) in enumerate(factors.items()):
    if i < 6:
        factor_corr = []
        for feature in factor_features:
            if feature in X_scaled.columns:
                corr = abs(X_scaled[feature].corr(y))
                factor_corr.append(corr)

        ax = axes[i]
        bars = ax.bar(range(len(factor_features)), factor_corr)
        ax.set_title(f'{factor_name}相关性')
        ax.set_xticks(range(len(factor_features)))
        ax.set_xticklabels(factor_features, rotation=45, ha='right')
        ax.set_ylabel('绝对值相关性')

        for bar, corr in zip(bars, factor_corr):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=8)

axes[5].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_五大因素相关性分析.png'), dpi=300, bbox_inches='tight')
plt.show()

# 创建新特征
print("\n=== 特征工程 ===")
X_engineered = X_scaled.copy()

# 1. 心理压力指数
X_engineered['mental_stress_index'] = (
        X_engineered['anxiety_level'] + X_engineered['depression'] - X_engineered['self_esteem']
)

# 2. 生理健康指数
X_engineered['physical_health_index'] = (
        X_engineered['sleep_quality'] - X_engineered['headache'] - X_engineered['breathing_problem']
)

# 3. 学术压力指数
X_engineered['academic_stress_index'] = (
        X_engineered['study_load'] + X_engineered['future_career_concerns'] - X_engineered['academic_performance']
)

# 4. 环境质量指数
X_engineered['environment_quality_index'] = (
        X_engineered['living_conditions'] + X_engineered['safety'] + X_engineered['basic_needs'] - X_engineered[
    'noise_level']
)

# 5. 社会支持指数
X_engineered['social_support_index'] = (
        X_engineered['social_support'] - X_engineered['peer_pressure'] - X_engineered['bullying'] + X_engineered[
    'extracurricular_activities']
)

# 6. 综合压力指数
X_engineered['comprehensive_stress_index'] = (
        X_engineered['mental_stress_index'] * 0.3 +
        X_engineered['academic_stress_index'] * 0.25 +
        X_engineered['social_support_index'] * (-0.2) +  # 负权重，表示缓解压力
        X_engineered['physical_health_index'] * (-0.15) +  # 负权重
        X_engineered['environment_quality_index'] * (-0.1)  # 负权重
)

print(f"特征工程后特征数量: {X_engineered.shape[1]}")

# 特征选择
selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X_engineered, y)
selected_features = X_engineered.columns[selector.get_support()].tolist()
print(f"\n选择的15个重要特征: {selected_features}")

# PCA分析
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_engineered)
print(f"PCA解释方差比例: {pca.explained_variance_ratio_}")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='压力水平')
plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.2%} 方差)')
plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.2%} 方差)')
plt.title('PCA降维可视化')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '5_PCA降维可视化.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. 建模与分析
print("\n=== 建模与分析 ===")
X_train, X_test, y_train, y_test = train_test_split(
    X_engineered[selected_features], y, test_size=0.3, random_state=42, stratify=y
)

# 定义模型
models = {
    '随机森林': RandomForestClassifier(random_state=42),
    '梯度提升': GradientBoostingClassifier(random_state=42),
    '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
    '支持向量机': SVC(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n训练 {name} 模型...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'y_pred': y_pred,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

    print(f"{name} 准确率: {accuracy:.4f}")
    print(f"分类报告:\n{classification_report(y_test, y_pred)}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'6_{name}_混淆矩阵.png'), dpi=300, bbox_inches='tight')
    plt.show()

# 模型性能比较
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, result) in enumerate(results.items()):
    if i < 4:
        report = result['report']
        classes = ['0', '1', '2']
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1 = [report[cls]['f1-score'] for cls in classes]

        x = np.arange(len(classes))
        width = 0.25

        axes[i].bar(x - width, precision, width, label='精确率')
        axes[i].bar(x, recall, width, label='召回率')
        axes[i].bar(x + width, f1, width, label='F1分数')

        axes[i].set_title(f'{name} (准确率: {result["accuracy"]:.3f})')
        axes[i].set_xlabel('压力水平')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(classes)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '7_模型性能对比.png'), dpi=300, bbox_inches='tight')
plt.show()

# 6. 模型优化
print("\n=== 模型优化 ===")
# 对随机森林进行超参数优化
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"优化后测试集准确率: {best_accuracy:.4f}")

# 特征重要性分析
feature_importance = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({
    '特征': selected_features,
    '重要性': feature_importance
}).sort_values('重要性', ascending=False)

plt.figure(figsize=(12, 8))
bars = plt.barh(feature_importance_df['特征'], feature_importance_df['重要性'])
plt.xlabel('特征重要性')
plt.title('随机森林特征重要性排名')
plt.gca().invert_yaxis()

for bar, imp in zip(bars, feature_importance_df['重要性']):
    plt.text(imp + 0.001, bar.get_y() + bar.get_height() / 2,
             f'{imp:.4f}', ha='left', va='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '8_特征重要性排名.png'), dpi=300, bbox_inches='tight')
plt.show()

# 交叉验证评估
cv_scores = cross_val_score(best_rf, X_engineered[selected_features], y, cv=10, scoring='accuracy')
print(f"\n10折交叉验证平均准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cv_scores, marker='o', linestyle='--', linewidth=2)
plt.axhline(y=cv_scores.mean(), color='r', linestyle='-', label=f'平均值: {cv_scores.mean():.3f}')
plt.fill_between(range(1, 11), cv_scores.mean() - cv_scores.std(),
                 cv_scores.mean() + cv_scores.std(), alpha=0.2, color='gray')
plt.xlabel('交叉验证折数')
plt.ylabel('准确率')
plt.title('10折交叉验证结果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '9_交叉验证结果.png'), dpi=300, bbox_inches='tight')
plt.show()

# 7. 结果解释与总结
print("\n=== 结果解释与总结 ===")
print("\n1. 关键发现:")
print("-" * 50)

# 分析特征重要性
top_features = feature_importance_df.head(10)
print("\n最重要的10个特征及其影响:")
for idx, row in top_features.iterrows():
    feature = row['特征']
    importance = row['重要性']

    # 解释特征影响
    if 'anxiety' in feature or 'depression' in feature:
        effect = "显著增加压力"
    elif 'self_esteem' in feature or 'sleep_quality' in feature:
        effect = "显著缓解压力"
    elif 'academic' in feature or 'study' in feature:
        effect = "与学业压力正相关"
    elif 'social' in feature:
        effect = "社会支持缓解压力"
    else:
        effect = "中等影响"

    print(f"  {feature}: 重要性={importance:.4f} ({effect})")

print("\n2. 模型表现总结:")
print("-" * 50)
for name, result in results.items():
    print(f"  {name}: 准确率 = {result['accuracy']:.4f}")

print(f"\n  优化后的随机森林: 准确率 = {best_accuracy:.4f}")

print("\n3. 数据分析结论:")
print("-" * 50)
print("""
1. 心理因素影响最大:
   - 焦虑水平和抑郁程度是预测学生压力的最重要指标
   - 自尊水平对缓解压力有显著作用

2. 学业压力显著:
   - 学习负担和未来职业担忧是主要压力源
   - 学术表现与压力呈负相关

3. 生理健康关键:
   - 睡眠质量是最重要的生理指标
   - 头痛和呼吸问题也是压力表现

4. 社会支持重要:
   - 社会支持网络能有效缓解压力
   - 同伴压力和欺凌会增加压力

5. 环境影响:
   - 居住条件、安全性和基本需求满足程度影响压力水平
   - 噪声水平有一定影响
""")

print("\n4. 建议措施:")
print("-" * 50)
print("""
1. 心理健康干预:
   - 定期开展心理健康筛查，重点关注焦虑和抑郁症状
   - 提供心理咨询服务，帮助学生建立积极的自我认知

2. 学业支持:
   - 优化课程安排，合理控制学习负担
   - 提供职业规划指导，减轻未来职业担忧
   - 建立师生沟通机制，改善师生关系

3. 健康促进:
   - 推广健康睡眠习惯，改善睡眠质量
   - 提供体育锻炼机会，缓解生理压力

4. 社会支持建设:
   - 建立同伴支持系统
   - 开展反欺凌教育活动
   - 鼓励参与课外活动

5. 环境改善:
   - 改善住宿条件，降低噪声水平
   - 保障基本生活需求
   - 营造安全的校园环境
""")

# 保存结果
output_file = os.path.join(output_dir, '分析结果总结.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("学生压力数据集分析报告\n")
    f.write("=" * 50 + "\n\n")

    f.write("1. 数据集概况:\n")
    f.write(f"   原始数据形状: {original_df.shape}\n")
    f.write(f"   清洗后数据形状: {df.shape}\n")
    f.write(f"   压力水平分布: {dict(df['stress_level'].value_counts().sort_index())}\n\n")

    f.write("2. 特征工程:\n")
    f.write(f"   原始特征数量: {X.shape[1]}\n")
    f.write(f"   工程后特征数量: {X_engineered.shape[1]}\n")
    f.write(f"   选择的重要特征数: {len(selected_features)}\n\n")

    f.write("3. 模型性能:\n")
    for name, result in results.items():
        f.write(f"   {name}: {result['accuracy']:.4f}\n")
    f.write(f"   优化后随机森林: {best_accuracy:.4f}\n")
    f.write(f"   10折交叉验证平均: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n\n")

    f.write("4. 最重要的10个特征:\n")
    for idx, row in top_features.iterrows():
        f.write(f"   {row['特征']}: {row['重要性']:.4f}\n")

    f.write("\n5. 建议措施总结:\n")
    f.write("   - 加强心理健康教育和干预\n")
    f.write("   - 优化学业安排和职业指导\n")
    f.write("   - 改善校园环境和住宿条件\n")
    f.write("   - 建立社会支持网络\n")
    f.write("   - 关注生理健康，特别是睡眠质量\n")

print(f"\n分析完成! 结果已保存到: {output_dir}")
print(f"文本报告: {output_file}")
print(f"生成的图表: {len([f for f in os.listdir(output_dir) if f.endswith('.png')])} 个PNG文件")

# 保存处理后的数据
processed_data_path = os.path.join(output_dir, 'processed_data.csv')
processed_df = pd.concat([X_engineered[selected_features], y], axis=1)
processed_df.to_csv(processed_data_path, index=False)
print(f"处理后的数据已保存: {processed_data_path}")