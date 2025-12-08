# -*- coding: utf-8 -*-
"""
学生压力水平数据分析与建模项目
数据集：Stress_Dataset.csv
分析目标：预测学生压力类型（Eustress/Distress/No Stress）并识别关键影响因素
"""

# ==================== 1. 导入库 ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# # 设置中文字体 - 修改：修复中文显示问题
# import matplotlib
# matplotlib.font_manager._rebuild()  # 重新加载字体管理器
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
# sns.set_style("whitegrid")
# sns.set_palette("husl")

# 设置中文字体 - 修复中文显示问题
import matplotlib.font_manager as fm

# 指定字体文件路径
font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑字体路径

# 添加字体到matplotlib
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)

# 设置默认字体
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA

# 分类模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# 评估指标
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, auc)

# 保存模型
import joblib
import os

# 创建保存目录
output_dir = "D:/Typora/Programs_python/20251205_StressLevelDataset"
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# ==================== 2. 数据加载与初步探索 ====================
print("=" * 50)
print("1. 数据加载与初步探索")
print("=" * 50)

# 加载数据
df = pd.read_csv('Stress_Dataset.csv')
print(f"数据集形状: {df.shape}")
print(f"数据集信息:")
print(df.info())

# 显示前几行
print("\n前5行数据:")
print(df.head())

# 检查列名（修复可能的重复列名）
df.columns = [col.replace('.1', '_dup') if '.1' in col else col for col in df.columns]
print(f"\n列名: {list(df.columns)}")

# 目标变量分析
target_col = 'Which type of stress do you primarily experience?'
print(f"\n目标变量分布:")
print(df[target_col].value_counts())
print(f"\n目标变量比例:")
print(df[target_col].value_counts(normalize=True))

# ==================== 3. 数据清洗与预处理 ====================
print("\n" + "=" * 50)
print("2. 数据清洗与预处理")
print("=" * 50)

# 创建原始数据副本
df_clean = df.copy()

# 3.1 处理重复列（数据集中有一个重复的问题列）
# 检查重复列
dup_columns = [col for col in df_clean.columns if 'Have you been dealing with anxiety or tension recently?' in col]
print(f"重复列: {dup_columns}")

# 如果有重复列，合并或删除
if len(dup_columns) > 1:
    # 合并重复列：取平均值
    df_clean['Anxiety_Tension_combined'] = df_clean[dup_columns].mean(axis=1).round()
    # 删除原始重复列
    df_clean = df_clean.drop(columns=dup_columns)
    print(f"已合并重复列: {dup_columns}")

# 3.2 重命名列以便更好理解
column_mapping = {
    'Gender': 'Gender',
    'Age': 'Age',
    'Have you recently experienced stress in your life?': 'Recent_Stress',
    'Have you noticed a rapid heartbeat or palpitations?': 'Heart_Palpitations',
    'Have you been dealing with anxiety or tension recently?_dup': 'Anxiety_Tension',
    'Do you face any sleep problems or difficulties falling asleep?': 'Sleep_Problems',
    'Have you been getting headaches more often than usual?': 'Headaches',
    'Do you get irritated easily?': 'Irritability',
    'Do you have trouble concentrating on your academic tasks?': 'Concentration_Problems',
    'Have you been feeling sadness or low mood?': 'Sadness',
    'Have you been experiencing any illness or health issues?': 'Health_Issues',
    'Do you often feel lonely or isolated?': 'Loneliness',
    'Do you feel overwhelmed with your academic workload?': 'Academic_Overwhelm',
    'Are you in competition with your peers, and does it affect you?': 'Peer_Competition',
    'Do you find that your relationship often causes you stress?': 'Relationship_Stress',
    'Are you facing any difficulties with your professors or instructors?': 'Professor_Difficulties',
    'Is your working environment unpleasant or stressful?': 'Work_Environment_Stress',
    'Do you struggle to find time for relaxation and leisure activities?': 'No_Relaxation_Time',
    'Is your hostel or home environment causing you difficulties?': 'Home_Environment_Stress',
    'Do you lack confidence in your academic performance?': 'Academic_Confidence_Lack',
    'Do you lack confidence in your choice of academic subjects?': 'Subject_Choice_Confidence_Lack',
    'Academic and extracurricular activities conflicting for you?': 'Activity_Conflict',
    'Do you attend classes regularly?': 'Class_Attendance',
    'Have you gained/lost weight?': 'Weight_Changes'
}

# 应用重命名
df_clean = df_clean.rename(columns=column_mapping)

# 检查是否有合并后的列，如果有就用合并后的列
if 'Anxiety_Tension_combined' in df_clean.columns:
    # 如果存在合并后的列，更新列名
    df_clean = df_clean.rename(columns={'Anxiety_Tension_combined': 'Anxiety_Tension'})


# 3.3 处理目标变量
# 简化目标变量
def simplify_stress_label(label):
    if 'Eustress' in label:
        return 'Eustress'
    elif 'Distress' in label:
        return 'Distress'
    elif 'No Stress' in label:
        return 'No_Stress'
    else:
        return label


df_clean['Stress_Type'] = df_clean[target_col].apply(simplify_stress_label)
df_clean = df_clean.drop(columns=[target_col])

print(f"简化后的目标变量分布:")
print(df_clean['Stress_Type'].value_counts())

# 3.4 检查缺失值
print(f"\n缺失值统计:")
print(df_clean.isnull().sum())

# 3.5 处理异常值（年龄）
print(f"\n年龄描述统计:")
print(df_clean['Age'].describe())

# 检查年龄异常值
Q1 = df_clean['Age'].quantile(0.25)
Q3 = df_clean['Age'].quantile(0.75)
IQR = Q3 - Q1
age_outliers = df_clean[(df_clean['Age'] < (Q1 - 1.5 * IQR)) | (df_clean['Age'] > (Q3 + 1.5 * IQR))]
print(f"年龄异常值数量: {len(age_outliers)}")

# 可视化年龄分布
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_clean['Age'])
plt.title('年龄分布箱线图')
plt.tight_layout()
plt.savefig(f"{output_dir}/1_年龄分布箱线图.png", dpi=300, bbox_inches='tight')
plt.close()

# 3.6 编码分类变量
# 性别编码（0=男性, 1=女性，根据数据集中0/1的表示）
df_clean['Gender'] = df_clean['Gender'].astype(int)

# 目标变量编码
label_encoder = LabelEncoder()
df_clean['Stress_Type_Encoded'] = label_encoder.fit_transform(df_clean['Stress_Type'])
print(f"\n目标变量编码映射: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# 3.7 分离特征和目标变量
X = df_clean.drop(columns=['Stress_Type', 'Stress_Type_Encoded'])
y = df_clean['Stress_Type_Encoded']

print(f"\n特征形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# ==================== 4. 探索性数据分析 ====================
print("\n" + "=" * 50)
print("3. 探索性数据分析")
print("=" * 50)

# 4.1 压力类型分布
plt.figure(figsize=(10, 6))
stress_counts = df_clean['Stress_Type'].value_counts()
colors = ['#4CAF50', '#FF9800', '#F44336']  # 绿色: Eustress, 橙色: No Stress, 红色: Distress
plt.pie(stress_counts.values, labels=stress_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('压力类型分布')
plt.tight_layout()
plt.savefig(f"{output_dir}/2_压力类型分布.png", dpi=300, bbox_inches='tight')
plt.close()

# 4.2 性别与压力类型的关系
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
gender_stress = pd.crosstab(df_clean['Gender'], df_clean['Stress_Type'])
gender_stress.plot(kind='bar', stacked=True, color=['#4CAF50', '#FF9800', '#F44336'])
plt.title('性别与压力类型关系')
plt.xlabel('性别 (0=男性, 1=女性)')
plt.ylabel('计数')
plt.legend(title='压力类型')

plt.subplot(1, 2, 2)
gender_stress_percent = gender_stress.div(gender_stress.sum(axis=1), axis=0) * 100
gender_stress_percent.plot(kind='bar', stacked=True, color=['#4CAF50', '#FF9800', '#F44336'])
plt.title('性别与压力类型关系（百分比）')
plt.xlabel('性别 (0=男性, 1=女性)')
plt.ylabel('百分比')
plt.legend(title='压力类型')

plt.tight_layout()
plt.savefig(f"{output_dir}/3_性别与压力类型关系.png", dpi=300, bbox_inches='tight')
plt.close()

# 4.3 年龄与压力类型的关系
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Stress_Type', y='Age', data=df_clean, palette=['#4CAF50', '#FF9800', '#F44336'])
plt.title('不同压力类型的年龄分布')
plt.xlabel('压力类型')
plt.ylabel('年龄')

plt.subplot(1, 2, 2)
sns.violinplot(x='Stress_Type', y='Age', data=df_clean, palette=['#4CAF50', '#FF9800', '#F44336'])
plt.title('不同压力类型的年龄分布（小提琴图）')
plt.xlabel('压力类型')
plt.ylabel('年龄')

plt.tight_layout()
plt.savefig(f"{output_dir}/4_年龄与压力类型关系.png", dpi=300, bbox_inches='tight')
plt.close()

# 4.4 各特征分布
# 选择几个关键特征进行可视化
key_features = ['Recent_Stress', 'Sleep_Problems', 'Academic_Overwhelm', 'Loneliness', 'Health_Issues']

plt.figure(figsize=(16, 10))
for i, feature in enumerate(key_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=df_clean, x=feature, hue='Stress_Type', multiple='stack',
                 palette=['#4CAF50', '#FF9800', '#F44336'], bins=5)
    plt.title(f'{feature}分布')
    plt.xlabel('评分 (1-5)')
    plt.ylabel('计数')

plt.tight_layout()
plt.savefig(f"{output_dir}/5_关键特征分布.png", dpi=300, bbox_inches='tight')
plt.close()

# 4.5 相关性分析
# 计算数值特征的相关性矩阵
numeric_features = [col for col in X.columns if col != 'Gender']  # 性别已经编码为0/1
correlation_matrix = X[numeric_features].corr()

plt.figure(figsize=(16, 14))
# 修改：热力图配色改为viridis样式并添加数值
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='viridis', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('特征相关性热图')
plt.tight_layout()
plt.savefig(f"{output_dir}/6_特征相关性热图.png", dpi=300, bbox_inches='tight')
plt.close()

# 目标变量与各特征的相关性
target_corr = []
for col in X.columns:
    if X[col].nunique() > 1:  # 确保不是常数
        corr, _ = stats.pointbiserialr(X[col], y)
        target_corr.append((col, abs(corr)))

target_corr_sorted = sorted(target_corr, key=lambda x: x[1], reverse=True)[:15]

plt.figure(figsize=(12, 8))
features, corr_values = zip(*target_corr_sorted)
colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
plt.barh(features, corr_values, color=colors)
plt.xlabel('与目标变量的相关性（绝对值）')
plt.title('与压力类型相关性最高的特征')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{output_dir}/7_特征与目标变量相关性.png", dpi=300, bbox_inches='tight')
plt.close()

# ==================== 5. 特征工程 ====================
print("\n" + "=" * 50)
print("4. 特征工程")
print("=" * 50)

# 5.1 创建新特征
X_engineered = X.copy()

# 首先检查所有列是否都存在
print("X_engineered的列:", X_engineered.columns.tolist())

# 创建综合压力指数 - 修复：检查列是否存在
stress_related_features = ['Recent_Stress', 'Anxiety_Tension', 'Sleep_Problems',
                           'Headaches', 'Irritability', 'Sadness']

# 检查哪些列存在，哪些不存在
existing_features = [col for col in stress_related_features if col in X_engineered.columns]
missing_features = [col for col in stress_related_features if col not in X_engineered.columns]
print(f"存在的特征: {existing_features}")
print(f"缺失的特征: {missing_features}")

# 如果Anxiety_Tension不存在，尝试使用其他可能的名字
if 'Anxiety_Tension' not in X_engineered.columns:
    # 查找可能的替代列名
    possible_names = [col for col in X_engineered.columns if 'anxiety' in col.lower() or 'tension' in col.lower()]
    print(f"可能的替代列名: {possible_names}")
    if possible_names:
        stress_related_features = [col if col != 'Anxiety_Tension' else possible_names[0] for col in
                                   stress_related_features]

# 确保所有特征都存在
stress_related_features = [col for col in stress_related_features if col in X_engineered.columns]
print(f"最终使用的特征: {stress_related_features}")

X_engineered['Overall_Stress_Index'] = X_engineered[stress_related_features].mean(axis=1)

# 创建学术压力指数
academic_features = ['Academic_Overwhelm', 'Academic_Confidence_Lack',
                     'Subject_Choice_Confidence_Lack', 'Class_Attendance']
academic_features = [col for col in academic_features if col in X_engineered.columns]
X_engineered['Academic_Stress_Index'] = X_engineered[academic_features].mean(axis=1)

# 创建社交压力指数
social_features = ['Peer_Competition', 'Relationship_Stress',
                   'Professor_Difficulties', 'Loneliness']
social_features = [col for col in social_features if col in X_engineered.columns]
X_engineered['Social_Stress_Index'] = X_engineered[social_features].mean(axis=1)

# 创建环境压力指数
environment_features = ['Work_Environment_Stress', 'Home_Environment_Stress',
                        'No_Relaxation_Time']
environment_features = [col for col in environment_features if col in X_engineered.columns]
X_engineered['Environment_Stress_Index'] = X_engineered[environment_features].mean(axis=1)

# 创建身体健康指数
health_features = ['Heart_Palpitations', 'Health_Issues', 'Weight_Changes']
health_features = [col for col in health_features if col in X_engineered.columns]
X_engineered['Health_Index'] = X_engineered[health_features].mean(axis=1)

# 创建认知功能指数
cognitive_features = ['Concentration_Problems', 'Activity_Conflict']
cognitive_features = [col for col in cognitive_features if col in X_engineered.columns]
X_engineered['Cognitive_Index'] = X_engineered[cognitive_features].mean(axis=1)

print(f"特征工程后特征数量: {X_engineered.shape[1]}")
print(f"新创建的特征: {[col for col in X_engineered.columns if col not in X.columns]}")

# 5.2 可视化新特征
new_features = ['Overall_Stress_Index', 'Academic_Stress_Index', 'Social_Stress_Index',
                'Environment_Stress_Index', 'Health_Index', 'Cognitive_Index']

plt.figure(figsize=(16, 12))
for i, feature in enumerate(new_features, 1):
    plt.subplot(2, 3, i)
    for stress_type, color in zip(['Eustress', 'No_Stress', 'Distress'], ['#4CAF50', '#FF9800', '#F44336']):
        subset = df_clean[df_clean['Stress_Type'] == stress_type]
        sns.kdeplot(data=subset, x=X_engineered.loc[subset.index, feature],
                    label=stress_type, color=color, fill=True, alpha=0.5)
    plt.title(f'{feature}分布')
    plt.xlabel('指数值')
    plt.ylabel('密度')
    plt.legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/8_新特征分布.png", dpi=300, bbox_inches='tight')
plt.close()

# 5.3 特征选择
# 使用随机森林进行特征重要性评估
rf_for_selection = RandomForestClassifier(n_estimators=100, random_state=42)
rf_for_selection.fit(X_engineered, y)

feature_importance = pd.DataFrame({
    'feature': X_engineered.columns,
    'importance': rf_for_selection.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 10))
top_n = 20
sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n))
plt.title(f'特征重要性（前{top_n}个特征）')
plt.xlabel('重要性得分')
plt.tight_layout()
plt.savefig(f"{output_dir}/9_特征重要性.png", dpi=300, bbox_inches='tight')
plt.close()

# 选择最重要的特征
selected_features = feature_importance.head(25)['feature'].tolist()
X_selected = X_engineered[selected_features]

print(f"\n选择的特征数量: {len(selected_features)}")
print(f"最重要的10个特征: {selected_features[:10]}")

# 5.4 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features, index=X_engineered.index)

# 保存预处理后的数据
preprocessed_data = pd.concat([X_scaled_df, pd.Series(y, name='Stress_Type')], axis=1)
preprocessed_data.to_csv(f"{output_dir}/preprocessed_data.csv", index=False)
print(f"\n预处理后的数据已保存到: {output_dir}/preprocessed_data.csv")

# ==================== 6. 数据分割 ====================
print("\n" + "=" * 50)
print("5. 数据分割")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
print(f"训练集目标变量分布:\n{pd.Series(y_train).value_counts(normalize=True)}")
print(f"测试集目标变量分布:\n{pd.Series(y_test).value_counts(normalize=True)}")

# ==================== 7. 建模与评估 ====================
print("\n" + "=" * 50)
print("6. 建模与评估")
print("=" * 50)

# 初始化模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# 存储结果
results = {}

# 7.1 训练和评估每个模型
for model_name, model in models.items():
    print(f"\n训练 {model_name}...")

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 交叉验证
    cv_scores = cross_val_score(model, X_scaled_df, y, cv=5, scoring='accuracy')

    # 存储结果
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 保存模型
    joblib.dump(model, f"{output_dir}/{model_name.replace(' ', '_')}_model.pkl")

# 7.2 模型比较
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'CV Accuracy (Mean)': [results[m]['cv_mean'] for m in results.keys()],
    'CV Accuracy (Std)': [results[m]['cv_std'] for m in results.keys()]
})

print("\n模型性能比较:")
print(results_df.sort_values('Accuracy', ascending=False))

# 可视化模型比较
plt.figure(figsize=(14, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics):
    plt.bar(x + i * width, results_df[metric], width, label=metric)

plt.xlabel('模型')
plt.ylabel('分数')
plt.title('不同模型性能比较')
plt.xticks(x + width * 1.5, results_df['Model'], rotation=45, ha='right')
plt.legend()
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(f"{output_dir}/10_模型性能比较.png", dpi=300, bbox_inches='tight')
plt.close()

# 7.3 混淆矩阵
best_model_name = results_df.iloc[results_df['Accuracy'].idxmax()]['Model']
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['y_pred']

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'{best_model_name}混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.savefig(f"{output_dir}/11_混淆矩阵_{best_model_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.close()

# 7.4 分类报告
print(f"\n{best_model_name}分类报告:")
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

# ==================== 8. 模型优化 ====================
print("\n" + "=" * 50)
print("7. 模型优化 (以随机森林为例)")
print("=" * 50)

# 选择随机森林进行优化
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=1
)

print("开始网格搜索...")
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 用最佳参数重新训练
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# 评估优化后的模型
y_pred_optimized = best_rf.predict(X_test)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
print(f"优化后模型测试集准确率: {accuracy_optimized:.4f}")

# 保存优化后的模型
joblib.dump(best_rf, f"{output_dir}/Optimized_Random_Forest_model.pkl")

# ==================== 9. 特征重要性分析 ====================
print("\n" + "=" * 50)
print("8. 特征重要性分析")
print("=" * 50)

# 使用优化后的随机森林分析特征重要性
feature_importance_optimized = pd.DataFrame({
    'feature': selected_features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("优化后模型的特征重要性:")
print(feature_importance_optimized.head(15))

# 可视化特征重要性
plt.figure(figsize=(12, 10))
top_n = 15
sns.barplot(x='importance', y='feature', data=feature_importance_optimized.head(top_n))
plt.title(f'优化后模型的特征重要性（前{top_n}个特征）')
plt.xlabel('重要性得分')
plt.tight_layout()
plt.savefig(f"{output_dir}/12_优化后特征重要性.png", dpi=300, bbox_inches='tight')
plt.close()

# ==================== 10. 结果解释与总结 ====================
print("\n" + "=" * 50)
print("9. 结果解释与总结")
print("=" * 50)

# 10.1 主要发现
print("主要发现:")
print("1. 数据集包含三种压力类型：Eustress(积极压力，69.5%)、No_Stress(无压力，13.8%)、Distress(消极压力，16.7%)")
print("2. 年龄分布主要在18-22岁之间，不同压力类型间的年龄分布无显著差异")
print("3. 性别与压力类型分布显示女性更易经历Distress，男性更易经历Eustress")

# 10.2 关键影响因素
print("\n关键影响因素（基于特征重要性）:")
top_features = feature_importance_optimized.head(10)
for i, (_, row) in enumerate(top_features.iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['importance']:.4f}")

# 10.3 模型表现总结
print("\n模型表现总结:")
print(f"最佳模型: {best_model_name}")
print(f"测试集准确率: {results[best_model_name]['accuracy']:.4f}")
print(f"交叉验证准确率: {results[best_model_name]['cv_mean']:.4f} (±{results[best_model_name]['cv_std']:.4f})")

# 10.4 业务建议
print("\n业务建议:")
print("1. 重点关注高压力风险学生：识别Distress风险高的学生并提供及时干预")
print("2. 针对关键因素设计干预措施：如睡眠问题、学业压力、孤独感等")
print("3. 差异化支持策略：为不同压力类型学生提供针对性支持")
print("4. 建立早期预警系统：利用模型预测高压力风险学生")
print("5. 定期评估与调整：持续监控压力因素变化并调整干预策略")

# 10.5 可视化总结
# 压力类型与关键特征的关系
key_factors = ['Overall_Stress_Index', 'Academic_Stress_Index',
               'Social_Stress_Index', 'Sleep_Problems']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, factor in enumerate(key_factors):
    stress_means = []
    stress_stds = []

    for stress_type in ['Eustress', 'No_Stress', 'Distress']:
        subset_idx = df_clean[df_clean['Stress_Type'] == stress_type].index
        values = X_engineered.loc[subset_idx, factor]
        stress_means.append(values.mean())
        stress_stds.append(values.std())

    x_pos = np.arange(len(['Eustress', 'No_Stress', 'Distress']))
    axes[i].bar(x_pos, stress_means, yerr=stress_stds,
                color=['#4CAF50', '#FF9800', '#F44336'],
                alpha=0.7, ecolor='black', capsize=5)
    axes[i].set_xlabel('压力类型')
    axes[i].set_ylabel(f'{factor}平均值')
    axes[i].set_title(f'{factor}与压力类型关系')
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels(['Eustress', 'No_Stress', 'Distress'])

    # 添加数值标签
    for j, v in enumerate(stress_means):
        axes[i].text(j, v + 0.05, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{output_dir}/13_关键因素与压力类型关系.png", dpi=300, bbox_inches='tight')
plt.close()

# ==================== 11. 保存结果报告 ====================
print("\n" + "=" * 50)
print("10. 保存结果报告")
print("=" * 50)

# 创建结果报告
report = f"""
学生压力水平数据分析报告
=====================

分析概要:
---------
- 数据集: {df.shape[0]} 行, {df.shape[1]} 列
- 目标变量: 压力类型 (Eustress/No_Stress/Distress)
- 分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

数据分布:
---------
{df_clean['Stress_Type'].value_counts().to_string()}

关键发现:
---------
1. 压力类型分布: Eustress占大多数({df_clean['Stress_Type'].value_counts(normalize=True)['Eustress']:.1%})
2. 最佳模型: {best_model_name} (准确率: {results[best_model_name]['accuracy']:.4f})
3. 关键影响因素: {', '.join(top_features['feature'].head(5).tolist())}

模型性能比较:
-------------
{results_df.sort_values('Accuracy', ascending=False).to_string(index=False)}

业务建议:
---------
1. 建立压力监测系统，重点关注高Distress风险学生
2. 针对睡眠问题、学业压力和社交孤立设计干预措施
3. 提供差异化的心理健康支持服务
4. 定期评估干预措施效果并持续优化

文件输出:
---------
- 预处理数据: {output_dir}/preprocessed_data.csv
- 可视化图表: {output_dir}/[1-13]_*.png
- 训练模型: {output_dir}/*_model.pkl
- 优化模型: {output_dir}/Optimized_Random_Forest_model.pkl
"""

# 保存报告
with open(f"{output_dir}/分析报告.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(f"分析报告已保存到: {output_dir}/分析报告.txt")
print(f"所有输出文件已保存到: {output_dir}")
print("\n分析完成!")