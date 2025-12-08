import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 读取数据
df = pd.read_csv('StressLevelDataset.csv')

# 计算相关性矩阵
correlation_matrix = df.corr()

# 设置图形大小
plt.figure(figsize=(18, 14))

# 创建cubehelix调色板
cubehelix_cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)

# 绘制热力图，使用cubehelix调色板
sns.heatmap(correlation_matrix,
            annot=True,
            cmap=cubehelix_cmap,
            fmt='.2f',
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": 0.8})

# 添加标题
plt.title('Stress Level Dataset Correlation Heatmap (Cubehelix)', fontsize=16, pad=20)

# 调整布局
plt.tight_layout()

# 确保保存目录存在
save_dir = r'D:\Typora\Programs_python\20251205_StressLevelDataset'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 生成带序号的文件名，防止覆盖
def get_unique_filename(base_name, extension, directory):
    """生成带序号的唯一文件名"""
    counter = 1
    filename = f"{base_name}{extension}"
    filepath = os.path.join(directory, filename)

    # 检查文件是否存在，如果存在则添加序号
    while os.path.exists(filepath):
        filename = f"{base_name}_{counter}{extension}"
        filepath = os.path.join(directory, filename)
        counter += 1

    return filepath, filename


# 保存图像
base_name = "correlation_heatmap_cubehelix"
extension = ".png"
save_path, filename = get_unique_filename(base_name, extension, save_dir)

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"热力图已保存为: {filename}")
print(f"完整路径: {save_path}")

# 显示图像
plt.show()

# 打印相关性统计信息
print("\n与压力水平(stress_level)相关性最高的前10个特征:")
stress_corr = correlation_matrix['stress_level'].abs().sort_values(ascending=False)
print(stress_corr.head(11))  # 包含自身

# 可选：显示目录中所有相关文件的列表
print(f"\n目录 '{save_dir}' 中的相关文件:")
correlation_files = [f for f in os.listdir(save_dir) if f.startswith("correlation_heatmap") and f.endswith(".png")]
if correlation_files:
    for file in correlation_files:
        print(f"  - {file}")
else:
    print("  暂无相关文件")