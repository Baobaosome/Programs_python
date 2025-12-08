import matplotlib.pyplot as plt
import matplotlib

# 显示当前Matplotlib使用的字体目录
print("字体缓存目录:", matplotlib.get_cachedir())
print("字体配置文件路径:", matplotlib.matplotlib_fname())

# 列出所有可用字体
from matplotlib.font_manager import fontManager

for font in fontManager.ttflist:
    if 'simhei' in font.name.lower() or 'ming' in font.name.lower() or 'kai' in font.name.lower():
        print(f"字体名称: {font.name}, 路径: {font.fname}")
