import matplotlib.pyplot as plt
import numpy as np

# 设置支持中文的字体（关键修改！）######################################
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统可用
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统可用
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
####################################################################

# 设置大字体
plt.rcParams.update({'font.size': 30})

# 自定义数据
categories = ['检索召回率', '检索命中率', '答案完整性', '有效回答比例']
values = [0.91, 0.99, 0.99, 0.99]
bar_labels = ['0.91', '0.99', '0.99', '0.99']

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制柱状图
bar_width = 0.6
bars = plt.bar(categories, values, width=bar_width,
               color=['skyblue', 'lightgreen', 'salmon', 'gold'])

# 添加数值标签
for bar, label in zip(bars, bar_labels):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, label,
             ha='center', va='bottom', fontsize=30)

# 标题和轴标签
plt.title('眼耳鼻喉科', fontsize=30, pad=20)
plt.xlabel('评估指标', fontsize=18, labelpad=10)
plt.ylabel('得分', fontsize=18, labelpad=10)

# 调整坐标轴
plt.ylim(0.8, 1.0)
plt.yticks(np.arange(0.8, 1.01, 0.05))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)

# 显示
plt.tight_layout()
plt.show()