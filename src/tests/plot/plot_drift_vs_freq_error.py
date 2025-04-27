#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 数据点
drift_rates = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]
freq_error_values = [10.58, 8.99, -0.74, 4.56, 0.54, 2.18, 3.12, -2.77, 1.08]
decode_success = [True, True, True, True, True, True, True, True, True]
candidate_counts = [2, 2, 2, 3, 2, 1, 2, 3, 2]

# 将布尔值转换为颜色
colors = ['green' if success else 'red' for success in decode_success]

# 创建图表
plt.figure(figsize=(12, 7))

# 创建散点图，大小根据候选消息数调整
scatter = plt.scatter(drift_rates, np.abs(freq_error_values), c=colors, 
                     s=[50 * count for count in candidate_counts], 
                     alpha=0.7, edgecolors='black')

# 添加图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='解码成功'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='解码失败')
]
plt.legend(handles=legend_elements, loc='upper right')

# 添加标签和标题
plt.xlabel('频率漂移率 (Hz/s)')
plt.ylabel('频偏误差绝对值 (Hz)')
plt.title('频率漂移率与频偏误差的关系')

# 添加网格线
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 添加原始频偏误差的值标签
for i, txt in enumerate(freq_error_values):
    plt.annotate(f"{txt} Hz", 
                (drift_rates[i], abs(freq_error_values[i])),
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center')

# 保存图像
plt.tight_layout()
plt.savefig('drift_vs_freq_error.png', dpi=300)

# 显示图像
plt.show()

print("图像已保存为 'drift_vs_freq_error.png'") 
