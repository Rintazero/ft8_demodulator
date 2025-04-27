#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 数据点
snr_values = [35, 30, 28, 26, 25, 23, 21, 20]
freq_error_values = [1.64, 0.20, 0.665074, 1.44, 0.43, -16.69, 338.95, -1859.10]
decode_success = [True, True, True, True, True, False, False, False]
candidate_counts = [3, 2, 2, 2, 2, 0, 0, 0]

# 将布尔值转换为颜色
colors = ['green' if success else 'red' for success in decode_success]

# 创建图表
plt.figure(figsize=(10, 6))

# 创建散点图，大小根据候选消息数调整
scatter = plt.scatter(snr_values, np.abs(freq_error_values), c=colors, 
                     s=[50 * (count + 1) for count in candidate_counts], 
                     alpha=0.7, edgecolors='black')

# 添加图例
plt.legend(['解码成功', '解码失败'], loc='upper right')

# 添加标签和标题
plt.xlabel('信噪比 (dB)')
plt.ylabel('频偏误差绝对值 (Hz)')
plt.title('信噪比与频偏误差的关系')

# 使用对数刻度，因为频偏误差的变化范围很大
plt.yscale('log')

# 添加网格线
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# 添加数据标签
for i, txt in enumerate(freq_error_values):
    plt.annotate(f"{txt} Hz", 
                (snr_values[i], abs(freq_error_values[i])),
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center')

# 保存图像
plt.tight_layout()
plt.savefig('snr_vs_freq_error.png', dpi=300)

# 显示图像
plt.show()

print("图像已保存为 'snr_vs_freq_error.png'") 
