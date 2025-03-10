# FT8解调器

这是一个用Python实现的FT8数字模式解调器。FT8是一种用于业余无线电通信的数字模式，特别适合弱信号条件下的通信。

## 功能特点

- 专注于FT8数字模式
- 使用频谱图分析进行信号检测
- 实现了Costas同步序列检测
- 使用软判决LDPC解码
- CRC校验确保消息完整性
- 使用数学公式计算功率和，而非查找表
- 模块化设计，数据类型独立定义
- 简化的数据结构，使用属性计算而非存储冗余信息
- 优化的候选信号搜索算法，减少循环层次并利用Python内置数据结构
- 合并的时间和频率索引，简化数据结构和处理逻辑
- 强制使用二维数组表示频谱数据，提高代码清晰度和效率
- 极简的数据结构设计，移除不必要的转换函数
- 精确的子采样处理，保留信号的细节信息

## 文件结构

- `spectrogram_analyse.py`: 频谱图计算和分析
- `ft8_decode.py`: FT8解调和解码
- `crc.py`: CRC计算和验证
- `ftx_types.py`: 数据类型定义
- `example.py`: 使用示例

## 数据结构

解调器使用以下主要数据结构：

- `FT8Waterfall`: 频谱瀑布数据
  - 只存储必要的信息：幅度数据、时间过采样率和频率过采样率
  - 使用属性（@property）动态计算其他参数，如频率bin数、时间块数和块步长
  - 强制使用二维数组表示频谱数据，形状为(频率, 时间)
  - 使用`__post_init__`验证输入数据的有效性

- `FT8Candidate`: 候选信号信息
  - 使用绝对索引：`abs_time`和`abs_freq`直接表示采样点位置
  - 极简设计，只包含必要的字段，无冗余方法
  - 存储同步分数
  - 保留子采样精度，确保信号处理的准确性

- `FT8Message`: 解码后的消息
  - 存储有效载荷和哈希值

- `FT8DecodeStatus`: 解码状态
  - 存储LDPC错误数
  - 存储CRC校验信息

## 算法优化

- **候选信号搜索**：
  - 合并了时间和频率的多层循环，减少了嵌套深度
  - 使用Python的`heapq`模块替代自定义堆实现，简化了代码
  - 通过负分数技巧实现最大堆（保留分数最高的候选）

- **数据结构优化**：
  - 使用单一的`abs_time`和`abs_freq`表示采样点位置
  - 在需要时分解为符号偏移和子采样偏移，保留所有精度
  - 移除了处理一维数组的冗余代码，专注于二维数组处理
  - 移除了不必要的转换函数，使代码更加简洁

- **精度优化**：
  - 避免使用整除操作导致子采样精度丢失
  - 在计算符号偏移时保留子采样信息
  - 在访问频谱数据时使用精确的采样点位置

## 使用方法

### 基本用法

```python
import numpy as np
from ft8_demodulator.ft8_decode import decode_ft8_message

# 加载音频数据
# wave_data: 音频数据，采样率为12000Hz的单声道音频
# sample_rate: 采样率，通常为12000Hz

# 解码FT8消息
results = decode_ft8_message(
    wave_data=wave_data,
    sample_rate=sample_rate,
    bins_per_tone=10,
    steps_per_symbol=10,
    max_candidates=20,
    min_score=10,
    max_iterations=20
)

# 处理结果
for message, status in results:
    # 处理解码后的消息
    print(f"消息哈希: 0x{message.hash:04x}")
    print(f"有效载荷: {' '.join(f'{b:02x}' for b in message.payload)}")
```

### 使用示例脚本

```bash
python -m ft8_demodulator.example <wav_file>
```

## 参数说明

- `bins_per_tone`: 每个音调的频率bin数（频率过采样率）
- `steps_per_symbol`: 每个符号的时间步数（时间过采样率）
- `max_candidates`: 最大候选信号数
- `min_score`: 最小同步分数阈值
- `max_iterations`: LDPC解码的最大迭代次数

## 数学公式

解调器使用以下数学公式进行计算：

- 功率和计算: y = 10*log10(1 + 10^(x/10))
  - 其中x是较弱信号的相对强度(dB)
  - y是信号电平增加值(dB)

## LDPC解码

LDPC（低密度奇偶校验码）解码部分需要用户自行实现。在`ft8_decode.py`文件中，`bp_decode`函数是一个占位符，用户需要实现自己的LDPC解码算法。

## 注意事项

- 输入音频应为12000Hz采样率的单声道音频
- FT8信号通常在音频频谱的50-3000Hz范围内
- 解码性能取决于信号质量和信噪比
- 频谱图数据必须是二维数组，形状为(频率, 时间)
- 子采样精度对于弱信号解码至关重要

## 参考资料

- [FT8 Technical Specification](https://physics.princeton.edu/pulsar/k1jt/FT8_Operating_Tips.pdf)
- [WSJT-X Project](https://physics.princeton.edu/pulsar/k1jt/wsjtx.html) 