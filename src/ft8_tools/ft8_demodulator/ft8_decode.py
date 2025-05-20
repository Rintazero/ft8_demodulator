import numpy as np
import scipy as sci
from typing import List, Tuple, Optional
import math
import heapq

# 从spectrogram_analyse.py导入常量和函数
from .spectrogram_analyse import (
    FT8_SYMBOL_DURATION_S,
    FT8_SYMBOL_FREQ_INTERVAL_HZ,
    calculate_spectrogram
)

# 导入CRC模块
from .crc import compute_crc, extract_crc

# 导入数据类型
from .ftx_types import (
    FT8Protocol,
    FT8Waterfall,
    FT8Candidate,
    FT8Message,
    FT8DecodeStatus
)

# 导入LDPC解码器
from .ldpc_decoder import bp_decode

# FT8常量
FT8_ND = 58  # 数据符号数
FT8_NUM_SYNC = 3  # 同步序列数
FT8_LENGTH_SYNC = 7  # 每个同步序列的长度
FT8_SYNC_OFFSET = 36  # 同步序列之间的偏移
FT8_LDPC_N = 174  # LDPC码长
FT8_LDPC_K = 91  # LDPC信息位长度
FT8_LDPC_K_BYTES = 12  # LDPC信息位字节数

# 编码映射表
FT8_Gray_map = [0, 1, 3, 2, 5, 6, 4, 7]  # 正确的Gray码映射表

# FT8 Costas同步序列
FT8_Costas_pattern = [3, 1, 4, 0, 6, 5, 2]




def ft8_sync_score(wf: FT8Waterfall, candidate: FT8Candidate) -> float:
    """计算FT8同步分数
    
    Args:
        wf: FT8瀑布图数据
        candidate: 待评估的候选信号
    
    Returns:
        float: 归一化的同步分数，分数越高表示同步性越好
    """
    score = 0.0
    num_comparisons = 0
    time_offset_base = candidate.abs_time // wf.time_osr
    
    # 遍历3个Costas同步序列
    for m in range(FT8_NUM_SYNC):
        sequence_start = m * FT8_SYNC_OFFSET
        
        # 遍历每个同步序列中的符号
        for k in range(FT8_LENGTH_SYNC):
            block = sequence_start + k
            block_abs = time_offset_base + block
            
            # 检查时间边界
            if block_abs < 0 or block_abs >= wf.num_blocks:
                continue
            
            # 从Costas模式获取预期的音调索引
            tone_idx = FT8_Costas_pattern[k]
            current_power = candidate.get_log_power(block, tone_idx)
            
            # 与相邻频率bin比较
            if tone_idx > 0:
                score += current_power - candidate.get_log_power(block, tone_idx - 1)
                num_comparisons += 1
            
            if tone_idx < 7:
                score += current_power - candidate.get_log_power(block, tone_idx + 1)
                num_comparisons += 1
            
            # 与相邻时间bin比较
            if k > 0 and block_abs > 0:
                score += current_power - candidate.get_log_power(block - 1, tone_idx)
                num_comparisons += 1
            
            if k < FT8_LENGTH_SYNC - 1 and block_abs + 1 < wf.num_blocks:
                score += current_power - candidate.get_log_power(block + 1, tone_idx)
                num_comparisons += 1
    
    # 如果没有有效比较或分数无效，返回负无穷
    if num_comparisons == 0 or np.isnan(score) or np.isinf(score):
        return float('-inf')
        
    return score / num_comparisons

def ft8_find_candidates(wf: FT8Waterfall, num_candidates: int, min_score: int) -> List[FT8Candidate]:
    """查找候选信号"""
    candidates = []
    num_tones = 8  # FT8使用8-FSK调制
    
    # 修正搜索范围计算，确保不会越界
    time_range = range(-10 * wf.time_osr, wf.num_blocks * wf.time_osr - wf.time_osr * (FT8_ND+1))
    freq_range = range(0, wf.mag.shape[0] - (num_tones - 1) * wf.freq_osr)
    print("time_range:", time_range)
    print("freq_range:", freq_range)
    
    score_list = []
    for abs_time in time_range:
        for abs_freq in freq_range:
            # 创建候选对象
            candidate = FT8Candidate(
                waterfall=wf,
                abs_time=abs_time,
                abs_freq=abs_freq
            )
            
            # 计算分数
            score = ft8_sync_score(wf, candidate)
            
            # 只处理有效的分数
            if score == float('-inf') or score < min_score:
                continue
                
            candidate.score = score
            score_list.append(score)
            
            # 使用最小堆维护最高分的候选
            if len(candidates) < num_candidates:
                heapq.heappush(candidates, (-score, candidate))
            elif -score < candidates[0][0]:
                heapq.heapreplace(candidates, (-score, candidate))
    
    # 提取候选并按分数降序排序
    result = [item[1] for item in sorted(candidates, key=lambda x: x[0])]
    
    print(f"Number of candidates found: {len(candidates)}")

    if score_list:
        print(f"Score statistics:")
        print(f"  Max score: {max(score_list):.2f}")
        print(f"  Min score: {min(score_list):.2f}")
    
    return result

def ft8_extract_symbol(wf: np.ndarray, logl: np.ndarray) -> None:
    """Calculate non-normalized log-likelihood log(p(1)/p(0)) for 3 message bits (1 FSK symbol)"""
    # 简化代码，用于n_syms==1的简单情况
    s2 = np.zeros(8)

    for j in range(8):
        s2[j] = wf[FT8_Gray_map[j]]

    logl[0] = max(s2[4], s2[5], s2[6], s2[7]) - max(s2[0], s2[1], s2[2], s2[3])
    logl[1] = max(s2[2], s2[3], s2[6], s2[7]) - max(s2[0], s2[1], s2[4], s2[5])
    logl[2] = max(s2[1], s2[3], s2[5], s2[7]) - max(s2[0], s2[2], s2[4], s2[6])


def ft8_extract_likelihood(wf: FT8Waterfall, cand: FT8Candidate, log174: np.ndarray) -> None:
    """计算174个消息位的对数似然log(p(1)/p(0))，用于后续的软判决LDPC解码"""
    # 直接使用abs_time，不进行整除，保留所有细节
    # 计算符号级别的偏移，但保留子采样信息用于后续计算
    time_offset_base = cand.abs_time // wf.time_osr

    # 遍历FSK音调并跳过Costas同步符号
    for k in range(FT8_ND):
        # 跳过7或14个同步符号
        sym_idx = k + (7 if k < 29 else 14)
        bit_idx = 3 * k

        # 检查时间边界
        block = time_offset_base + sym_idx
        if (block < 0) or (block >= wf.num_blocks):
            log174[bit_idx + 0] = 0
            log174[bit_idx + 1] = 0
            log174[bit_idx + 2] = 0
        else:
            # 提取当前符号的8个频率bin
            symbol_data = np.zeros(8)
            for i in range(8):
                symbol_data[i] = cand.get_log_power(sym_idx, i)
            
            ft8_extract_symbol(symbol_data, log174[bit_idx:])

def ftx_normalize_logl(log174: np.ndarray) -> None:
    """归一化对数似然比"""
    # 计算log174的方差
    mean = np.mean(log174)
    variance = np.mean((log174 - mean) ** 2)

    # 归一化log174分布并用实验发现的系数缩放它
    norm_factor = math.sqrt(24.0 / variance)
    log174 *= norm_factor

def pack_bits(bit_array: np.ndarray, num_bits: int) -> bytearray:
    """将每个位表示为bit_array[]中的零/非零字节，打包为从packed[]的第一个字节的MSB开始的打包位字符串"""
    num_bytes = (num_bits + 7) // 8
    packed = bytearray(num_bytes)

    mask = 0x80
    byte_idx = 0
    for i in range(num_bits):
        if bit_array[i]:
            packed[byte_idx] |= mask
        mask >>= 1
        if not mask:
            mask = 0x80
            byte_idx += 1

    return packed

def ftx_compute_crc(data: bytearray, num_bits: int) -> int:
    """计算CRC校验和"""
    return compute_crc(data, num_bits)

def ftx_extract_crc(data: bytearray) -> int:
    """从数据中提取CRC"""
    return extract_crc(data)

def ft8_decode_candidate(wf: FT8Waterfall, cand: FT8Candidate, max_iterations: int) -> Tuple[bool, FT8Message, FT8DecodeStatus]:
    """解码候选信号"""
    message = FT8Message()
    status = FT8DecodeStatus()
    
    log174 = np.zeros(FT8_LDPC_N)  # 消息位编码为似然
    ft8_extract_likelihood(wf, cand, log174)

    ftx_normalize_logl(log174)

    # 使用LDPC解码器
    plain174, ldpc_errors = bp_decode(log174, max_iterations)
    status.ldpc_errors = ldpc_errors

    if status.ldpc_errors > 0:
        return False, message, status

    # 提取有效载荷 + CRC (前FTX_LDPC_K位)打包到字节数组中
    a91 = pack_bits(plain174, FT8_LDPC_K)
    
    # 提取CRC并检查它
    status.crc_extracted = extract_crc(a91)
    
    # 创建用于CRC计算的临时缓冲区
    crc_buffer = bytearray(12)  # 足够存储96位
    # 复制原始数据
    for i in range(10):  # 只复制前10个字节，包含77位消息
        crc_buffer[i] = a91[i]
    
    # 清除载荷后的位，准备进行CRC计算
    crc_buffer[9] &= 0xF8  # 保留前5位（77-72=5位）
    crc_buffer[10] = 0     # 清零后续字节
    crc_buffer[11] = 0
    
    # 计算前82位的CRC
    status.crc_calculated = compute_crc(crc_buffer, 82)

    if status.crc_extracted != status.crc_calculated:
        return False, message, status

    # 重用CRC值作为消息的哈希
    message.hash = status.crc_calculated

    # 复制payload（前77位消息）
    for i in range(10):
        message.payload[i] = a91[i]
    message.payload[9] &= 0xf8

    return True, message, status

def create_waterfall_from_spectrogram(spectrogram: np.ndarray, time_osr: int, freq_osr: int) -> FT8Waterfall:
    """从频谱图创建瀑布数据结构"""
    # 确保spectrogram是二维数组
    if len(spectrogram.shape) != 2:
        raise ValueError("spectrogram必须是二维数组，形状为(频率, 时间)")
        
    # 直接创建FT8Waterfall对象
    return FT8Waterfall(
        mag=spectrogram,  # 保持二维数组形式
        time_osr=time_osr,
        freq_osr=freq_osr
    )

def decode_ft8_message(wave_data: np.ndarray, sample_rate: int, 
                      bins_per_tone: int = 2, steps_per_symbol: int = 2,
                      max_candidates: int = 20, min_score: int = 10,
                      max_iterations: int = 20,
                      freq_min: float = None,  # 最小频率限制 (Hz)
                      freq_max: float = None,  # 最大频率限制 (Hz)
                      time_min: float = None,  # 最小时间限制 (秒)
                      time_max: float = None   # 最大时间限制 (秒)
                      ) -> List[Tuple[FT8Message, FT8DecodeStatus]]:
    """解码FT8消息的主函数
    
    Args:
        wave_data: 波形数据
        sample_rate: 采样率
        bins_per_tone: 每个音调的频率bin数
        steps_per_symbol: 每个符号的时间步数
        max_candidates: 最大候选数量
        min_score: 最小分数阈值
        max_iterations: 最大迭代次数
        freq_min: 最小频率限制 (Hz)，None表示不限制
        freq_max: 最大频率限制 (Hz)，None表示不限制
        time_min: 最小时间限制 (秒)，None表示不限制
        time_max: 最大时间限制 (秒)，None表示不限制
    
    Returns:
        List[Tuple[FT8Message, FT8DecodeStatus]]: 解码结果列表
    """
    # 计算频谱图
    spectrogram, f, t = calculate_spectrogram(
        wave_data, sample_rate, bins_per_tone, steps_per_symbol
    )
    
    

    # 只取正频率部分
    # positive_freq_mask = f >= 0
    # spectrogram = spectrogram[positive_freq_mask]
    # f = f[positive_freq_mask]

    # 应用频率限制
    if freq_min is not None or freq_max is not None:
        freq_min = freq_min if freq_min is not None else f[0]
        freq_max = freq_max if freq_max is not None else f[-1]
        freq_mask = (f >= freq_min) & (f <= freq_max)
        spectrogram = spectrogram[freq_mask]
        f = f[freq_mask]
    
    # 应用时间限制
    if time_min is not None or time_max is not None:
        time_min = time_min if time_min is not None else t[0]
        time_max = time_max if time_max is not None else t[-1]
        time_mask = (t >= time_min) & (t <= time_max)
        spectrogram = spectrogram[:, time_mask]
        t = t[time_mask]
    
    import matplotlib.pyplot as plt
    # 绘制频谱图
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]])
    plt.colorbar(label='Intensity (dB)')
    plt.title('FT8 Signal Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    
    # 创建瀑布数据结构
    wf = create_waterfall_from_spectrogram(
        spectrogram, steps_per_symbol, bins_per_tone
    )
    
    # 查找候选信号
    candidates = ft8_find_candidates(wf, max_candidates, min_score)
    
    # 标注候选点
    for i, cand in enumerate(candidates):
        # 计算候选点在时间和频率轴上的实际位置，不除以过采样率
        # 直接将候选点的索引映射到物理单位范围
        time_sec = t[0] + (cand.abs_time * (t[-1] - t[0])) / (wf.num_blocks * wf.time_osr)
        freq_hz = f[0] + (cand.abs_freq * (f[-1] - f[0])) / (wf.mag.shape[0])
        
        # 在频谱图上标注候选点
        plt.plot(time_sec, freq_hz, 'ro', markersize=4)
        # 添加标签，显示候选点的编号和分数
        plt.annotate(f"{i+1}:{cand.score:.1f}", 
                     (time_sec, freq_hz), 
                     xytext=(5, 5), 
                     textcoords='offset points',
                     color='white',
                     fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.7))
    
    plt.show()
    plt.savefig('ft8_spectrogram_with_candidates.png')
    plt.close()
    
    # 解码候选信号
    results = []
    for cand in candidates:
        success, message, status = ft8_decode_candidate(wf, cand, max_iterations)
        if success:
            time_sec = cand.abs_time / sample_rate
            freq_hz = (cand.abs_freq / wf.freq_osr) * FT8_SYMBOL_FREQ_INTERVAL_HZ
            
            score = cand.score
            results.append((message, status, time_sec, freq_hz, score))
    
    print(f"Decoded messages: {results}")
    return results
