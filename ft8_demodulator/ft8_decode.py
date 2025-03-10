import numpy as np
import scipy as sci
from typing import List, Tuple, Optional
import math
import heapq

# 从spectrogram_analyse.py导入常量和函数
from ft8_demodulator.spectrogram_analyse import (
    FT8_SYMBOL_DURATION_S,
    FT8_SYMBOL_FREQ_INTERVAL_HZ,
    calculate_spectrogram
)

# 导入CRC模块
from ft8_demodulator.crc import compute_crc, extract_crc

# 导入数据类型
from ft8_demodulator.ftx_types import (
    FT8Protocol,
    FT8Waterfall,
    FT8Candidate,
    FT8Message,
    FT8DecodeStatus
)

# FT8常量
FT8_ND = 58  # 数据符号数
FT8_NUM_SYNC = 3  # 同步序列数
FT8_LENGTH_SYNC = 7  # 每个同步序列的长度
FT8_SYNC_OFFSET = 36  # 同步序列之间的偏移
FT8_LDPC_N = 174  # LDPC码长
FT8_LDPC_K = 91  # LDPC信息位长度
FT8_LDPC_K_BYTES = 12  # LDPC信息位字节数

# 编码映射表
FT8_Gray_map = [0, 1, 3, 2, 6, 7, 5, 4]

# FT8 Costas同步序列
FT8_Costas_pattern = [3, 1, 4, 0, 6, 5, 2]

def db_power_sum(x: float) -> float:
    """
    计算功率和的公式: y = 10*log10(1 + 10^(x/10))
    
    Args:
        x: 较弱信号的相对强度(dB)
    
    Returns:
        信号电平增加值(dB)
    """
    return 10 * math.log10(1 + 10 ** (x / 10))


def ft8_sync_score(wf: FT8Waterfall, candidate: FT8Candidate) -> int:
    """计算FT8同步分数"""
    score = 0.0
    num_average = 0

    time_offset_base = candidate.abs_time // wf.time_osr
    # 计算同步符号的平均分数 (m+k = 0-7, 36-43, 72-79)
    for m in range(FT8_NUM_SYNC):
        for k in range(FT8_LENGTH_SYNC):
            block = (FT8_SYNC_OFFSET * m) + k          # 相对于消息
            block_abs = time_offset_base + block       # 相对于捕获的信号
            
            # 检查时间边界
            if block_abs < 0:
                continue
            if block_abs >= wf.num_blocks:
                break

            sm = FT8_Costas_pattern[k]  # 预期bin的索引
            
            if sm > 0:
                # 查看低一个频率bin
                score += candidate.get_log_power(block, sm) - candidate.get_log_power(block, sm - 1)
                num_average += 1
            if sm < 7:
                # 查看高一个频率bin
                score += candidate.get_log_power(block, sm) - candidate.get_log_power(block, sm + 1)
                num_average += 1
            if (k > 0) and (block_abs > 0):
                # 查看时间上前一个符号
                score += candidate.get_log_power(block, sm) - candidate.get_log_power(block - 1, sm)
                num_average += 1
            if ((k + 1) < FT8_LENGTH_SYNC) and ((block_abs + 1) < wf.num_blocks):
                # 查看时间上后一个符号
                score += candidate.get_log_power(block, sm) - candidate.get_log_power(block + 1, sm)
                num_average += 1

    if num_average > 0:
        score /= num_average

    return score

def ft8_find_candidates(wf: FT8Waterfall, num_candidates: int, min_score: int) -> List[FT8Candidate]:
    """查找候选信号"""
    candidates = []
    num_tones = 8  # FT8使用8-FSK调制
    
    # 计算所有可能的时间和频率位置
    time_range = range(-10 * wf.time_osr, 20 * wf.time_osr)
    freq_range = range(0, (wf.num_bins - num_tones + 1) * wf.freq_osr)
    
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
            candidate.score = score
            
            if score < min_score:
                continue
                
            # 如果还没有收集足够的候选，或者当前候选比堆中最差的更好
            if len(candidates) < num_candidates:
                # 将分数取负值，因为heapq是最小堆
                heapq.heappush(candidates, (-score, candidate))
            elif -score < candidates[0][0]:  # 比较负分数，实际上是比较正分数的大小
                # 替换堆中最差的候选
                heapq.heapreplace(candidates, (-score, candidate))
    
    # 提取候选并按分数降序排序
    result = [item[1] for item in sorted(candidates, key=lambda x: x[0])]
    
    return result

def ft8_extract_symbol(wf: np.ndarray, logl: np.ndarray) -> None:
    """计算3个消息位(1个FSK符号)的非归一化对数似然log(p(1)/p(0))"""
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

def bp_decode(log174: np.ndarray, max_iterations: int, plain174: np.ndarray, ldpc_errors: List[int]) -> None:
    """LDPC信念传播解码"""
    # 这里需要实现LDPC解码
    # 暂时用随机值填充plain174
    plain174[:] = np.random.randint(0, 2, size=len(plain174))
    ldpc_errors[0] = 0

def ft8_decode_candidate(wf: FT8Waterfall, cand: FT8Candidate, max_iterations: int) -> Tuple[bool, FT8Message, FT8DecodeStatus]:
    """解码候选信号"""
    message = FT8Message()
    status = FT8DecodeStatus()
    
    log174 = np.zeros(FT8_LDPC_N)  # 消息位编码为似然
    ft8_extract_likelihood(wf, cand, log174)

    ftx_normalize_logl(log174)

    plain174 = np.zeros(FT8_LDPC_N, dtype=np.uint8)  # 消息位(0/1)
    ldpc_errors = [0]
    bp_decode(log174, max_iterations, plain174, ldpc_errors)
    status.ldpc_errors = ldpc_errors[0]

    if status.ldpc_errors > 0:
        return False, message, status

    # 提取有效载荷 + CRC (前FTX_LDPC_K位)打包到字节数组中
    a91 = pack_bits(plain174, FT8_LDPC_K)

    # 提取CRC并检查它
    status.crc_extracted = ftx_extract_crc(a91)
    # [1]: 'CRC是在源编码消息上计算的，从77位零扩展到82位。'
    a91[9] &= 0xF8
    a91[10] &= 0x00
    status.crc_calculated = ftx_compute_crc(a91, 96 - 14)

    if status.crc_extracted != status.crc_calculated:
        return False, message, status

    # 重用CRC值作为消息的哈希(TODO: 仅14位，也许应该使用完整的16或32位?)
    message.hash = status.crc_calculated

    for i in range(10):
        message.payload[i] = a91[i]

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
                      max_iterations: int = 20) -> List[Tuple[FT8Message, FT8DecodeStatus]]:
    """解码FT8消息的主函数"""
    # 计算频谱图
    spectrogram, f, t = calculate_spectrogram(
        wave_data, sample_rate, bins_per_tone, steps_per_symbol
    )
    
    # import matplotlib.pyplot as plt
    # # 绘制频谱图
    # plt.figure(figsize=(10, 6))
    # plt.imshow(spectrogram, aspect='auto', origin='lower')
    # plt.colorbar(label='强度 (dB)')
    # plt.title('FT8信号频谱图')
    # plt.xlabel('时间 (秒)')
    # plt.ylabel('频率 (Hz)')
    # plt.savefig('ft8_spectrogram2222.png')
    # plt.close()

    # 创建瀑布数据结构
    wf = create_waterfall_from_spectrogram(
        spectrogram, steps_per_symbol, bins_per_tone
    )
    
    # 查找候选信号
    candidates = ft8_find_candidates(wf, max_candidates, min_score)

    print(f"找到的候选信号数量: {len(candidates)}")
    print(f"候选信号: {candidates}")
    # 解码候选信号
    results = []
    for cand in candidates:
        success, message, status = ft8_decode_candidate(wf, cand, max_iterations)
        if success:
            results.append((message, status))
    
    return results 


