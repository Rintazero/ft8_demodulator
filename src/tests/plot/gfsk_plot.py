import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ft8_tools.ft8_generator import crc, ldpc, encoder, modulator

def plot_full_waveform():
    payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
    itones = encoder.ft8_encode(payload)
    gfsk_waveform = modulator.gfsk_modulation_waveform_generator(itones,10e3)
    
    # 创建时间轴
    t = np.arange(len(gfsk_waveform)) / 10e3  # 采样率为 10kHz
    
    # 计算同步部分的范围（第2-8个符号）
    samples_per_tone = int(len(gfsk_waveform) / len(itones))
    sync_start = samples_per_tone  # 从第二个符号开始
    sync_length = 7 * samples_per_tone  # 7个符号长度
    
    # 绘制波形
    plt.figure(figsize=(12, 6))
    # 绘制第一个符号（蓝色）
    plt.plot(t[:sync_start], gfsk_waveform[:sync_start], 'b-', label='Data Symbol')
    # 绘制同步部分（红色）
    plt.plot(t[sync_start:sync_start+sync_length], 
            gfsk_waveform[sync_start:sync_start+sync_length], 
            'r-', label='Sync Sequence')
    # 绘制剩余数据部分（蓝色）
    plt.plot(t[sync_start+sync_length:], 
            gfsk_waveform[sync_start+sync_length:], 
            'b-', label='_nolegend_')  # _nolegend_防止重复显示标签
    plt.grid(True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.title('GFSK Modulation Waveform')
    plt.legend()
    plt.savefig('gfsk_waveform_full.png')
    plt.close()

def plot_sync_detail():
    payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
    itones = encoder.ft8_encode(payload)
    gfsk_waveform = modulator.gfsk_modulation_waveform_generator(itones,10e3)
    
    # 创建时间轴
    samples_per_tone = int(len(gfsk_waveform) / len(itones))
    detail_length = 8 * samples_per_tone  # 显示前8个符号
    t = np.arange(detail_length) / 10e3
    
    # 计算同步序列的范围
    sync_start = samples_per_tone  # 从第二个符号开始
    sync_length = 7 * samples_per_tone  # 7个符号长度
    
    # 绘制波形
    plt.figure(figsize=(15, 8))
    # 绘制第一个符号（蓝色）
    plt.plot(t[:sync_start], gfsk_waveform[:sync_start], 'b-', label='Data Symbol')
    # 绘制同步部分（红色）
    plt.plot(t[sync_start:sync_start+sync_length], 
            gfsk_waveform[sync_start:sync_start+sync_length], 
            'r-', label='Sync Sequence')
    
    plt.grid(True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('GFSK Modulation Waveform - First 8 Symbols')
    plt.legend()
    plt.savefig('gfsk_waveform_sync.png', bbox_inches='tight')
    plt.close()

def plot_with_freq_offset():
    payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
    itones = encoder.ft8_encode(payload)
    gfsk_waveform = modulator.gfsk_modulation_waveform_generator(itones,10e3)
    
    # 创建时间轴
    t = np.arange(len(gfsk_waveform)) / 10e3  # 采样率为 10kHz
    
    # 直接添加线性频偏（斜率为0.001的线性函数）
    slope = 0.001  # 调整斜率大小
    offset_waveform = gfsk_waveform + slope * np.arange(len(gfsk_waveform))
    
    # 计算同步部分的范围（第2-8个符号）
    samples_per_tone = int(len(gfsk_waveform) / len(itones))
    sync_start = samples_per_tone
    sync_length = 7 * samples_per_tone
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制原始波形
    ax1.plot(t[:sync_start], np.real(gfsk_waveform[:sync_start]), 'b-', label='Data Symbol')
    ax1.plot(t[sync_start:sync_start+sync_length], 
            np.real(gfsk_waveform[sync_start:sync_start+sync_length]), 
            'r-', label='Sync Sequence')
    ax1.plot(t[sync_start+sync_length:], 
            np.real(gfsk_waveform[sync_start+sync_length:]), 
            'b-', label='_nolegend_')
    ax1.grid(True)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('Original GFSK Waveform')
    ax1.legend()
    
    # 绘制带频偏的波形
    ax2.plot(t[:sync_start], np.real(offset_waveform[:sync_start]), 'b-', label='Data Symbol')
    ax2.plot(t[sync_start:sync_start+sync_length], 
            np.real(offset_waveform[sync_start:sync_start+sync_length]), 
            'r-', label='Sync Sequence')
    ax2.plot(t[sync_start+sync_length:], 
            np.real(offset_waveform[sync_start+sync_length:]), 
            'b-', label='_nolegend_')
    ax2.grid(True)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('GFSK Waveform with Linear Offset')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('gfsk_waveform_with_offset.png')
    plt.close('all')

if __name__ == "__main__":
    plot_full_waveform()
    plot_sync_detail()
    plot_with_freq_offset()
    