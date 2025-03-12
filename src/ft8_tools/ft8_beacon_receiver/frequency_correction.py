import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# 设置matplotlib字体设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用DejaVu Sans以获得更好的兼容性
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

def gfsk_pulse(bt, t):
    """
    生成GFSK脉冲
    
    参数:
    bt: 带宽时间乘积
    t: 时间序列
    
    返回:
    output: GFSK脉冲
    """
    k = np.pi * np.sqrt(2.0/np.log(2.0))
    output = 0.5 * (scipy.special.erf(k*bt*(t+0.5)) - scipy.special.erf(k*bt*(t-0.5)))
    return output

def correct_frequency_drift(wave_complex, fs, sym_bin, sym_t, waterfall_freq_range=None, params=None):
    """
    对FT8信号进行频率漂移校正
    
    参数:
    wave_complex: 复数形式的输入信号
    fs: 采样率
    sym_bin: 符号频率间隔
    sym_t: 符号时间长度
    waterfall_freq_range: 瀑布图频率范围 (默认: None，表示使用全部STFT频域范围)
    params: 可选参数字典，包含以下字段：
        - time_osr: 时间过采样率 (默认: 2)
        - freq_osr: 频率过采样率 (默认: 2)
        - nsync_sym: 同步符号数量 (默认: 7)
        - ndata_sym: 数据符号数量 (默认: 58)
        - zscore_threshold: Z分数阈值 (默认: 5)
        - max_iteration_num: 最大迭代次数 (默认: 400)
        - debug_plots: 是否生成调试图 (默认: False)
    
    返回:
    tuple: (校正后的信号, 估计的频率漂移率)
        - 校正后的信号: 采样率与输入信号相同(fs)的复数信号
        - 估计的频率漂移率: 每秒的频率漂移率(Hz/s)
    """
    # 设置默认参数
    default_params = {
        'time_osr': 2,
        'freq_osr': 2,
        'nsync_sym': 7,
        'ndata_sym': 58,
        'zscore_threshold': 5,
        'max_iteration_num': 400,
        'debug_plots': False
    }
    
    if params is None:
        params = default_params
    else:
        for key, value in default_params.items():
            if key not in params:
                params[key] = value

    # 基本变量
    time_osr = params['time_osr']
    freq_osr = params['freq_osr']
    nsync_sym = params['nsync_sym']
    ndata_sym = params['ndata_sym']
    zscore_threshold = params['zscore_threshold']
    max_iteration_num = params['max_iteration_num']
    debug_plots = params['debug_plots']
    
    # 信号长度
    nsamples = len(wave_complex)
    tlength = nsamples / fs

    # STFT 分析
    nsps = int(sym_t * fs)
    nfft = int(fs * sym_t * freq_osr)
    hann_window = scipy.signal.windows.hann(M=nfft, sym=False)
    sft = scipy.signal.ShortTimeFFT(win=hann_window, hop=nsps//time_osr, fs=fs, fft_mode="onesided")
    sx = sft.stft(np.real(wave_complex))
    
    # 频率范围过滤
    freqs = np.linspace(0, fs/2, sx.shape[0])
    
    # 如果未指定频率范围，则使用全部STFT频域范围
    if waterfall_freq_range is None:
        waterfall_freq_range = (0, fs/2)
        
    freq_mask = (freqs >= waterfall_freq_range[0]) & (freqs <= waterfall_freq_range[1])
    sx_filtered = sx[freq_mask, :]
    sx_filtered_real = np.real(sx_filtered)
    sx_filtered_imag = np.imag(sx_filtered)
    sx_filtered_power = sx_filtered_real**2 + sx_filtered_imag**2
    sx_filtered_db = 10 * np.log10(sx_filtered_power)

    # 计算最大幅度频率-时间序列
    window_sum = np.zeros(sx_filtered_db.shape)
    for i in range(sx_filtered_db.shape[0]):
        for j in range(sx_filtered_db.shape[1]):
            if i < sx_filtered_db.shape[0] - freq_osr:
                window_sum[i][j] = np.sum(sx_filtered_db[i:i+freq_osr, j])
            else:
                window_sum[i][j] = np.sum(sx_filtered_db[i:, j])
                
    window_indices = np.argmax(window_sum, axis=0)
    max_freq_indices = np.zeros(sx_filtered_db.shape[1])
    for i in range(sx_filtered_db.shape[1]):
        max_freq_indices[i] = window_indices[i] + np.argmax(sx_filtered_db[window_indices[i]:window_indices[i]+freq_osr, i])
    max_freq_indices = max_freq_indices.astype(int)
    
    

    # 计算实际频率
    freqs_filtered = freqs[freq_mask]
    max_freqs = waterfall_freq_range[0] + freqs_filtered[max_freq_indices]
    
    # 处理前绘制max_freqs
    if debug_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, tlength, len(max_freqs)), max_freqs, marker='o', linestyle='-', color='blue', label='Original Data')
        plt.title('Maximum Frequency vs Time Before Processing')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True)
        plt.legend()
        plt.xlim(0, tlength)
        plt.ylim(waterfall_freq_range[0], waterfall_freq_range[1])
        plt.savefig('max_frequencies_before_processing.png')
        plt.close()

    # 使用Z分数方法识别异常值
    iteration_num = 0
    outlier_indices = np.array([])
    
    while True:
        x = np.arange(len(max_freq_indices)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, max_freq_indices)
        max_freq_indices_fitted = model.predict(x)
        
        residuals = max_freq_indices - max_freq_indices_fitted
        z_scores = np.abs(stats.zscore(residuals))
        outlier_indices = np.where(z_scores > zscore_threshold)[0]
        outliers = max_freq_indices[outlier_indices]

        if len(outlier_indices) == 0 or iteration_num == max_iteration_num:
            break

        i = np.argmax(z_scores[outlier_indices])
        sx_filtered_db[outliers[i], outlier_indices[i]] = np.min(sx_filtered_db)
        window_indices[outlier_indices[i]] = np.argmax(sx_filtered_db[:, outlier_indices[i]])
        max_freq_indices[outlier_indices[i]] = window_indices[outlier_indices[i]] + np.argmax(
            sx_filtered_db[window_indices[outlier_indices[i]]:window_indices[outlier_indices[i]]+freq_osr, outlier_indices[i]])

        iteration_num += 1

    # 更新最大频率
    max_freqs = waterfall_freq_range[0] + freqs_filtered[max_freq_indices]
    
    # 处理后绘制max_freqs
    if debug_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, tlength, len(max_freqs)), max_freqs, marker='o', linestyle='-', color='blue', label='Processed Data')
        plt.title('Maximum Frequency vs Time After Processing')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True)
        plt.legend()
        plt.xlim(0, tlength)
        plt.ylim(waterfall_freq_range[0], waterfall_freq_range[1])
        
        # 用红色标记异常值
        plt.scatter(outlier_indices * sym_t/time_osr, max_freqs[outlier_indices], color='red', marker='x', s=100, label='Outliers')
        plt.legend()
        plt.savefig('max_frequencies_after_processing.png')
        plt.close()

    # 构造同步序列
    sync_seq = (np.array([3, 1, 4, 0, 6, 5, 2]) + 1)
    sync_seq = sync_seq - np.mean(sync_seq)


    # 每个符号的GFSK脉冲整形
    samples_per_sym = time_osr * 2
    t_pulse = np.linspace(-1, 1, samples_per_sym+1)
    gfsk_shape = gfsk_pulse(bt=2.0, t=t_pulse)

    # 扩展同步序列长度以适应整形脉冲
    sync_correlation_seq = np.zeros((nsync_sym-1) * time_osr + samples_per_sym + 1)

    # 对每个同步符号进行脉冲整形
    for sym_idx in range(nsync_sym):
        sync_correlation_seq[sym_idx * time_osr:(sym_idx * time_osr) + samples_per_sym + 1] += gfsk_shape * sync_seq[sym_idx]

    # 创建三个同步序列
    three_sync_correlation_seq = np.zeros((3*nsync_sym + ndata_sym - 1) * time_osr + 1 + samples_per_sym)



    for i in range(3):
        start_idx = i*(nsync_sym+ndata_sym//2)*time_osr
        end_idx = start_idx + len(sync_correlation_seq)
        three_sync_correlation_seq[start_idx:end_idx] = sync_correlation_seq


        # 绘制three_sync_correlation_seq
    if debug_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(three_sync_correlation_seq, marker='.', linestyle='-', color='b')
        plt.title('Three Sync Correlation Sequence')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.savefig('three_sync_correlation_seq.png')
        plt.close()
        
    # 相关计算
    sync_correlation = np.correlate(max_freqs, three_sync_correlation_seq, mode='full')

    # TODO: 这里需要优化，因为同步序列相关波形在同步序列结束后的值会影响相关峰值的检测
    sync_correlation[:len(three_sync_correlation_seq) -1] = 0
    # 绘制同步相关波形
    if debug_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(sync_correlation, marker='o', linestyle='-', color='b')
        plt.title('Sync Sequence Correlation')
        plt.xlabel('Sample Index')
        plt.ylabel('Correlation Value')
        plt.grid(True)
        plt.savefig('sync_correlation.png')
        plt.close()



    correlation_peak_index = np.argmax(sync_correlation)
    correlation_peak_time_block_index = correlation_peak_index - (len(three_sync_correlation_seq) - 1) + samples_per_sym//2


    print("correlation_peak_time_block_index", correlation_peak_time_block_index)
    # 回归分析
    regression_x = np.array([])
    regression_y = np.array([])

    for i in range(3):
        start_idx = i*(nsync_sym+ndata_sym//2)*time_osr + correlation_peak_time_block_index
        end_idx = start_idx + (nsync_sym-1) * time_osr + 1
        x_step = sym_t/time_osr
        regression_x = np.append(regression_x, np.arange(start_idx, min(end_idx, len(max_freqs))) * x_step)
        regression_y = np.append(regression_y, max_freqs[start_idx:min(end_idx, len(max_freqs))])


    regression_x = regression_x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(regression_x, regression_y)
    f_shift_est_k_hz_ps = model.coef_[0]  # 估计的频率漂移率 (Hz/s)
    
    # 转换为每个采样点的频率漂移率
    f_shift_est_k_hz_psample = f_shift_est_k_hz_ps / fs

    # 频率偏移补偿
    compensation_carrier = np.exp(-2j*np.pi*f_shift_est_k_hz_psample*np.arange(nsamples)**2/(2*fs))
    wave_compensated = wave_complex * compensation_carrier

    # 可视化
    if debug_plots:
        # STFT幅度图
        plt.figure(figsize=(10, 6))
        plt.imshow(sx_filtered_db, aspect='auto', origin='lower', 
                extent=[0, tlength, waterfall_freq_range[0], waterfall_freq_range[1]])
        plt.colorbar(label='Magnitude (dB)')
        plt.title('Short-Time Fourier Transform (STFT) Magnitude (dB)')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.savefig('stft_magnitude.png')
        plt.close()

        # 最大频率随时间变化图
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, tlength, len(max_freqs)), max_freqs, marker='o', linestyle='-', zorder=1)
        plt.title('Maximum Frequency vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid()
        plt.xlim(0, tlength)
        plt.ylim(waterfall_freq_range[0], waterfall_freq_range[1])
        highlight_index = correlation_peak_time_block_index + list(range(nsync_sym*time_osr))
        plt.scatter(highlight_index * sym_t/time_osr, max_freqs[highlight_index], color='red', marker='o', zorder=2)
        plt.scatter(outlier_indices * sym_t/time_osr, max_freqs[outlier_indices], color='green', marker='o', zorder=2)
        plt.grid()
        plt.savefig('max_frequencies.png')
        plt.close()

        # 同步相关图
        plt.figure(figsize=(10, 6))
        plt.plot(sync_correlation, marker='o', linestyle='-', color='b')
        plt.title('Sync Correlation')
        plt.xlabel('Sample Index')
        plt.ylabel('Correlation Value')
        plt.grid()
        plt.xlim(0, len(sync_correlation))
        plt.ylim(np.min(sync_correlation), np.max(sync_correlation))
        plt.savefig('sync_correlation.png')
        plt.close()

        # 回归分析图
        plt.figure(figsize=(10, 6))
        plt.scatter(regression_x, regression_y, color='blue', alpha=0.5, label='Sync Sequence Points')
        plt.plot(regression_x, model.predict(regression_x), color='red', label='Fitted Line')
        plt.title('Frequency Drift Regression Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True)
        plt.legend()
        plt.savefig('regression_analysis.png')
        plt.close()

        # 校正后的信号STFT
        compensated_freq_range = (waterfall_freq_range[0], waterfall_freq_range[0]+200)
        wave_compensated_sx = sft.stft(np.real(wave_compensated))
        compensated_freq_mask = (freqs >= compensated_freq_range[0]) & (freqs <= compensated_freq_range[1])
        wave_compensated_sx_filtered = wave_compensated_sx[compensated_freq_mask, :]
        wave_compensated_sx_filtered_real = np.real(wave_compensated_sx_filtered)
        wave_compensated_sx_filtered_imag = np.imag(wave_compensated_sx_filtered)
        wave_compensated_sx_filtered_power = wave_compensated_sx_filtered_real**2 + wave_compensated_sx_filtered_imag**2
        wave_compensated_sx_filtered_db = 10 * np.log10(wave_compensated_sx_filtered_power)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wave_compensated_sx_filtered_db, aspect='auto', origin='lower', 
                extent=[0, tlength, compensated_freq_range[0], compensated_freq_range[1]])
        plt.colorbar(label='Magnitude (dB)')
        plt.title('Compensated Signal STFT Magnitude (Filtered)')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.savefig('wave_compensated_stft.png')
        plt.close()

    return wave_compensated, f_shift_est_k_hz_psample
