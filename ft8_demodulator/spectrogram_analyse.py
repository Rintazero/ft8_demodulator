import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

FT8_BAUD_RATE = 6.25 # symbols per second
FT8_SYMBOL_DURATION_S = 1 / FT8_BAUD_RATE
FT8_SYMBOL_FREQ_INTERVAL_HZ = 6.25

SPECTROGRAM_BINS_PER_TONE = 10
SPECTROGRAM_STEPS_PER_SYMBOL = 10

FT8_NUM_SYNC_SEQUENCE = 3
FT8_NUM_SYNC_SYMBOLS_PER_SEQUENCE = 7
FT8_SYNC_PATTERN = [3, 1, 4, 0, 6, 5, 2]
FT8_SYNC_SEQUENCE_OFFSET = 36

class Spectrogram:
    def __init__(self,fs,bins_per_tone,steps_per_symbol,num_symbols,min_freq,max_freq,):
        self.fs = fs                                        # 采样率
        self.bins_per_tone = bins_per_tone                  # 每个 tone 对应的 fblock 数 (频率过采样系数)
        self.steps_per_symbol = steps_per_symbol            # 每个 symbol 对应的 tblock 数 (时间过采样系数)
        self.num_tblocks = num_symbols * steps_per_symbol    # 频谱图的列数 (符号数 * 时间过采样系数)
        self.min_freq = min_freq                              # 最小频率
        self.max_freq = max_freq

        self.nfft = int(self.fs / FT8_SYMBOL_FREQ_INTERVAL_HZ * self.bins_per_tone)
        self.bin_width = self.fs / self.nfft                    # 每个 fblock 对应的频率间隔

        self.num_samples_per_step = int(FT8_SYMBOL_DURATION_S * self.fs / self.steps_per_symbol)

        self.idxMinFreq = int(self.min_freq / self.bin_width)
        self.idxMaxFreq = int(self.max_freq / self.bin_width)
        self.num_fblocks = self.idxMaxFreq - self.idxMinFreq
        self.spectrogram = np.ndarray(shape=(self.num_fblocks,self.num_tblocks))
        self.tframe = np.zeros(shape=(self.nfft,),dtype=np.complex128)        # 时间帧 一次移入一个 tblock
        self.fframe = np.zeros(shape=(self.nfft,),dtype=np.complex128)        # 频率帧
        self.twindow = np.hanning(self.nfft)

    def step(self,wave_data):
        ''' 移入一个 tblock 对应时间长度的数据 进行 STFT '''
        self.tframe[0:self.nfft-self.num_samples_per_step] = self.tframe[self.num_samples_per_step:self.nfft]
        self.tframe[self.nfft-self.num_samples_per_step:self.nfft] = wave_data
        
        self.fframe = np.fft.fft(self.tframe * self.twindow)
        self.fframe = np.fft.fftshift(self.fframe)
        fframe_mag = np.abs(self.fframe[self.nfft//2+self.idxMinFreq:self.nfft//2+self.idxMaxFreq])
        fframe_mag_db = 20 * np.log10(fframe_mag)
        self.spectrogram[:,0:self.num_tblocks-1] = self.spectrogram[:,1:self.num_tblocks]
        self.spectrogram[:,self.num_tblocks-1] = fframe_mag_db
    
    def show_spectrogram(self):
        plt.imshow(self.spectrogram,aspect='auto',origin='lower')
        plt.colorbar()
        plt.show()


def calculate_spectrogram(wave_data,sample_rate,spectrogram_bins_per_tone,spectrogram_steps_per_symbol) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """ 计算频谱图 """
    spectrogram_time_step_s = FT8_SYMBOL_DURATION_S / spectrogram_steps_per_symbol
    spectrogram_freq_step_Hz = FT8_SYMBOL_FREQ_INTERVAL_HZ / spectrogram_bins_per_tone

    samples_per_symbol = int(sample_rate / FT8_SYMBOL_FREQ_INTERVAL_HZ)

    dft_length = samples_per_symbol * spectrogram_bins_per_tone
    overlap_samples = samples_per_symbol - samples_per_symbol // spectrogram_steps_per_symbol

    f,t,spectrogram = sci.signal.spectrogram(wave_data,fs=sample_rate,window='hann',nperseg=samples_per_symbol,noverlap=overlap_samples,nfft=dft_length,detrend=False,return_onesided=False,scaling='spectrum')

    spectrogram = np.concatenate((spectrogram[len(spectrogram)//2+1:],spectrogram[:len(spectrogram)//2]),axis=0)
    spectrogram = 10*np.log10(np.abs(spectrogram))
    
    f = np.concatenate((f[len(f)//2+1:],f[:len(f)//2]))

    return spectrogram,f,t
    
def select_frequency_band(spectrogram,f,start_freq,end_freq) -> tuple[np.ndarray,np.ndarray]:
    """ 选择频带 """
    if start_freq > end_freq:
        raise ValueError("start_freq must be less than end_freq")
    elif start_freq < np.min(f) or end_freq > np.max(f):
        raise ValueError("start_freq or end_freq is out of range")
    start_index = np.argmin(np.abs(f - start_freq))
    end_index = np.argmin(np.abs(f - end_freq))
    return spectrogram[start_index:end_index,:],f[start_index:end_index]

class CorrelationPositionCandidate:
    def __init__(self,frequency,time,score):
        self.frequency = frequency
        self.time = time
        self.score = score

def selcet_correlation_position_candidates(spectrogram,bins_per_tone,steps_per_symbol,start_time_s,end_time_s,MaxNumCandidates) -> list[CorrelationPositionCandidate]:
    """ 选择相关位置候选 """
    candidates = []
    start_time_idx = start_time_s // FT8_SYMBOL_DURATION_S * steps_per_symbol
    end_time_idx = end_time_s // FT8_SYMBOL_DURATION_S * steps_per_symbol

    for time_idx in range(start_time_idx,end_time_idx):
        # 遍历 candidate 时间位置
        for freq_idx in range(np.shape(spectrogram)[0]):
            # 遍历 candidate 频率位置
            numAvg = 0
            sym = 0
            score = 0
            for seq_idx in range(FT8_NUM_SYNC_SEQUENCE):
                for sym_idx in range(FT8_NUM_SYNC_SYMBOLS_PER_SEQUENCE):
                    symIdxAbs = time_idx + (FT8_SYNC_SEQUENCE_OFFSET*seq_idx + sym_idx) * steps_per_symbol
                    if symIdxAbs < 0:
                        continue
                    elif symIdxAbs >= np.shape(spectrogram)[1]:
                        break
                    else:
                        sym = FT8_SYNC_PATTERN[sym_idx]
                        freqIdxAbs = freq_idx + sym * bins_per_tone
                        if freqIdxAbs < 0 or freqIdxAbs >= np.shape(spectrogram)[0]:
                            continue
                        # 累计与相邻块的差值
                        if sym > 0 and freqIdxAbs - bins_per_tone >= 0:
                            score += spectrogram[freqIdxAbs,symIdxAbs] - spectrogram[freqIdxAbs - bins_per_tone,symIdxAbs]
                            numAvg += 1
                        if sym < FT8_NUM_SYNC_SYMBOLS_PER_SEQUENCE - 1 and freqIdxAbs + bins_per_tone < np.shape(spectrogram)[0]:
                            score += spectrogram[freqIdxAbs,symIdxAbs] - spectrogram[freqIdxAbs + bins_per_tone,symIdxAbs]
                            numAvg += 1
                        if sym_idx > 0 and symIdxAbs - steps_per_symbol >= 0:
                            score += spectrogram[freqIdxAbs,symIdxAbs] - spectrogram[freqIdxAbs,symIdxAbs - steps_per_symbol]
                            numAvg += 1
                        if sym_idx < FT8_NUM_SYNC_SYMBOLS_PER_SEQUENCE - 1 and symIdxAbs + steps_per_symbol < np.shape(spectrogram)[1]:
                            score += spectrogram[freqIdxAbs,symIdxAbs] - spectrogram[freqIdxAbs,symIdxAbs + steps_per_symbol]
                            numAvg += 1
            if numAvg > 0:
                score /= numAvg
            if len(candidates) < MaxNumCandidates or score > max([candidate.score for candidate in candidates]):
                candidates.append(CorrelationPositionCandidate(freq_idx,time_idx,score))

    


    return candidates

    

    
