import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

FT8_BAUD_RATE = 6.25 # symbols per second
FT8_SYMBOL_DURATION_S = 1 / FT8_BAUD_RATE
FT8_SYMBOL_FREQ_INTERVAL_HZ = 6.25

class SpectrogramAnalyser:
    """ 频谱分析器 """
    _BASELINE_SEGMENT_NUM = 10
    _MIN_PSD_PERCENT_FOR_BASELINE = 10 # 10%

    def __init__(self,wave,sample_rate,bins_per_tone,steps_per_symbol,min_freq_Hz,max_freq_Hz,time_offset_bound_s):
        self.wave = wave
        self.sample_rate = sample_rate
        self.spectrogram_bins_per_tone = bins_per_tone # freq oversampling
        self.spectrogram_steps_per_symbol = steps_per_symbol # time oversampling

        # Tuning contants for correlation
        self.correlation_min_freq_Hz = min_freq_Hz # Lowest frequency to search for signals
        self.correlation_max_freq_Hz = max_freq_Hz # Highest frequency to search for signals
        self.correlation_time_offset_bound_s = time_offset_bound_s # Time offset bound for correlation (+/- correlation_time_offset_bound_s)

        # 
        self.spectrogram_time_step_s = FT8_SYMBOL_DURATION_S / self.spectrogram_steps_per_symbol
        self.spectrogram_freq_step_Hz = FT8_SYMBOL_FREQ_INTERVAL_HZ / self.spectrogram_bins_per_tone

        self.spectrogram = self._calculate_spectrogram(self.wave,self.sample_rate,self.spectrogram_bins_per_tone,self.spectrogram_steps_per_symbol)
        self.baseline = self._calculate_baseline(self.spectrogram,self.correlation_min_freq_Hz,self.correlation_max_freq_Hz,self.spectrogram_bins_per_tone)



    @staticmethod
    def _calculate_spectrogram(wave_data,sample_rate,spectrogram_bins_per_tone,spectrogram_steps_per_symbol):
        """ 计算频谱图 """
        # 计算频谱图
        
        samples_per_symbol = int(sample_rate / FT8_SYMBOL_FREQ_INTERVAL_HZ)

        dft_length = samples_per_symbol * spectrogram_bins_per_tone

        overlap_samples = samples_per_symbol - samples_per_symbol // spectrogram_steps_per_symbol

        f,t,spectrogram = sci.signal.spectrogram(wave_data,fs=sample_rate,window='hann',nperseg=samples_per_symbol,noverlap=overlap_samples,nfft=dft_length,detrend=False,return_onesided=False,scaling='spectrum')

        # f = np.concatenate((f[len(f)//2+1:],f[:len(f)//2]))

        spectrogram = np.concatenate((spectrogram[len(spectrogram)//2+1:],spectrogram[:len(spectrogram)//2]),axis=0)
        
        return spectrogram

    @staticmethod
    def _calculate_baseline(spectrogram,min_freq_Hz,max_freq_Hz,spectrogram_bins_per_tone):
        """ 计算基线 """
        low_freq_idx = int(min_freq_Hz / FT8_SYMBOL_FREQ_INTERVAL_HZ * spectrogram_bins_per_tone)
        high_freq_idx = int(max_freq_Hz / FT8_SYMBOL_FREQ_INTERVAL_HZ * spectrogram_bins_per_tone)
        spectrogram_freq_mean_db = 10 * np.log10(np.mean(spectrogram[low_freq_idx:high_freq_idx,:],axis=1)) # 对应频率下 功率(dB) 均值

        # Divide the PSD(Power Spectral Density) into X segments
        indexs = np.arange(high_freq_idx - low_freq_idx)
        segments = np.array_split(indexs,SpectrogramAnalyser._BASELINE_SEGMENT_NUM)

        base_indexs = []
        base_values = []

        for segment in segments:
            segment_mean_db = spectrogram_freq_mean_db[segment]
            base = np.percentile(segment_mean_db,SpectrogramAnalyser._MIN_PSD_PERCENT_FOR_BASELINE)
            selectors = segment_mean_db <= base
            base_indexs.append(segment[selectors])
            base_values.append(segment_mean_db[selectors])

        base_indexs = np.concatenate(base_indexs)
        base_values = np.concatenate(base_values)

        midpoint = (high_freq_idx - low_freq_idx) // 2
        base_indexs -= midpoint

        poly = np.polynomial.Polynomial(np.polynomial.polynomial.polyfit(base_indexs,base_values,SpectrogramAnalyser._BASELINE_SEGMENT_NUM//2-1))

        baseline = poly(indexs - midpoint)

        return baseline

    def noise_baseline(self, freq):
        """ Get the noise baseline for a given frequency or array of frequencies """

        index = np.rint((freq - self.correlation_min_freq_Hz) / FT8_SYMBOL_FREQ_INTERVAL_HZ * self.spectrogram_bins_per_tone).astype(int)
        psd = np.power(10,0.1*self.baseline[index])
        return psd

    def plot_noise_baseline(self):
        """ Plot the noise baseline """
        f = np.linspace(self.correlation_min_freq_Hz,self.correlation_max_freq_Hz,self.baseline.size)
        plt.plot(f,self.baseline)
        plt.title('Noise Baseline')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Baseline Intensity (dB/Hz)')
        plt.show()
