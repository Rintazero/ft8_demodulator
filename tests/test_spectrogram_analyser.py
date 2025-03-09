import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ft8_demodulator.spectrogram_analyser import SpectrogramAnalyser

import unittest


# 确保 ft8_generator 的路径在导入之前被添加
sys.path.append('E:/Projects/ft8_generator')
import ft8_generator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt


__name__ = "__main__"

class TestSpectrogramAnalyser(unittest.TestCase):
    def test_plot_spectrogram(self):
        # 生成一个测试信号
        fs = 2e3
        f0 = 300
        fc = 500
        payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
        wave = ft8_generator.ft8_generator(payload,fs=fs,f0=f0,fc=fc)

        # wave_data = np.zeros(len(wave),dtype=np.complex128)
        
        wave_data = wave * np.exp(-1 * 1j * 2 * np.pi * fc * np.arange(len(wave))/fs)

        

        analyser = SpectrogramAnalyser(wave_data,fs,10,10,0,500,2.5)
        analyser.plot_noise_baseline()

        # f,t,spectrogram = SpectrogramAnalyser._calculate_spectrogram(wave_data, fs, 10, 10)
        
        # spectrogram = np.concatenate((spectrogram[len(spectrogram)//2+1:],spectrogram[:len(spectrogram)//2]),axis=0)

        spectrogram = analyser.spectrogram

        plt.imshow(10 * np.log10(np.abs(spectrogram)), aspect='auto', origin='lower',extent=[0, len(wave)/fs, -fs/2, fs/2])
        plt.colorbar(label='Intensity (dB)')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()
        # print("wave: ", wave)

if __name__ == "__main__":
    unittest.main()
