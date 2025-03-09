import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ft8_demodulator.spectrogram_analyse import calculate_spectrogram,select_frequency_band,Spectrogram,selcet_correlation_position_candidates

import unittest

# 确保 ft8_generator 的路径在导入之前被添加
sys.path.append('E:/Projects/ft8_generator')
import ft8_generator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt


__name__ = "__main__"

class TestSpectrogramAnalyse(unittest.TestCase):
    def test_calculate_spectrogram(self):
        # 生成一个测试信号
        fs = 2e3
        f0 = 300
        fc = 500
        payload = np.array([0x1C, 0x3F, 0x8A, 0x6A, 0xE2, 0x07, 0xA1, 0xE3, 0x94, 0x51], dtype=np.uint8)
        wave = ft8_generator.ft8_generator(payload,fs=fs,f0=f0,fc=fc)

        # wave_data = np.zeros(len(wave),dtype=np.complex128)
        
        
        wave_data = wave * np.exp(-1 * 1j * 2 * np.pi * fc * np.arange(len(wave))/fs) # 模拟解调,信号应该在 f0 处, 且 fc+f0 处有待被低通滤除的镜像信号

        spectrogram,f,t = calculate_spectrogram(wave_data,fs,10,10)
        spectrogram,f = select_frequency_band(spectrogram,f,0,500)
        candidates = selcet_correlation_position_candidates(spectrogram,10,10,-0.16,0.16*2,10)

        print(spectrogram.shape,f.shape,t.shape)

        # wf = Spectrogram(fs,10,10,79,0,500)

        # for i in range(len(wave_data)//wf.num_samples_per_step):
        #     wf.step(wave_data[i*wf.num_samples_per_step:(i+1)*wf.num_samples_per_step])
        # wf.show_spectrogram()

        # 绘制频谱图
        plt.imshow(10 * np.log10(np.abs(spectrogram)), aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]])
        plt.colorbar(label='Intensity (dB)')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()

        # 选择频带
        spectrogram,f = select_frequency_band(spectrogram,f,0,500)

        # 绘制频谱图
        plt.imshow(10 * np.log10(np.abs(spectrogram)), aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]])
        plt.colorbar(label='Intensity (dB)')
        plt.title('Spectrogram')
        plt.show()

if __name__ == "__main__":
    unittest.main()

