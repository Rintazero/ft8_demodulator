import datetime
import numpy
import sys
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import scipy.io.wavfile as wavfile

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ft8_tools.channel.channel import Channel
from ft8_tools.ft8_generator import ft8_baseband_generator

fc_Hz = 2.45e9
fs_Hz = 50e3
f0_Hz = 100
SignalTime_s = 20

save_path = "./src/tests/channel/doppler_shift_test/"

satelliteTLE = {
    "name": "StarLink-1030",
    "TLE_line1": "1 44735U 19074Y   24151.67073227  .00005623  00000+0  39580-3 0  9994",
    "TLE_line2": "2 44735  53.0540 235.6876 0001395  85.6354 274.4795 15.06429209250797",
}

groundStation = {
    "name": "Station",
    "latitude_deg": 20.75046789797617,
    "longitude_deg": 116.55005431954011,
    "altitude_m": 0,
}

channel = Channel(groundStation,satelliteTLE)

StartTime = datetime.datetime(year = 2024, month = 5, day = 31, hour = 16, minute = 5, second = 51, microsecond = 0)

satellite_overhead_time_prediction_candidates = channel.satellite_overhead_time_prediction(StartTime, StartTime + datetime.timedelta(days = 2), 30)

# print(satellite_overhead_time_prediction_candidates)

selected_candidate = satellite_overhead_time_prediction_candidates[1]

channel.get_overhead_prediction_candidate_info(selected_candidate[0], selected_candidate[1], is_save_fig=True, save_fig_path=save_path)

SignalTxTimeStamp = selected_candidate[0] + selected_candidate[1] // 2 - datetime.timedelta(seconds = SignalTime_s // 2)

channel.get_doppler_frequency_shift_sequence(SignalTxTimeStamp, SignalTime_s, fs_Hz, fc_Hz, save_path=save_path)