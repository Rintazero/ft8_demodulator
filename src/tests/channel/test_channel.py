import datetime
import numpy
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ft8_tools.channel.channel import Channel

satelliteTLE = {
    "name": "StarLink-1030",
    "TLE_line1": "1 44735U 19074Y   24151.67073227  .00005623  00000+0  39580-3 0  9994",
    "TLE_line2": "2 44735  53.0540 235.6876 0001395  85.6354 274.4795 15.06429209250797",
}

groundStation = {
    "name": "Station",
    "latitude_deg": 20.7634433315784,
    "longitude_deg": 116.560494091634,
    "altitude_m": 0,
}

channel = Channel(groundStation,satelliteTLE)

timestamp = datetime.datetime(year = 2024, month = 5, day = 31, hour = 16, minute = 5, second = 51)

t = timestamp + datetime.timedelta(minutes = 0)

normalized_doppler_frequency_shift_by_ecef = channel.calculate_normalized_doppler_frequency_shift_by_ecef(t)
normalized_doppler_frequency_shift_by_eci = channel.calculate_normalized_doppler_frequency_shift_by_eci(t)

print(normalized_doppler_frequency_shift_by_ecef)
print(normalized_doppler_frequency_shift_by_eci)



