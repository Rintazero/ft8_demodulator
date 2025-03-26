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


StartTime = datetime.datetime(year = 2024, month = 5, day = 31, hour = 16, minute = 5, second = 51)

duration_s = 2000 * 10
delta_t_s = 10
num_samples = int(duration_s / delta_t_s)

normalized_doppler_frequency_shift_seq = numpy.zeros(num_samples)
elevation_deg_groundStation_to_satellite_seq = numpy.zeros(num_samples)

for i in range(num_samples):
    # i = i - num_samples // 2
    t = StartTime - datetime.timedelta(seconds = (i - num_samples // 2) * delta_t_s)

    normalized_doppler_frequency_shift_by_ecef = channel.calculate_normalized_doppler_frequency_shift_by_ecef(t)
    elevation_deg_groundStation_to_satellite = channel.calculate_elevation_groundStation_to_satellite(t)

    normalized_doppler_frequency_shift_seq[i] = normalized_doppler_frequency_shift_by_ecef
    elevation_deg_groundStation_to_satellite_seq[i] = elevation_deg_groundStation_to_satellite

# 可视化 normalized_doppler_frequency_shift 和 elevation_deg_groundStation_to_satellite 随时间的变化
import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax1 = plt.subplots()

# Plot normalized_doppler_frequency_shift
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Normalized Doppler Frequency Shift', color='tab:blue')
ax1.plot([i * delta_t_s for i in range(num_samples)], normalized_doppler_frequency_shift_seq, color='tab:blue', label='Normalized Doppler Frequency Shift')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for elevation
ax2 = ax1.twinx()
ax2.set_ylabel('Elevation (degrees)', color='tab:red')
ax2.plot([i * delta_t_s for i in range(num_samples)], elevation_deg_groundStation_to_satellite_seq, color='tab:red', label='Elevation')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Title and grid
plt.title('Normalized Doppler Frequency Shift and Elevation Over Time')
fig.tight_layout()  # to ensure the right y-label is not slightly clipped
plt.grid()

# Show the plot
plt.show()


# normalized_doppler_frequency_shift_by_eci = channel.calculate_normalized_doppler_frequency_shift_by_eci(t)


# print(normalized_doppler_frequency_shift_by_ecef)
# print(normalized_doppler_frequency_shift_by_eci)
# print(elevation_deg_groundStation_to_satellite)


