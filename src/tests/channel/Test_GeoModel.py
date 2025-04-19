import numpy as np
import matplotlib.pyplot as plt

# Constants
R_E = 6371.393  # km
c = 299792.458  # m/s
alpha_t0 = np.radians(60)  # maximum elevation angle in radians
omega_ECI = 2 * np.pi / (1.5865525713 * 60 * 60)  # rad/s
omega_E = 7.292e-5  # rad/s
h = 535  # km
i = np.radians(60)  # inclination in radians

# Calculations
omega_ECEF = (omega_ECI - omega_E * np.cos(i))
r = R_E + h
gamma_t0 = np.arccos(R_E/r * np.cos(alpha_t0)) - alpha_t0

alpha_v = np.radians(10)
gamma_v = np.arccos(R_E/r * np.cos(alpha_v)) - alpha_v
tau = 2/omega_ECEF * np.arccos(np.cos(gamma_v)/np.cos(gamma_t0))

# Time range
StartT_s = -10 * 60
EndT_s = 10 * 60

# Initialize frequency array
f_d = np.zeros(EndT_s - StartT_s + 1)

# Calculate Doppler frequency
for t_s in range(StartT_s, EndT_s + 1):
    dphi_t = omega_ECEF * t_s
    denominator = np.sqrt(R_E*R_E + r*r - 2*r*R_E*np.cos(gamma_t0)*np.cos(dphi_t))
    f_d[t_s - StartT_s] = -1/c * (r*R_E*np.cos(gamma_t0)*np.sin(dphi_t)*omega_ECEF) / denominator

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(f_d)
plt.xlabel('Time (s)')
plt.ylabel('Doppler Frequency (Hz)')
plt.title('Doppler Frequency vs Time')
plt.grid(True)
plt.show()

start_angle = 10
end_angle = 90


tau_list = []

for angle in range(start_angle, end_angle + 1):
    alpha_t0 = np.radians(angle)
    gamma_t0 = np.arccos(R_E/r * np.cos(alpha_t0)) - alpha_t0
    tau = 2/(omega_ECEF) * np.arccos(np.cos(gamma_v)/np.cos(gamma_t0))
    tau_list.append(tau)

plt.figure(figsize=(10, 6))
plt.plot(tau_list)
plt.xlabel('Angle (deg)')
plt.ylabel('Tau (s)')
plt.title('Tau vs Angle')
plt.grid(True)
plt.show()

# Initialize frequency array


StartT_angle = 0
EndT_angle = 90
f_d = np.zeros(EndT_angle - StartT_angle + 1)
# Calculate Doppler frequency
for angle in range(StartT_angle, EndT_angle + 1):
    alpha = np.radians(angle)
    gamma = np.arccos(R_E/r * np.cos(alpha)) - alpha
    dphi_t = np.arccos(np.cos(gamma)/np.cos(gamma_t0))
    denominator = np.sqrt(R_E*R_E + r*r - 2*r*R_E*np.cos(gamma_t0)*np.cos(dphi_t))
    f_d[angle - StartT_angle] = 1/c * (r*R_E*np.cos(gamma_t0)*np.sin(dphi_t)*omega_ECEF) / denominator

plt.figure(figsize=(10, 6))
plt.plot(f_d)
plt.xlabel('Angle (deg)')
plt.ylabel('Doppler Frequency (Hz)')
plt.title('Doppler Frequency vs Angle')
plt.grid(True)
plt.show()