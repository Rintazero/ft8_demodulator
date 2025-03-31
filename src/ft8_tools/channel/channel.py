import pyproj
import skyfield
import skyfield.api
import numpy
import sgp4.api
import datetime
import folium
import numpy as np
import matplotlib.pyplot as plt
import os
from pymap3d.aer import eci2aer
from pymap3d.eci import eci2ecef
from pymap3d.ecef import geodetic2ecef,geodetic2eci,eci2geodetic
from pymap3d.sidereal import datetime2sidereal
import numpy as np
from scipy import stats


class GroundStation:
    def __init__(self,name,latitude_deg,longitude_deg,altitude_m):
        self.name = name
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.altitude_m = altitude_m

    def get_ground_station_position_eci(self,timestamp: datetime.datetime):
        eci_pos = geodetic2eci(self.latitude_deg,self.longitude_deg,self.altitude_m,timestamp)
        return eci_pos

    def get_ground_station_position_ecef(self,timestamp: datetime.datetime):
        ecef_pos = geodetic2ecef(self.latitude_deg,self.longitude_deg,self.altitude_m)
        return ecef_pos

class Channel:
    def __init__(self,groundStation,satelliteTLE):
        self.groundStation = GroundStation(groundStation["name"],groundStation["latitude_deg"],groundStation["longitude_deg"],groundStation["altitude_m"])
        self.satelliteTLE = satelliteTLE
        self.satellite = sgp4.api.Satrec.twoline2rv(satelliteTLE["TLE_line1"],satelliteTLE["TLE_line2"])

    def calculate_normalized_doppler_frequency_shift_by_ecef(self,timestamp: datetime.datetime):
        c = 299792458
        eci_pos = self.groundStation.get_ground_station_position_eci(timestamp)
        gSta_pos_ecef = self.groundStation.get_ground_station_position_ecef(timestamp)
        jd, fr = sgp4.api.jday(timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,timestamp.second+timestamp.microsecond/1e6)
        e,r,v = self.satellite.sgp4(jd, fr)
        
        r = numpy.array(r) * 1e3
        v = numpy.array(v) * 1e3

        sat_pos_ecef = np.array(eci2ecef(r[0], r[1], r[2], timestamp))
        sat_vel_ecef = eci2ecef_velocity(v, timestamp)

        vector_pos_groundStation_to_satellite = sat_pos_ecef - np.array(gSta_pos_ecef)
        vector_pos_groundStation_to_satellite_unit = vector_pos_groundStation_to_satellite / numpy.linalg.norm(vector_pos_groundStation_to_satellite)
        vector_vel_groundStation_to_satellite = numpy.dot(vector_pos_groundStation_to_satellite_unit, sat_vel_ecef)

        
        normalized_doppler_frequency_shift = -1 * vector_vel_groundStation_to_satellite / c

        return normalized_doppler_frequency_shift
    
    def calculate_normalized_doppler_frequency_shift_by_eci(self,timestamp: datetime.datetime):
        c = 299792458
        eci_pos = self.groundStation.get_ground_station_position_eci(timestamp)
        jd, fr = sgp4.api.jday(timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,timestamp.second+timestamp.microsecond/1e6)
        e,r,v = self.satellite.sgp4(jd, fr)
        
        r = numpy.array(r) * 1e3
        v = numpy.array(v) * 1e3

        vector_pos_groundStation_to_satellite = np.array(r) - eci_pos
        vector_pos_groundStation_to_satellite_unit = vector_pos_groundStation_to_satellite / numpy.linalg.norm(vector_pos_groundStation_to_satellite)
        vector_vel_groundStation_to_satellite = numpy.dot(vector_pos_groundStation_to_satellite_unit, v)

        normalized_doppler_frequency_shift = -1 * vector_vel_groundStation_to_satellite / c

        return normalized_doppler_frequency_shift
    
    def calculate_elevation_groundStation_to_satellite(self,timestamp: datetime.datetime):
        # eci_pos = self.groundStation.get_ground_station_position_eci(timestamp)
        jd, fr = sgp4.api.jday(timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,timestamp.second+timestamp.microsecond/1e6)
        e,r,v = self.satellite.sgp4(jd, fr)

        r = numpy.array(r) * 1e3

        az,el,_range = eci2aer(r[0],r[1],r[2],self.groundStation.latitude_deg,self.groundStation.longitude_deg,self.groundStation.altitude_m,timestamp)
        
        return el

    def get_satellite_star_point(self,timestamp: datetime.datetime):
        jd, fr = sgp4.api.jday(timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,timestamp.second+timestamp.microsecond/1e6)
        e,r,v = self.satellite.sgp4(jd, fr)
        r = numpy.array(r) * 1e3
        satellite_geodetic_pos = eci2geodetic(r[0],r[1],r[2],timestamp)
        return satellite_geodetic_pos

    def get_orbital_period(self) -> float:
        """
        计算卫星轨道周期（单位：分钟）
        从 TLE 数据中的平均运动(mean motion)计算
        """
        # 从 TLE 第二行获取平均运动（rev/day）
        mean_motion = float(self.satelliteTLE["TLE_line2"][52:63])
        
        # 计算周期（分钟）
        # 平均运动是每天绕地球的圈数
        # 周期 = 24小时 / 平均运动
        period_minutes = 24 * 60 / mean_motion
        
        return period_minutes

    def satellite_overhead_time_prediction(self, start_time: datetime.datetime, end_time: datetime.datetime, elevation_threshold_deg: float):
        # 计算卫星在给定时间范围内的所有过顶时间
        # 输入: 开始时间, 结束时间, 最小仰角阈值
        # 输出: 过顶时间列表
        
        # 1. 获取卫星轨道数据
        candidates = []
        delta_t = datetime.timedelta(minutes = 1)
        delta_tt = datetime.timedelta(seconds = 1)
        t = start_time
        t_enter = t
        t_leave = t
        overhead_time_minutes = 0
        max_elevation = -90
        while t < end_time:
            max_elevation = -90
            elevation = self.calculate_elevation_groundStation_to_satellite(t)
            if elevation > elevation_threshold_deg:
                t_enter = t
                while elevation > elevation_threshold_deg:
                    t_enter -= delta_tt
                    elevation = self.calculate_elevation_groundStation_to_satellite(t_enter)
                    max_elevation = max(max_elevation,elevation)
                t_enter += delta_tt
                t_leave = t + delta_tt
                elevation = self.calculate_elevation_groundStation_to_satellite(t_leave)
                while elevation > elevation_threshold_deg:
                    t_leave += delta_tt
                    elevation = self.calculate_elevation_groundStation_to_satellite(t_leave)
                    max_elevation = max(max_elevation,elevation)
                t_leave -= delta_tt
                overhead_time_minutes = t_leave - t_enter
                candidates.append((t_enter,overhead_time_minutes,max_elevation))
                t = t_leave
            t += delta_t

        # 2. 计算卫星在给定时间范围内的所有过顶时间
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def get_satellite_star_point_map_by_folium(self,start_time: datetime.datetime, num_samples: int, delta_t: datetime.timedelta, max_num_draw_points: int = 100, is_save_fig: bool = False, save_fig_path: str = None):
        satellite_geodetic_pos_seq = []
        for i in range(num_samples):
            timestamp = start_time + delta_t * i
            satellite_geodetic_pos = self.get_satellite_star_point(timestamp)
            satellite_geodetic_pos_seq.append(satellite_geodetic_pos)

        # Create a map centered around the first satellite position
        initial_position = satellite_geodetic_pos_seq[0]
        satellite_map = folium.Map(location=[initial_position[0], initial_position[1]], zoom_start=6)

        # Create a marker for the starting position of the satellite
        folium.Marker(location=[initial_position[0], initial_position[1]], 
                      popup=f'Start Position\nTime: {start_time}', 
                      icon=folium.Icon(color='green')).add_to(satellite_map)

        # Create a marker for the ending position of the satellite
        final_position = satellite_geodetic_pos_seq[-1]
        folium.Marker(location=[final_position[0], final_position[1]], 
                      popup=f'End Position\nTime: {start_time + delta_t * num_samples}', 
                      icon=folium.Icon(color='red')).add_to(satellite_map)
        
        # Create a marker for the ground station position
        folium.Marker(location=[self.groundStation.latitude_deg, self.groundStation.longitude_deg], 
                      popup='Ground Station', 
                      icon=folium.Icon(color='blue')).add_to(satellite_map)
        
        # Draw points directly on the map with a limit on the number of points
        step = max(1, len(satellite_geodetic_pos_seq) // max_num_draw_points)  # Calculate step size for even distribution
        for i in range(0, len(satellite_geodetic_pos_seq), step):
            pos = satellite_geodetic_pos_seq[i]
            folium.CircleMarker(location=[pos[0], pos[1]], radius=1, color='blue', fill=True, fill_color='blue', fill_opacity=0.6).add_to(satellite_map)

        if is_save_fig:
            # Create directory if it doesn't exist
            os.makedirs(save_fig_path, exist_ok=True)
            # Save the map
            satellite_map.save(os.path.join(save_fig_path, 'satellite_star_point_map.html'))

    def get_overhead_prediction_candidate_info(self, start_time: datetime.datetime, duration: datetime.timedelta, is_save_fig: bool = False, save_fig_path: str = None):
        delta_t = datetime.timedelta(seconds=1)
        t = start_time
        
        normalized_doppler_frequency_shift_seq = []
        elevation_seq = []
        
        while t < start_time + duration:
            normalized_doppler_frequency_shift = self.calculate_normalized_doppler_frequency_shift_by_ecef(t)
            elevation = self.calculate_elevation_groundStation_to_satellite(t)
            
            normalized_doppler_frequency_shift_seq.append(normalized_doppler_frequency_shift)
            elevation_seq.append(elevation)
            
            t += delta_t
        
        # Visualization of normalized_doppler_frequency_shift and elevation
        fig, ax1 = plt.subplots()
        
        # Plot normalized_doppler_frequency_shift
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Normalized Doppler Frequency Shift', color='tab:blue')
        ax1.plot([i for i in range(len(normalized_doppler_frequency_shift_seq))], normalized_doppler_frequency_shift_seq, color='tab:blue', label='Normalized Doppler Frequency Shift')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Create a second y-axis for elevation
        ax2 = ax1.twinx()
        ax2.set_ylabel('Elevation (degrees)', color='tab:red')
        ax2.plot([i for i in range(len(elevation_seq))], elevation_seq, color='tab:red', label='Elevation')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title('Normalized Doppler Frequency Shift and Elevation Over Time')
        fig.tight_layout()
        plt.grid()
        if is_save_fig:
            # Create directory if it doesn't exist
            os.makedirs(save_fig_path, exist_ok=True)
            candidate_info_path = os.path.join(save_fig_path, 'overhead_prediction_candidate_info.txt')
            with open(candidate_info_path, 'w') as f:
                f.write("Overhead Prediction Candidate Info\n")
                f.write("----------------------------------\n")
                f.write("Satellite Info\n")
                f.write(f"Satellite Name: {self.satelliteTLE['name']}\n")
                f.write(f"Satellite TLE Line 1: {self.satelliteTLE['TLE_line1']}\n")
                f.write(f"Satellite TLE Line 2: {self.satelliteTLE['TLE_line2']}\n")
                f.write("----------------------------------\n")
                f.write("Ground Station Info\n")
                f.write(f"Ground Station Name: {self.groundStation.name}\n")
                f.write(f"Ground Station Latitude: {self.groundStation.latitude_deg}\n")
                f.write(f"Ground Station Longitude: {self.groundStation.longitude_deg}\n")
                f.write(f"Ground Station Altitude: {self.groundStation.altitude_m}\n")
                f.write("----------------------------------\n")
                f.write("Overhead Prediction Candidate Info\n")
                f.write(f"Start Time: {start_time}\n")
                f.write(f"Duration: {duration}\n")
                
            # Save the plot
            plt.savefig(os.path.join(save_fig_path, 'normalized_doppler_frequency_shift_and_elevation_over_time.png'))
            # Save the map
            self.get_satellite_star_point_map_by_folium(start_time, duration // delta_t, delta_t, duration // delta_t, is_save_fig, save_fig_path)
        else:
            plt.show()
    
    def get_doppler_frequency_shift_sequence(self, start_time: datetime.datetime, signal_time_s: float, fs_Hz: int, fc_Hz: float, save_path: str = None):
        num_samples = int(signal_time_s * fs_Hz)
        doppler_shift_Hz_seq = numpy.zeros(num_samples, dtype=np.float64)
        for i in range(num_samples):
            if i % (num_samples // 10) == 0:
                print("doppler_shift_calc_progress: ", i/num_samples*100, "%")
            timestamp = datetime.datetime.fromtimestamp(start_time.timestamp() + i/fs_Hz)
            doppler_shift_Hz_seq[i] = self.calculate_normalized_doppler_frequency_shift_by_ecef(timestamp) * fc_Hz

        # Perform linear regression on the doppler_shift_Hz_seq
        x = range(num_samples)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, doppler_shift_Hz_seq)
        doppler_shift_Hz_seq_linear = slope * x + intercept

        time_seq = [i / fs_Hz for i in range(len(doppler_shift_Hz_seq))]
        regression_line = [slope * i + intercept for i in x]
        
        plt.figure(figsize=(10, 5))
        plt.plot(time_seq, doppler_shift_Hz_seq, label='Doppler Shift (Hz)', color='blue')
        plt.plot(time_seq, regression_line, label='Linear Regression', color='red', linestyle='--')

        plt.title('Doppler Shift Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Doppler Shift (Hz)')
        plt.grid()
        plt.legend(loc='upper left')  # Added legend location

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'doppler_frequency_shift.png'))

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            info_path = os.path.join(save_path, 'doppler_frequency_shift_info.txt')
            with open(info_path, 'w') as f:
                f.write("Doppler Frequency Shift Info\n")
                f.write("----------------------------------\n")
                f.write("Parameters\n")
                f.write(f"Start Time: {start_time}\n")
                f.write(f"Signal Time(s): {signal_time_s}\n")
                f.write(f"fs_Hz: {fs_Hz}\n")
                f.write(f"fc_Hz: {fc_Hz}\n")
                f.write("----------------------------------\n")
                f.write("Linear Regression Info\n")
                f.write(f"Slope: {slope}\n")
                f.write(f"Intercept: {intercept}\n")
                f.write(f"R-squared: {r_value}\n")
                f.write(f"P-value: {p_value}\n")
                f.write(f"Standard Error: {std_err}\n")
                

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, 'doppler_frequency_shift.npy'), doppler_shift_Hz_seq)

        return doppler_shift_Hz_seq

def eci2ecef_velocity(v_eci, time):
    """
    Convert velocity from ECI to ECEF using rotation matrix
    """
    gst = datetime2sidereal(time, 0.0)
    rot = np.array([[np.cos(gst), np.sin(gst), 0],
                   [-np.sin(gst), np.cos(gst), 0],
                   [0, 0, 1]])
    return rot @ v_eci


